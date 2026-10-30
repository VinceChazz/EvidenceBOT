"""EvidenceBot end-to-end pipeline.

retrieve -> standardise/deduplicate -> chunk -> hybrid rank -> evidence gate
   -> (abstain) | (grounded generation -> citation validation -> optional repair)
   -> evidence strength -> comparison table

The evidence gate is authoritative: when it returns UNANSWERABLE, no generation
call is made and the exact insufficient-evidence message is returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import INSUFFICIENT_EVIDENCE_MESSAGE, Settings
from evidence.citation_validator import (
    CitationRecord,
    CitationValidationReport,
    build_citation_records,
    validate_citations,
)
from evidence.evidence_evaluator import EvidenceAssessment, EvidenceEvaluator
from evidence.evidence_strength import EvidenceStrength, assess_evidence_strength
from evidence.stance import findings_table
from generation.extractive import build_extractive_digest
from generation.llm import LLMClient, LLMResult, build_llm_client
from generation.prompts import (
    SYSTEM_PROMPT,
    build_regeneration_prompt,
    build_user_prompt,
)
from processing.document_processor import ProcessedCorpus, process_documents
from processing.models import Passage, new_run_id
from ranking.embeddings import CachedEmbeddingService, build_embedding_service
from ranking.reranker import HybridRanker, RankingResult
from retrieval.retrieval_manager import RetrievalManager, RetrievalOutcome, build_default_manager
from utils.helpers import clean_multiline, clean_text
from utils.logging_config import get_logger

logger = get_logger("pipeline")


@dataclass
class EvidenceBotResult:
    """Everything the UI needs for one question."""

    question: str
    run_id: str
    retrieval: RetrievalOutcome
    corpus: ProcessedCorpus
    ranking: RankingResult
    assessment: EvidenceAssessment
    strength: EvidenceStrength
    answer: Optional[str] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)
    evidence_assessment_text: Optional[str] = None
    limitations: List[str] = field(default_factory=list)
    unanswered_questions: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    citations: List[CitationRecord] = field(default_factory=list)
    validation: Optional[CitationValidationReport] = None
    generation: Optional[LLMResult] = None
    generate_attempts: int = 0
    llm_error: Optional[str] = None
    generated_by_llm: bool = False
    extractive_digest: List[Dict[str, Any]] = field(default_factory=list)
    comparison: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False

    # ------------------------------------------------------------------
    @property
    def decision(self) -> str:
        return self.assessment.decision

    @property
    def is_abstention(self) -> bool:
        return self.assessment.is_abstention

    @property
    def is_partial(self) -> bool:
        return self.assessment.is_partial

    @property
    def abstention_message(self) -> Optional[str]:
        return INSUFFICIENT_EVIDENCE_MESSAGE if self.is_abstention else None

    @property
    def documents_retrieved(self) -> int:
        return self.corpus.document_count

    @property
    def passages_retrieved(self) -> int:
        return self.corpus.passage_count

    @property
    def relevant_passages(self) -> int:
        return len(self.ranking.relevant)

    @property
    def evidence_passages(self) -> List[Passage]:
        return self.assessment.supported_passages

    @property
    def citations_used(self) -> int:
        return len(self.citations)

    @property
    def sources_accessed(self) -> List[str]:
        return [r.source for r in self.retrieval.successes if r.documents]

    @property
    def source_failures(self) -> List[Dict[str, Any]]:
        return [r.summary() for r in self.retrieval.failures]

    def analytics(self) -> Dict[str, Any]:
        return {
            "documents_retrieved": self.documents_retrieved,
            "passages_retrieved": self.passages_retrieved,
            "candidates_ranked": self.ranking.stats.get("candidates", 0),
            "relevant_passages": self.relevant_passages,
            "evidence_passages_used": len(self.evidence_passages),
            "citations_used": self.citations_used,
            "sources_accessed": self.sources_accessed,
            "sources_failed": [f["source"] for f in self.source_failures],
            "retrieval_seconds": round(self.timings.get("retrieval", 0.0), 2),
            "ranking_seconds": round(self.timings.get("ranking", 0.0), 2),
            "generation_seconds": round(self.timings.get("generation", 0.0), 2),
            "total_seconds": round(self.timings.get("total", 0.0), 2),
            "top_relevance_score": self.ranking.top_score,
            "relevance_threshold": self.ranking.stats.get("threshold"),
            "embedding_backend": self.ranking.embedding_backend,
            "vector_backend": self.ranking.vector_backend,
            "decision": self.decision,
            "evidence_strength": self.strength.grade,
            "fallback_used": self.fallback_used,
            "generated_by_llm": self.generated_by_llm,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "run_id": self.run_id,
            "decision": self.decision,
            "message": self.abstention_message,
            "answer": self.answer,
            "claims": self.claims,
            "limitations": self.limitations,
            "unanswered_questions": self.unanswered_questions,
            "conflicts": self.conflicts,
            "assessment": self.assessment.to_dict(),
            "strength": self.strength.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "validation": self.validation.to_dict() if self.validation else None,
            "analytics": self.analytics(),
            "warnings": self.warnings,
        }


class EvidenceBot:
    """Orchestrates retrieval, evaluation, generation and verification."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_manager: Optional[RetrievalManager] = None,
        embedder: Optional[CachedEmbeddingService] = None,
        llm: Optional[LLMClient] = None,
        evaluator: Optional[EvidenceEvaluator] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.retrieval_manager = retrieval_manager or build_default_manager(self.settings)
        self.embedder = embedder or build_embedding_service(self.settings)
        self.llm = llm or build_llm_client(self.settings)
        self.evaluator = evaluator or EvidenceEvaluator(self.settings)

    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        sources: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        minimum_passages: Optional[int] = None,
        allow_generation: bool = True,
    ) -> EvidenceBotResult:
        """Run the full pipeline for one question."""

        started = time.perf_counter()
        question = clean_text(question)
        settings = self.settings
        top_k = top_k or settings.top_k
        threshold = settings.relevance_threshold if threshold is None else threshold
        minimum_passages = minimum_passages or settings.minimum_evidence_passages
        run_id = new_run_id()

        # ------------------------------------------------------- retrieval
        t0 = time.perf_counter()
        retrieval = self.retrieval_manager.retrieve(question, sources=sources)
        retrieval_seconds = time.perf_counter() - t0

        # ------------------------------------------- processing & chunking
        corpus = process_documents(retrieval.documents, settings)

        # ---------------------------------------------------------- ranking
        t0 = time.perf_counter()
        ranker = HybridRanker(settings, embedder=self.embedder)
        ranking = ranker.rank(
            question,
            corpus.passages,
            top_k=top_k,
            threshold=threshold,
        )
        ranking_seconds = time.perf_counter() - t0

        # ---------------------------------------------------- evidence gate
        evaluator = self.evaluator
        if minimum_passages != settings.minimum_evidence_passages:
            evaluator = EvidenceEvaluator(
                settings.with_overrides(minimum_evidence_passages=minimum_passages)
            )
        assessment = evaluator.evaluate(
            question,
            ranking.supporting,
            background_passages=ranking.background,
            ranking_stats=ranking.stats,
        )
        strength = assess_evidence_strength(assessment)

        result = EvidenceBotResult(
            question=question,
            run_id=run_id,
            retrieval=retrieval,
            corpus=corpus,
            ranking=ranking,
            assessment=assessment,
            strength=strength,
            warnings=list(retrieval.warnings),
            fallback_used=retrieval.fallback_used,
            comparison=findings_table(assessment.supported_passages, question),
            timings={"retrieval": retrieval_seconds, "ranking": ranking_seconds},
        )

        if retrieval.fallback_used:
            result.warnings.append(
                "Offline sample corpus in use — the records below are illustrative, "
                "not live literature."
            )

        # ------------------------------------------------------ abstention
        if assessment.is_abstention:
            logger.info("abstaining: %s (%s)", assessment.reason_code, assessment.reason)
            result.limitations = [assessment.reason]
            result.timings["generation"] = 0.0
            result.timings["total"] = time.perf_counter() - started
            return result

        # ---------------------------------------------------- evidence set
        evidence = list(assessment.supported_passages)[: min(top_k, 8)]
        result.citations = build_citation_records(evidence)

        if not allow_generation or not self.llm.is_available:
            result.llm_error = (
                None if allow_generation else "Answer generation disabled for this request."
            ) or self.llm.unavailable_reason
            result.extractive_digest = build_extractive_digest(question, evidence)
            result.warnings.append(
                f"Language model unavailable ({result.llm_error}). Showing an extractive "
                "evidence digest instead of a generated answer."
            )
            result.timings["generation"] = 0.0
            result.timings["total"] = time.perf_counter() - started
            return result

        # ------------------------------------------------------- generation
        partial = assessment.is_partial
        user_prompt = build_user_prompt(question, evidence, assessment, partial=partial)
        t0 = time.perf_counter()
        generation = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)
        attempts = 1
        result.generation = generation

        if not generation.ok:
            result.llm_error = generation.error
            result.extractive_digest = build_extractive_digest(question, evidence)
            result.warnings.append(
                f"Answer generation failed ({generation.error}). Showing an extractive "
                "evidence digest instead."
            )
            result.timings["generation"] = time.perf_counter() - t0
            result.timings["total"] = time.perf_counter() - started
            return result

        parsed = self._normalise_generated(generation.data)
        validation = validate_citations(
            parsed["answer"],
            evidence,
            settings,
            extra_claims=self._claims_with_markers(parsed["supporting_claims"]),
        )

        # ------------------------------------------------ repair if needed
        while (
            validation.needs_regeneration()
            and attempts < max(1, settings.max_generation_attempts)
        ):
            logger.info("citation validation failed, regenerating (attempt %d)", attempts + 1)
            repair_prompt = build_regeneration_prompt(
                parsed["answer"] + "\n" + "\n".join(c["claim"] for c in parsed["supporting_claims"]),
                validation.to_dict(),
                evidence,
            )
            retry = self.llm.generate_json(
                SYSTEM_PROMPT, repair_prompt, use_cache=False
            )
            attempts += 1
            result.generation = retry
            if not retry.ok:
                result.warnings.append(f"Regeneration failed: {retry.error}")
                break
            parsed = self._normalise_generated(retry.data)
            validation = validate_citations(
                parsed["answer"],
                evidence,
                settings,
                extra_claims=self._claims_with_markers(parsed["supporting_claims"]),
            )

        result.generate_attempts = attempts
        result.generated_by_llm = True
        result.validation = validation
        result.answer = validation.sanitized_answer or parsed["answer"]
        result.claims = parsed["supporting_claims"]
        result.evidence_assessment_text = parsed["evidence_assessment"]
        result.limitations = parsed["limitations"]
        result.unanswered_questions = parsed["unanswered_questions"]
        result.conflicts = parsed["conflicts"]

        if not validation.is_valid:
            result.warnings.append(
                "This answer could not be fully citation-verified; invalid or unsupported "
                "citations were removed. Treat it as unverified."
            )
        elif validation.uncited_claims:
            result.warnings.append(
                "Some statements in the answer are not individually cited; the answer is not "
                "presented as fully verified."
            )

        if partial and not result.unanswered_questions:
            result.unanswered_questions = list(assessment.unsupported_aspects)

        result.timings["generation"] = time.perf_counter() - t0
        result.timings["total"] = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _claims_with_markers(claims: Sequence[Dict[str, Any]]) -> List[str]:
        """Turn {"claim","source_ids"} into "claim [1,3]" for validation."""

        out: List[str] = []
        for item in claims:
            ids = [str(i) for i in item.get("source_ids") or [] if str(i).isdigit()]
            text = item.get("claim", "")
            if not text:
                continue
            out.append(f"{text} [{', '.join(ids)}]" if ids else text)
        return out

    @staticmethod
    def _normalise_generated(data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce the model's JSON into a predictable shape."""

        def _as_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [clean_text(value)] if clean_text(value) else []
            if isinstance(value, list):
                out: List[str] = []
                for item in value:
                    if isinstance(item, str):
                        text = clean_text(item)
                    elif isinstance(item, dict):
                        text = clean_text(
                            item.get("text")
                            or item.get("claim")
                            or item.get("description")
                            or ""
                        )
                    else:
                        text = clean_text(str(item))
                    if text:
                        out.append(text)
                return out
            return [clean_text(str(value))]

        claims_raw = data.get("supporting_claims") or []
        claims: List[Dict[str, Any]] = []
        if isinstance(claims_raw, list):
            for item in claims_raw:
                if isinstance(item, dict):
                    claim_text = clean_text(item.get("claim") or item.get("text") or "")
                    ids = item.get("source_ids") or item.get("citations") or []
                    if isinstance(ids, (int, str)):
                        ids = [ids]
                    parsed_ids: List[int] = []
                    for value in ids if isinstance(ids, list) else []:
                        digits = "".join(ch for ch in str(value) if ch.isdigit())
                        if digits:
                            parsed_ids.append(int(digits))
                    if claim_text:
                        claims.append({"claim": claim_text, "source_ids": parsed_ids})
                elif isinstance(item, str) and clean_text(item):
                    claims.append({"claim": clean_text(item), "source_ids": []})

        return {
            "answer": clean_multiline(str(data.get("answer") or "")),
            "supporting_claims": claims,
            "evidence_assessment": clean_text(
                str(data.get("evidence_assessment") or data.get("assessment") or "")
            ),
            "limitations": _as_list(data.get("limitations")),
            "unanswered_questions": _as_list(
                data.get("unanswered_questions") or data.get("unanswered")
            ),
            "conflicts": _as_list(data.get("conflicts") or data.get("conflicting_findings")),
        }


def ask(question: str, settings: Optional[Settings] = None, **kwargs) -> EvidenceBotResult:
    """Convenience one-shot API (used by scripts and smoke tests)."""

    return EvidenceBot(settings).ask(question, **kwargs)
