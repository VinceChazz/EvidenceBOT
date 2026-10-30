"""Evidence evaluation and the abstention gate.

This module is the heart of EvidenceBot and is deliberately independent of the
LLM: it decides, from retrieval measurements and transparent lexical rules,
whether the retrieved passages can support an answer at all.

Four dimensions are evaluated:

1. **Retrieval relevance** — are the passages about the question?
2. **Evidence sufficiency** — is there enough usable content, from enough
   sources, covering what the question actually asks?
3. **Source quality** — peer-reviewed article vs. review vs. preprint vs.
   background material.
4. **Evidence consistency** — do the sources agree, disagree or report null /
   inconclusive findings?

The output :class:`EvidenceAssessment` carries a binding decision:
``ANSWERABLE``, ``PARTIALLY_ANSWERABLE`` or ``UNANSWERABLE``. When the decision is
``UNANSWERABLE`` the generation LLM is never called and the exact message
"Insufficient evidence. No reliable sources found in the retrieved passages." is
returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import INSUFFICIENT_EVIDENCE_MESSAGE, Settings
from evidence.abstention import (
    ANSWERABLE,
    PARTIALLY_ANSWERABLE,
    UNANSWERABLE,
    AbstentionDecision,
    abstain,
)
from evidence.stance import ConsistencyResult, assess_consistency, detect_stance
from processing.models import Passage
from utils.helpers import (
    ABSOLUTE_CLAIM_CUES,
    clean_text,
    question_concepts,
    split_question_components,
    unique_terms,
)
from utils.logging_config import get_logger

logger = get_logger("evidence.evaluator")

#: Universal/absolute claims that essentially no study design can establish.
STRICT_ABSOLUTE_CUES = {
    "guarantee", "guarantees", "guaranteed", "never", "always", "100%",
    "impossible", "prove", "proves", "proof", "definitively", "undoubtedly",
    "certainly", "eliminate", "eliminates", "completely", "totally",
    "universally",
}

#: Causal-claim cues that require stronger (interventional) evidence.
CAUSAL_CUES = {
    "cause", "causes", "caused", "causing", "lead", "leads", "drive", "drives",
    "promote", "promotes", "prevent", "prevents", "cure", "cures", "treat",
    "treats",
}

#: Absolute floor used when a custom Settings object does not define one.
MIN_EVIDENCE_WORDS = 45


@dataclass
class Criterion:
    """One transparent evaluation criterion."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "explanation": self.explanation,
        }


@dataclass
class AspectCoverage:
    """Coverage of one question component by the retrieved evidence."""

    component: str
    concepts: List[str]
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def supported(self) -> bool:
        return self.coverage >= 0.5 and bool(self.concepts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "coverage": round(self.coverage, 3),
            "covered": self.covered,
            "missing": self.missing,
            "supported": self.supported,
        }


@dataclass
class EvidenceAssessment:
    """Complete, auditable result of the evidence stage."""

    decision: str
    reason_code: str
    reason: str
    abstention: AbstentionDecision
    relevance: Dict[str, Any] = field(default_factory=dict)
    sufficiency: Dict[str, Any] = field(default_factory=dict)
    source_quality: Dict[str, Any] = field(default_factory=dict)
    consistency: ConsistencyResult = field(default_factory=ConsistencyResult)
    aspects: List[AspectCoverage] = field(default_factory=list)
    criteria: List[Criterion] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    absolute_claim: bool = False
    absolute_cues: List[str] = field(default_factory=list)
    causal_claim: bool = False
    supported_passages: List[Passage] = field(default_factory=list)
    background_passages: List[Passage] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def message(self) -> Optional[str]:
        return self.abstention.message

    @property
    def is_abstention(self) -> bool:
        return self.decision == UNANSWERABLE

    @property
    def is_partial(self) -> bool:
        return self.decision == PARTIALLY_ANSWERABLE

    @property
    def supported_aspects(self) -> List[str]:
        return [a.component for a in self.aspects if a.supported]

    @property
    def unsupported_aspects(self) -> List[str]:
        return [a.component for a in self.aspects if not a.supported]

    def criteria_frame_rows(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.criteria]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "message": self.message,
            "relevance": self.relevance,
            "sufficiency": self.sufficiency,
            "source_quality": self.source_quality,
            "consistency": {
                "conclusion": self.consistency.conclusion,
                "label": self.consistency.label,
                "stance_counts": self.consistency.stance_counts,
                "agreement_ratio": round(self.consistency.agreement_ratio, 3),
                "notes": self.consistency.notes,
            },
            "aspects": [a.to_dict() for a in self.aspects],
            "criteria": [c.to_dict() for c in self.criteria],
            "notes": self.notes,
            "absolute_claim": self.absolute_claim,
            "absolute_cues": self.absolute_cues,
            "causal_claim": self.causal_claim,
        }


def _coverage_of_component(component: str, passages: Sequence[Passage]) -> AspectCoverage:
    concepts = question_concepts(component)
    if not concepts:
        return AspectCoverage(component=component, concepts=[], coverage=0.0)
    union: set = set()
    for passage in passages:
        union |= unique_terms(passage.text)
    covered = [c for c in concepts if c in union]
    missing = [c for c in concepts if c not in union]
    return AspectCoverage(
        component=component,
        concepts=concepts,
        covered=covered,
        missing=missing,
        coverage=len(covered) / len(concepts),
    )


class EvidenceEvaluator:
    """Deterministic, independently testable evidence gate."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        question: str,
        supporting_passages: Sequence[Passage],
        background_passages: Optional[Sequence[Passage]] = None,
        ranking_stats: Optional[Dict[str, Any]] = None,
    ) -> EvidenceAssessment:
        settings = self.settings
        ranking_stats = ranking_stats or {}
        background_passages = list(background_passages or [])
        passages = [
            p for p in supporting_passages if not p.is_background
        ]
        criteria: List[Criterion] = []
        notes: List[str] = []

        absolute_cues = sorted({c for c in ABSOLUTE_CLAIM_CUES if c in question.lower()})
        strict_absolute = sorted({c for c in STRICT_ABSOLUTE_CUES if c in question.lower()})
        causal_cues = sorted({c for c in CAUSAL_CUES if c in question.lower()})

        # ---------------------------------------------------------- relevance
        top_score = max((p.relevance_score for p in passages), default=0.0)
        mean_score = (
            sum(p.relevance_score for p in passages) / len(passages) if passages else 0.0
        )
        relevance = {
            "top_score": round(top_score, 4),
            "mean_score": round(mean_score, 4),
            "passages": len(passages),
            "threshold": settings.relevance_threshold,
            "background_passages": len(background_passages),
        }
        relevance_passed = bool(passages) and top_score >= settings.relevance_threshold
        criteria.append(
            Criterion(
                name="Retrieval relevance",
                passed=relevance_passed,
                value=round(top_score, 3),
                threshold=settings.relevance_threshold,
                explanation=(
                    "Best hybrid relevance score of the retrieved scientific passages "
                    "(retrieval measurement, not a probability of correctness)."
                ),
            )
        )

        # --------------------------------------------------------- sufficiency
        min_words = getattr(settings, "min_evidence_words", MIN_EVIDENCE_WORDS)
        total_words = sum(len(clean_text(p.text).split()) for p in passages)
        sufficient_count = len(passages) >= settings.minimum_evidence_passages
        sufficient_content = total_words >= min_words
        criteria.append(
            Criterion(
                name="Minimum evidence passages",
                passed=sufficient_count,
                value=len(passages),
                threshold=settings.minimum_evidence_passages,
                explanation="Number of on-topic scientific passages above the relevance threshold.",
            )
        )
        criteria.append(
            Criterion(
                name="Evidence content volume",
                passed=sufficient_content,
                value=total_words,
                threshold=min_words,
                explanation="Total words of usable evidence (abstracts/full text) retrieved.",
            )
        )

        # ------------------------------------------------------ source quality
        peer_reviewed = [p for p in passages if p.is_peer_reviewed and not p.is_preprint]
        preprints = [p for p in passages if p.is_preprint or not p.is_peer_reviewed]
        review_like = [
            p
            for p in passages
            if p.source_type in {"review", "systematic_review", "meta_analysis"}
        ]
        interventional = [
            p
            for p in passages
            if p.source_type
            in {"randomized_controlled_trial", "clinical_trial", "meta_analysis", "systematic_review"}
        ]
        full_text = [p for p in passages if p.provenance == "abstract+full_text"]
        source_quality = {
            "peer_reviewed": len(peer_reviewed),
            "preprints_or_unreviewed": len(preprints),
            "review_like": len(review_like),
            "interventional_or_synthesis": len(interventional),
            "full_text_available": len(full_text),
            "background_excluded": len(background_passages),
            "peer_reviewed_present": bool(peer_reviewed),
        }
        criteria.append(
            Criterion(
                name="Source quality",
                passed=bool(peer_reviewed),
                value=len(peer_reviewed),
                threshold=">= 1 peer-reviewed source",
                explanation=(
                    "Preprints and background material are down-weighted; only "
                    "peer-reviewed records fully satisfy this criterion."
                ),
            )
        )
        if passages and not peer_reviewed:
            notes.append(
                "Only preprint or non-peer-reviewed sources support this question; "
                "treat the findings as preliminary."
            )
        if background_passages:
            notes.append(
                f"{len(background_passages)} background (encyclopaedic) passage(s) were "
                "retrieved but excluded from the evidence decision."
            )

        # --------------------------------------------------------- consistency
        consistency = assess_consistency(passages, question)
        if consistency.is_conflicting:
            notes.append(
                "The retrieved studies disagree. The answer must report the conflict "
                "rather than a single established conclusion."
            )

        # ------------------------------------------------------------- aspects
        components = split_question_components(question)
        aspects = [_coverage_of_component(c, passages) for c in components]
        supported_aspects = [a for a in aspects if a.supported]
        coverage = (
            sum(a.coverage for a in aspects) / len(aspects) if aspects else 0.0
        )
        sufficiency = {
            "passages": len(passages),
            "required_passages": settings.minimum_evidence_passages,
            "words": total_words,
            "question_coverage": round(coverage, 3),
            "minimum_question_coverage": settings.min_question_coverage,
            "supported_aspects": [a.component for a in supported_aspects],
            "unsupported_aspects": [a.component for a in aspects if not a.supported],
        }

        # ------------------------------------------- direct null evidence check
        direct_null_support = any(
            detect_stance(p.text, question).stance == "null"
            and _coverage_of_component(question, [p]).coverage >= 0.5
            for p in passages
        )
        criteria.append(
            Criterion(
                name="Absolute claim check",
                passed=not strict_absolute,
                value=strict_absolute or "none",
                threshold="no universal/guarantee wording",
                explanation=(
                    "Universal claims ('guarantee', 'never', '100%') cannot be established "
                    "by observational or trial evidence of finite scope."
                ),
            )
        )

        # ---------------------------------------------------------- decisions
        abstention: AbstentionDecision

        if not relevance_passed:
            abstention = abstain(
                "no_relevant_passages",
                "No retrieved passage cleared the relevance threshold for this question.",
                ["Retrieval relevance"],
            )
        elif (
            strict_absolute
            and settings.absolute_claim_policy == "abstain"
            and not direct_null_support
        ):
            abstention = abstain(
                "absolute_claim_not_supported",
                (
                    "The question asks for a universal guarantee ("
                    + ", ".join(strict_absolute)
                    + "). No retrieved study can establish an absolute claim of that kind."
                ),
                ["Absolute claim check"],
            )
        elif not sufficient_count:
            abstention = abstain(
                "insufficient_passages",
                (
                    f"Only {len(passages)} on-topic passage(s) were retrieved; "
                    f"EvidenceBot requires at least {settings.minimum_evidence_passages}."
                ),
                ["Minimum evidence passages"],
            )
        elif not sufficient_content:
            abstention = abstain(
                "insufficient_content",
                f"Only {total_words} words of usable evidence were retrieved.",
                ["Evidence content volume"],
            )
        elif not supported_aspects:
            abstention = abstain(
                "question_not_addressed",
                (
                    "The retrieved passages are topically related but do not address the "
                    "substance of the question."
                ),
                ["Question coverage"],
            )
        elif strict_absolute and settings.absolute_claim_policy == "abstain":
            # Direct null evidence exists for the exposure/outcome, so the claim can
            # be addressed — but only partially (the absolute wording itself is
            # unsupported).
            abstention = AbstentionDecision(
                decision=PARTIALLY_ANSWERABLE,
                reason_code="absolute_claim_partial",
                reason=(
                    "Retrieved studies address the exposure and outcome but cannot "
                    "establish the absolute wording of the question."
                ),
                criteria_failed=["Absolute claim check"],
            )
        elif len(supported_aspects) < len(aspects):
            abstention = AbstentionDecision(
                decision=PARTIALLY_ANSWERABLE,
                reason_code="partial_coverage",
                reason=(
                    "Evidence covers part of the question but not: "
                    + "; ".join(a.component for a in aspects if not a.supported)
                ),
                criteria_failed=["Question coverage"],
            )
        else:
            abstention = AbstentionDecision(
                decision=ANSWERABLE,
                reason_code="evidence_sufficient",
                reason="Sufficient, relevant and comparable evidence was retrieved.",
            )

        # A causal question backed only by observational designs is weakened,
        # but this is a quality caveat rather than an abstention trigger.
        if causal_cues and passages and not interventional:
            notes.append(
                "The question asks about a causal relationship while the retrieved "
                "evidence is observational only; causality cannot be established."
            )

        if consistency.conclusion in {"conflicting", "mixed"}:
            notes.append(
                "Conflicting or mixed findings detected across the retrieved studies."
            )

        criteria.append(
            Criterion(
                name="Question coverage",
                passed=bool(supported_aspects) and len(supported_aspects) == len(aspects),
                value=f"{len(supported_aspects)}/{len(aspects)} components",
                threshold=settings.min_question_coverage,
                explanation=(
                    "Share of each question component's key concepts that appear in the "
                    "retrieved evidence."
                ),
            )
        )

        assessment = EvidenceAssessment(
            decision=abstention.decision,
            reason_code=abstention.reason_code,
            reason=abstention.reason,
            abstention=abstention,
            relevance=relevance,
            sufficiency=sufficiency,
            source_quality=source_quality,
            consistency=consistency,
            aspects=aspects,
            criteria=criteria,
            notes=notes,
            absolute_claim=bool(absolute_cues),
            absolute_cues=absolute_cues,
            causal_claim=bool(causal_cues),
            supported_passages=list(passages),
            background_passages=list(background_passages),
        )

        logger.info(
            "evidence decision=%s reason=%s (passages=%d, top=%.3f, consistency=%s)",
            assessment.decision,
            assessment.reason_code,
            len(passages),
            top_score,
            consistency.conclusion,
        )
        return assessment


def evaluate_evidence(
    question: str,
    supporting_passages: Sequence[Passage],
    background_passages: Optional[Sequence[Passage]] = None,
    settings: Optional[Settings] = None,
    ranking_stats: Optional[Dict[str, Any]] = None,
) -> EvidenceAssessment:
    """Functional wrapper around :class:`EvidenceEvaluator`."""

    return EvidenceEvaluator(settings).evaluate(
        question,
        supporting_passages,
        background_passages=background_passages,
        ranking_stats=ranking_stats,
    )


def abstention_message() -> str:
    return INSUFFICIENT_EVIDENCE_MESSAGE
