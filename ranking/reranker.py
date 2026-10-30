"""Hybrid retrieval and reranking.

Score = dense semantic similarity (embeddings + FAISS) blended with BM25 lexical
relevance, plus a concept-coverage term. The resulting ``relevance_score`` is a
*retrieval* measurement only: it says how well a passage matches the question,
never whether the science in it is correct or sufficient. Sufficiency is decided
by the evidence evaluator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import Settings
from processing.models import Passage
from ranking.bm25 import BM25Index
from ranking.embeddings import CachedEmbeddingService, build_embedding_service, cosine_matrix
from ranking.vector_store import BaseVectorStore, create_vector_store
from utils.helpers import (
    question_concepts,
    split_question_components,
    unique_terms,
)
from utils.logging_config import get_logger

logger = get_logger("ranking.reranker")


@dataclass
class RankingResult:
    passages: List[Passage] = field(default_factory=list)
    relevant: List[Passage] = field(default_factory=list)
    supporting: List[Passage] = field(default_factory=list)
    background: List[Passage] = field(default_factory=list)
    rejected: List[Passage] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    embedding_backend: str = "unknown"
    vector_backend: str = "unknown"
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return self.stats.get("candidates", 0)  # type: ignore[return-value]

    @property
    def relevant_count(self) -> int:
        return len(self.relevant)

    @property
    def supporting_count(self) -> int:
        return len(self.supporting)

    @property
    def top_score(self) -> float:
        return self.passages[0].relevance_score if self.passages else 0.0

    def to_rows(self) -> List[Dict[str, object]]:
        rows = []
        for rank, passage in enumerate(self.passages, start=1):
            rows.append(
                {
                    "rank": rank,
                    "passage_id": passage.passage_id,
                    "title": passage.title,
                    "source": passage.source_name,
                    "source_type": passage.source_type,
                    "year": passage.publication_year,
                    "dense": round(passage.dense_score, 3),
                    "bm25": round(passage.bm25_score, 3),
                    "lexical": round(passage.lexical_score, 3),
                    "relevance": round(passage.relevance_score, 3),
                    "used_as_evidence": passage.relevance_score
                    >= (self.stats.get("threshold") or -1),
                }
            )
        return rows


class HybridRanker:
    """Combines embeddings/FAISS and BM25, then reranks the candidates."""

    def __init__(
        self,
        settings: Settings,
        embedder: Optional[CachedEmbeddingService] = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedding_service(settings)

    # ------------------------------------------------------------------
    def _hybrid_scores(
        self,
        question: str,
        passages: Sequence[Passage],
        dense: np.ndarray,
        bm25: np.ndarray,
        lexical: np.ndarray,
    ) -> np.ndarray:
        calibrated = np.array([self.embedder.calibrate(float(c)) for c in dense], dtype=np.float32)
        hybrid = self.settings.dense_weight * calibrated + self.settings.bm25_weight * bm25
        combined = 0.8 * hybrid + 0.2 * lexical
        return np.clip(combined, 0.0, 1.0)

    # ------------------------------------------------------------------
    def rank(
        self,
        question: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        candidate_k: Optional[int] = None,
    ) -> RankingResult:
        started = time.perf_counter()
        top_k = top_k or self.settings.top_k
        threshold = self.settings.relevance_threshold if threshold is None else threshold
        candidate_k = candidate_k or self.settings.candidate_k

        result = RankingResult(embedding_backend=self.embedder.name)

        if not passages:
            result.elapsed_seconds = time.perf_counter() - started
            result.stats = {
                "candidates": 0,
                "threshold": threshold,
                "embedding_backend": self.embedder.name,
                "vector_backend": "n/a",
            }
            return result

        texts = [p.text for p in passages]
        matrix = self.embedder.embed_passages(texts)
        store: BaseVectorStore = create_vector_store(
            matrix, [{"passage_id": p.passage_id} for p in passages]
        )
        query_vector = self.embedder.embed_query(question)
        dense = store.scores(query_vector)
        if dense.shape[0] != len(passages):
            dense = cosine_matrix(query_vector, matrix)

        bm25_index = BM25Index(texts)
        bm25 = bm25_index.scores(question)
        if bm25.shape[0] != len(passages):  # defensive: keep arrays aligned
            bm25 = np.zeros((len(passages),), dtype=np.float32)

        # Concept coverage is evaluated per question component, so a passage that
        # thoroughly covers one part of a compound question is not penalised for the
        # rest (the evidence gate decides whether all parts are covered).
        component_concepts = [
            set(question_concepts(component))
            for component in split_question_components(question)
        ] or [set(question_concepts(question))]
        component_concepts = [c for c in component_concepts if c] or [set()]
        lexical_rows = []
        for text in texts:
            terms = unique_terms(text)
            coverage = [
                len(concepts & terms) / len(concepts) if concepts else 0.0
                for concepts in component_concepts
            ]
            lexical_rows.append(max(coverage) if coverage else 0.0)
        lexical = np.array(lexical_rows, dtype=np.float32)

        combined = self._hybrid_scores(question, passages, dense, bm25, lexical)
        order = np.argsort(-combined)
        candidates = [passages[int(i)] for i in order[: max(candidate_k, top_k)]]

        # Attach scores to the passages (mutating the shared Passage objects).
        for i, passage in enumerate(passages):
            passage.stance = "unclear"
            passage.dense_score = float(dense[i])
            passage.bm25_score = float(bm25[i])
            passage.lexical_score = float(lexical[i])
            passage.relevance_score = float(combined[i])

        # Diversity: cap chunks taken from the same publication.
        selected: List[Passage] = []
        per_document: Dict[str, int] = {}
        for passage in candidates:
            used = per_document.get(passage.document_id, 0)
            if used >= 2 and len(candidates) > top_k:
                continue
            selected.append(passage)
            per_document[passage.document_id] = used + 1
            if len(selected) >= top_k:
                break

        relevant = [p for p in selected if p.relevance_score >= threshold]
        on_topic = [
            p
            for p in relevant
            if p.lexical_score >= self.settings.min_term_overlap
        ]
        supporting = [p for p in on_topic if not p.is_background]
        background = [p for p in on_topic if p.is_background]
        relevant_ids = {id(p) for p in relevant}
        rejected = [p for p in selected if id(p) not in relevant_ids]

        result.passages = selected
        result.relevant = relevant
        result.supporting = supporting
        result.background = background
        result.rejected = rejected
        result.vector_backend = store.backend
        result.elapsed_seconds = time.perf_counter() - started
        result.stats = {
            "candidates": len(candidates),
            "passages_total": len(passages),
            "relevant": len(relevant),
            "supporting": len(supporting),
            "background": len(background),
            "threshold": threshold,
            "top_k": top_k,
            "embedding_backend": self.embedder.name,
            "vector_backend": store.backend,
            "bm25_backend": bm25_index.backend,
            "top_score": round(float(combined[order[0]]), 3) if len(combined) else 0.0,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
        }
        logger.info(
            "ranking: %d candidates -> %d relevant (%d supporting) in %.2fs [%s/%s]",
            len(candidates),
            len(relevant),
            len(supporting),
            result.elapsed_seconds,
            self.embedder.name,
            store.backend,
        )
        return result


def rank_passages(
    question: str,
    passages: Sequence[Passage],
    settings: Settings,
    embedder: Optional[CachedEmbeddingService] = None,
    **kwargs,
) -> RankingResult:
    ranker = HybridRanker(settings, embedder=embedder)
    return ranker.rank(question, passages, **kwargs)
