"""In-memory vector store (FAISS with a NumPy fallback).

The store is built per query from the retrieved passages; nothing is persisted,
which keeps the pipeline stateless and avoids stale embeddings from earlier runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.logging_config import get_logger

logger = get_logger("ranking.vector_store")

try:  # pragma: no cover - exercised implicitly when faiss is installed
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


@dataclass
class VectorHit:
    index: int
    score: float
    metadata: Dict[str, Any]


class BaseVectorStore:
    """Interface used by the hybrid ranker."""

    backend = "base"

    def __init__(self, vectors: np.ndarray, metadata: Optional[Sequence[Dict[str, Any]]] = None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.vectors = vectors
        self.metadata = list(metadata or [{} for _ in range(len(vectors))])

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, query: np.ndarray, k: int = 10) -> List[VectorHit]:
        raise NotImplementedError

    def scores(self, query: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NumpyVectorStore(BaseVectorStore):
    """Exact inner-product search (vectors are L2-normalised => cosine)."""

    backend = "numpy"

    def scores(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32).ravel()
        if self.vectors.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: query {query.shape[0]} vs index {self.vectors.shape[1]}"
            )
        return self.vectors @ query

    def search(self, query: np.ndarray, k: int = 10) -> List[VectorHit]:
        sims = self.scores(query)
        if sims.size == 0:
            return []
        k = max(1, min(k, sims.size))
        order = np.argsort(-sims)[:k]
        return [
            VectorHit(index=int(i), score=float(sims[i]), metadata=self.metadata[int(i)])
            for i in order
        ]


class FaissVectorStore(BaseVectorStore):
    """FAISS ``IndexFlatIP`` exact search over normalised vectors."""

    backend = "faiss"

    def __init__(self, vectors: np.ndarray, metadata: Optional[Sequence[Dict[str, Any]]] = None):
        super().__init__(vectors, metadata)
        if len(self) == 0:
            self.index = None
            return
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])  # type: ignore[union-attr]
        self.index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))

    def scores(self, query: np.ndarray) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0,), dtype=np.float32)
        query = np.asarray(query, dtype=np.float32).ravel().reshape(1, -1)
        scores, indices = self.index.search(np.ascontiguousarray(query), len(self))  # type: ignore[union-attr]
        out = np.full((len(self),), -1.0, dtype=np.float32)
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                out[int(idx)] = float(score)
        return out

    def search(self, query: np.ndarray, k: int = 10) -> List[VectorHit]:
        if len(self) == 0:
            return []
        k = max(1, min(k, len(self)))
        query = np.asarray(query, dtype=np.float32).ravel().reshape(1, -1)
        scores, indices = self.index.search(  # type: ignore[union-attr]
            np.ascontiguousarray(query), k
        )
        hits: List[VectorHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(
                VectorHit(index=int(idx), score=float(score), metadata=self.metadata[int(idx)])
            )
        return hits


def create_vector_store(
    vectors: np.ndarray,
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    prefer_faiss: bool = True,
) -> BaseVectorStore:
    """Create a FAISS store when available, else the NumPy fallback."""

    if prefer_faiss and FAISS_AVAILABLE and len(vectors) > 0:
        try:
            return FaissVectorStore(vectors, metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS store failed (%s); using NumPy backend", exc)
    return NumpyVectorStore(vectors, metadata)


def backend_status() -> Dict[str, Any]:
    return {"faiss_available": FAISS_AVAILABLE, "faiss_version": getattr(faiss, "__version__", None)}
