"""Embedding providers.

Two backends are available:

* ``openai`` — the OpenAI embeddings API (default whenever a key is configured).
* ``local``  — a deterministic hashed-feature embedder (word, stem, char-ngram
  and bigram features). It is not a neural model: it exists so the pipeline is
  fully functional without a paid key and so the test-suite never needs network.

Both expose the same interface and a ``calibration`` range used to map raw
cosine similarity into a comparable 0-1 relevance scale.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import Settings
from utils.cache import TTLCache
from utils.helpers import clean_text, simple_stem, stable_hash, tokenize
from utils.logging_config import get_logger

logger = get_logger("ranking.embeddings")


class EmbeddingProvider(ABC):
    """Common interface for embedding backends."""

    name = "base"
    #: Raw cosine similarities below/above these values map to 0/1.
    calibration: Tuple[float, float] = (0.0, 1.0)

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) L2-normalised float32 matrix."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def calibrate(self, cosine: float) -> float:
        low, high = self.calibration
        if high <= low:
            return max(0.0, min(1.0, cosine))
        return max(0.0, min(1.0, (cosine - low) / (high - low)))


class LocalHashingEmbeddings(EmbeddingProvider):
    """Deterministic feature-hashing embedder (no network, no cost)."""

    name = "local-hashing"
    #: Empirical range for this embedder on scientific abstracts.
    calibration = (0.06, 0.55)

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _hash(feature: str) -> Tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little")
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def _features(self, text: str) -> List[Tuple[str, float]]:
        text = clean_text(text).lower()
        tokens = [t for t in tokenize(text) if len(t) > 1]
        features: List[Tuple[str, float]] = []
        for token in tokens:
            features.append((f"w:{token}", 1.0))
            stem = simple_stem(token)
            if stem != token:
                features.append((f"s:{stem}", 0.7))
        for first, second in zip(tokens, tokens[1:]):
            features.append((f"b:{first}_{second}", 0.5))
        compact = re.sub(r"\s+", " ", text)
        for i in range(0, max(0, len(compact) - 3)):
            gram = compact[i : i + 4]
            if gram.strip():
                features.append((f"c:{gram}", 0.18))
        return features

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                index, sign = self._hash(feature)
                matrix[row, index % self._dimension] += sign * weight
        matrix = np.sqrt(np.abs(matrix)) * np.sign(matrix)
        return self._normalise(matrix)


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings with batching, retries and an on-disk cache."""

    name = "openai"

    def __init__(
        self,
        settings: Settings,
        cache: Optional[TTLCache] = None,
        client: Optional[object] = None,
    ) -> None:
        from openai import OpenAI

        self.settings = settings
        self.model = settings.openai_embedding_model
        self.batch_size = max(1, settings.embedding_batch_size)
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        # text-embedding-3-* on scientific sentences typically cosines 0.25-0.85.
        self.calibration = (0.20, 0.80)
        self.cache = cache or TTLCache(
            "embeddings",
            ttl_seconds=max(settings.cache_ttl_seconds, 86400),
            enabled=settings.cache_enabled,
        )
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        # Known output sizes for the text-embedding-3 family; resolved from the
        # first API response otherwise.
        if self._dimension is None:
            self._dimension = 3072 if "large" in self.model else 1536
        return self._dimension

    def _cache_key(self, text: str) -> str:
        return f"{self.model}|{stable_hash(text)}"

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = [clean_text(t) or " " for t in texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        pending: List[Tuple[int, str]] = []
        for idx, text in enumerate(texts):
            cached = self.cache.get(self._cache_key(text))
            if cached is not None:
                vectors[idx] = np.asarray(cached, dtype=np.float32)
            else:
                pending.append((idx, text))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model, input=[text for _, text in batch]
            )
            for (idx, text), item in zip(batch, response.data):
                vector = np.asarray(item.embedding, dtype=np.float32)
                vectors[idx] = vector
                self.cache.set(self._cache_key(text), [float(v) for v in vector])
            logger.debug("embedded %d texts with %s", len(batch), self.model)

        matrix = np.vstack([v if v is not None else np.zeros(self.dimension, dtype=np.float32)
                            for v in vectors])
        self._dimension = matrix.shape[1]
        return self._normalise(matrix)


class CachedEmbeddingService:
    """Thin wrapper that adds query caching, aligned embedding and degradation."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self._query_cache = TTLCache("query_embeddings", ttl_seconds=3600, persist=False)
        self.degraded = False
        self.degraded_reason: Optional[str] = None

    # ------------------------------------------------------------------
    def _degrade(self, exc: Exception) -> None:
        """Switch to the offline embedder if the API backend fails mid-run."""

        if self.degraded:
            return
        self.degraded = True
        self.degraded_reason = f"{type(exc).__name__}: {exc}"[:180]
        self.provider = LocalHashingEmbeddings()
        self._query_cache.clear()
        logger.warning(
            "embedding backend failed (%s); continuing with the local hashing embedder",
            self.degraded_reason,
        )

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        try:
            return self.provider.embed_texts(list(texts))
        except Exception as exc:  # noqa: BLE001 - never fail the whole pipeline
            self._degrade(exc)
            return self.provider.embed_texts(list(texts))

    def embed_query(self, question: str) -> np.ndarray:
        key = stable_hash(self.provider.name, question)
        cached = self._query_cache.get(key)
        if cached is not None and len(cached) == self.dimension:
            return np.asarray(cached, dtype=np.float32)
        try:
            vector = self.provider.embed_texts([question])[0]
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            vector = self.provider.embed_texts([question])[0]
        self._query_cache.set(key, [float(v) for v in vector])
        return vector

    def calibrate(self, cosine: float) -> float:
        return self.provider.calibrate(cosine)

    @property
    def is_semantic_model(self) -> bool:
        return self.provider.name == "openai"


def build_embedding_service(
    settings: Settings, force_backend: Optional[str] = None, client: Optional[object] = None
) -> CachedEmbeddingService:
    """Create the embedding service appropriate for the current settings."""

    backend = (force_backend or settings.resolved_embedding_backend()).lower()
    if backend == "openai":
        if not settings.openai_api_key:
            logger.warning(
                "EMBEDDING_BACKEND=openai but no OPENAI_API_KEY is set — "
                "falling back to the local hashing embedder."
            )
            return CachedEmbeddingService(LocalHashingEmbeddings())
        try:
            return CachedEmbeddingService(OpenAIEmbeddings(settings, client=client))
        except Exception as exc:  # noqa: BLE001 - never hard-fail on embeddings
            logger.warning("OpenAI embeddings unavailable (%s); using local embedder", exc)
            return CachedEmbeddingService(LocalHashingEmbeddings())
    return CachedEmbeddingService(LocalHashingEmbeddings())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_norm = float(np.linalg.norm(query)) or 1.0
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return (matrix @ query) / (norms * query_norm)
