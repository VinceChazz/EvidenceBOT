"""BM25 keyword retrieval over the retrieved passages."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.helpers import content_terms, tokenize
from utils.logging_config import get_logger

logger = get_logger("ranking.bm25")

try:
    from rank_bm25 import BM25Okapi

    RANK_BM25_AVAILABLE = True
except Exception:  # noqa: BLE001 - fall back to the built-in implementation
    BM25Okapi = None  # type: ignore
    RANK_BM25_AVAILABLE = False


class SimpleBM25:
    """Compact Okapi BM25 implementation (used when rank_bm25 is absent)."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(doc) for doc in corpus_tokens]
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.corpus)
        self.avgdl = (
            sum(len(doc) for doc in self.corpus) / self.doc_count if self.doc_count else 0.0
        )
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        df: Dict[str, int] = {}
        for doc in self.corpus:
            freqs: Dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1
        for token, count in df.items():
            self.idf[token] = math.log(1 + (self.doc_count - count + 0.5) / (count + 0.5))

    def get_scores(self, query: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for idx, freqs in enumerate(self.doc_freqs):
            doc_len = len(self.corpus[idx]) or 1
            score = 0.0
            for token in query:
                if token not in freqs:
                    continue
                freq = freqs[token]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(token, 0.0) * freq * (self.k1 + 1) / (denom or 1)
            scores.append(score)
        return scores


class BM25Index:
    """BM25 index over passage texts with scientific stop-word removal."""

    def __init__(self, passages_text: Sequence[str], use_stems: bool = True) -> None:
        self.texts = list(passages_text)
        self.use_stems = use_stems
        self.corpus_tokens = [self._tokens(text) for text in self.texts]
        self._model: Optional[object] = None
        self.backend = "rank_bm25" if RANK_BM25_AVAILABLE else "builtin"
        if not self.texts:
            return
        if RANK_BM25_AVAILABLE:
            # rank_bm25 cannot handle an empty document list; guard for tiny corpora.
            self._model = BM25Okapi(self.corpus_tokens)
        else:
            self._model = SimpleBM25(self.corpus_tokens)

    def _tokens(self, text: str) -> List[str]:
        return content_terms(text) if self.use_stems else tokenize(text)

    @property
    def vocabulary(self) -> set:
        vocab = set()
        for tokens in self.corpus_tokens:
            vocab.update(tokens)
        return vocab

    def raw_scores(self, query: str) -> np.ndarray:
        if not self.texts or self._model is None:
            return np.zeros((0,), dtype=np.float32)
        tokens = self._tokens(query)
        # Restrict to query terms present in the corpus to avoid wasted work.
        vocab = self.vocabulary
        tokens = [t for t in tokens if t in vocab] or tokens
        scores = np.asarray(self._model.get_scores(tokens), dtype=np.float32)  # type: ignore[attr-defined]
        return scores

    def scores(self, query: str, saturation: float = 6.0) -> np.ndarray:
        """BM25 scores squashed into 0-1 (saturating, corpus-independent)."""

        raw = self.raw_scores(query)
        if raw.size == 0:
            return raw
        raw = np.maximum(raw, 0.0)
        return raw / (raw + saturation)

    def top_indices(self, query: str, k: int = 10) -> List[int]:
        raw = self.raw_scores(query)
        if raw.size == 0:
            return []
        return [int(i) for i in np.argsort(-raw)[: max(1, k)]]


def bm25_status() -> Dict[str, object]:
    return {"rank_bm25_available": RANK_BM25_AVAILABLE}
