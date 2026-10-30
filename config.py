"""Central configuration for EvidenceBot.

All tunable parameters live here. Values are resolved in this order:

1. Explicit constructor arguments (used by tests).
2. Environment variables (loaded from a local ``.env`` file if present).
3. Streamlit secrets (``.streamlit/secrets.toml``) when running inside Streamlit.
4. Sensible defaults.

Secrets (Groq, OpenAI, and NCBI API keys) are only ever read from the environment
or Streamlit secrets. They are never hard-coded or rendered in the UI or logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env once, without overriding variables already present in the process.
load_dotenv(override=False)

#: Exact message required by the EvidenceBot specification.
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence. No reliable sources found in the retrieved passages."
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "evidencebot")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _get_secret(name: str) -> Optional[str]:
    """Look up ``name`` in Streamlit secrets, returning None outside Streamlit."""

    try:  # Imported lazily: config is also used by pure-Python tests.
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets"):
            try:
                value = st.secrets.get(name)  # type: ignore[attr-defined]
            except Exception:
                value = None
            if value:
                return str(value)
    except Exception:
        pass
    return None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        secret = _get_secret(name)
        return secret if secret else default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime configuration for the whole pipeline."""

    # ---------------------------------------------------------------- LLM ---
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: (
            _env(
                "LLM_MODEL",
                _env("OPENAI_MODEL", "llama-3.3-70b-versatile")
            )
            or "llama-3.3-70b-versatile"
        )
    )
    openai_embedding_model: str = field(
        default_factory=lambda: (
            _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            or "text-embedding-3-small"
        )
    )
    openai_base_url: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_BASE_URL")
    )
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_output_tokens: int = field(
        default_factory=lambda: _env_int("LLM_MAX_OUTPUT_TOKENS", 1200)
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0)
    )
    #: "auto" -> OpenAI embeddings when a key exists, otherwise the local embedder.
    embedding_backend: str = field(
        default_factory=lambda: (
            _env("EMBEDDING_BACKEND", "local") or "local"
        ).lower()
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64)
    )
    # LLM provider configuration
    llm_provider: str = field(
        default_factory=lambda: (_env("LLM_PROVIDER", "groq") or "groq").lower()
    )

    groq_api_key: Optional[str] = field(
        default_factory=lambda: _env("GROQ_API_KEY")
    )
    # --------------------------------------------------------- Retrieval ---
    max_results_per_source: int = field(
        default_factory=lambda: _env_int("MAX_RESULTS_PER_SOURCE", 8)
    )
    retrieval_workers: int = field(default_factory=lambda: _env_int("RETRIEVAL_WORKERS", 6))
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT_SECONDS", 12.0)
    )
    request_retries: int = field(default_factory=lambda: _env_int("REQUEST_RETRIES", 2))
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 3600))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))

    # NCBI recommends identifying yourself for E-utilities usage.
    ncbi_email: str = field(
        default_factory=lambda: _env("NCBI_EMAIL", "evidencebot@example.org")
        or "evidencebot@example.org"
    )
    ncbi_tool: str = field(default_factory=lambda: _env("NCBI_TOOL", "EvidenceBot") or "EvidenceBot")
    ncbi_api_key: Optional[str] = field(default_factory=lambda: _env("NCBI_API_KEY"))
    #: Polite contact address sent to Crossref.
    crossref_mailto: str = field(
        default_factory=lambda: _env("CROSSREF_MAILTO", "evidencebot@example.org")
        or "evidencebot@example.org"
    )

    enabled_sources: tuple = (
        "pubmed",
        "europe_pmc",
        "arxiv",
        "crossref",
        "wikipedia",
    )
    #: Europe PMC full-text download is capped and only attempted for open access items.
    max_fulltext_fetches: int = field(
        default_factory=lambda: _env_int("MAX_FULLTEXT_FETCHES", 2)
    )
    max_fulltext_chars: int = field(
        default_factory=lambda: _env_int("MAX_FULLTEXT_CHARS", 12000)
    )

    # ---------------------------------------------------------- Processing ---
    chunk_size_words: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_WORDS", 170))
    chunk_overlap_words: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_WORDS", 40))
    min_chunk_words: int = field(default_factory=lambda: _env_int("MIN_CHUNK_WORDS", 12))

    # ------------------------------------------------------------- Ranking ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 10))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 30))
    relevance_threshold: float = field(
        default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.30)
    )
    minimum_evidence_passages: int = field(
        default_factory=lambda: _env_int("MINIMUM_EVIDENCE_PASSAGES", 2)
    )
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 0.6))
    bm25_weight: float = field(default_factory=lambda: _env_float("BM25_WEIGHT", 0.4))
    #: A passage must clear this lexical support level to be treated as directly on-topic.
    min_term_overlap: float = field(
        default_factory=lambda: _env_float("MIN_TERM_OVERLAP", 0.15)
    )

    # ------------------------------------------------------------ Evidence ---
    #: Minimum share of question concepts present in the evidence to call it fully answerable.
    min_question_coverage: float = field(
        default_factory=lambda: _env_float("MIN_QUESTION_COVERAGE", 0.55)
    )
    #: Total words of usable evidence required before an answer may be generated.
    min_evidence_words: int = field(
        default_factory=lambda: _env_int("MIN_EVIDENCE_WORDS", 45)
    )
    #: Consistency needs at least this many on-topic passages to be judged.
    min_passages_for_consistency: int = field(
        default_factory=lambda: _env_int("MIN_PASSAGES_FOR_CONSISTENCY", 3)
    )
    #: Claims with absolute quantifiers ("never", "guarantee") demand the strongest evidence.
    absolute_claim_policy: str = field(
        default_factory=lambda: (_env("ABSOLUTE_CLAIM_POLICY", "abstain") or "abstain").lower()
    )
    #: Minimum lexical/embedding overlap for a sentence to count as supported by its citation.
    claim_support_threshold: float = field(
        default_factory=lambda: _env_float("CLAIM_SUPPORT_THRESHOLD", 0.45)
    )
    max_generation_attempts: int = field(
        default_factory=lambda: _env_int("MAX_GENERATION_ATTEMPTS", 2)
    )

    # -------------------------------------------------------------- Demo ---
    #: Offline sample corpus, used only as an explicit, clearly labelled fallback
    #: when every live source is unreachable (e.g. sandboxed networks).
    enable_offline_sample_fallback: bool = field(
        default_factory=lambda: _env_bool("ENABLE_OFFLINE_SAMPLE_FALLBACK", True)
    )

    # ------------------------------------------------------------- Misc ---
    log_level: str = field(default_factory=lambda: (_env("LOG_LEVEL", "INFO") or "INFO").upper())

    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.enabled_sources, str):  # pragma: no cover - defensive
            self.enabled_sources = tuple(
                s.strip() for s in self.enabled_sources.split(",") if s.strip()
            )
        self.llm_provider = self.llm_provider.lower().strip()
        if self.llm_provider not in {"groq", "openai"}:
            raise ValueError(
                "LLM_PROVIDER must be either 'groq' or 'openai'."
            )
        self.embedding_backend = self.embedding_backend.lower().strip()
        if self.embedding_backend not in {"local", "openai", "auto"}:
            raise ValueError(
                "EMBEDDING_BACKEND must be 'local', 'openai', or 'auto'."
            )
        self.dense_weight, self.bm25_weight = _normalise_weights(
            self.dense_weight, self.bm25_weight
        )
        self.top_k = max(1, self.top_k)
        self.candidate_k = max(self.top_k, self.candidate_k)
        self.minimum_evidence_passages = max(1, self.minimum_evidence_passages)
        self.relevance_threshold = min(max(self.relevance_threshold, 0.0), 1.0)

    # ---------------------------------------------------------------------
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "groq":
            return bool(self.groq_api_key and self.groq_api_key.strip())

        return self.has_openai_key

    @property
    def llm_api_key(self) -> Optional[str]:
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.openai_api_key

    @property
    def llm_base_url(self) -> Optional[str]:
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return self.openai_base_url

    @property
    def llm_model(self) -> str:
        return self.openai_model

    @property
    def use_openai_embeddings(self) -> bool:
        if self.embedding_backend == "local":
            return False
        if self.embedding_backend == "openai":
            return True
        return bool(self.openai_api_key)

    def resolved_embedding_backend(self) -> str:
        return "openai" if self.use_openai_embeddings else "local"

    def redacted(self) -> Dict[str, Any]:
        """Return a log-safe view of the settings (never exposes the API key)."""

        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if "key" in f.name.lower() and f.name != "openai_embedding_model":
                out[f.name] = "***set***" if value else None
            else:
                out[f.name] = value
        return out

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a copy with the given fields replaced."""

        data = {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
        unknown = set(kwargs) - set(data)
        if unknown:
            raise ValueError(
                f"Unknown settings override(s): {', '.join(sorted(unknown))}"
            )
        data.update(kwargs)
        return Settings(**data)


def _normalise_weights(dense: float, sparse: float) -> tuple:
    dense = max(0.0, float(dense))
    sparse = max(0.0, float(sparse))
    total = dense + sparse
    if total <= 0:
        return 0.6, 0.4
    return dense / total, sparse / total


#: Process-wide default instance. Tests usually build their own Settings.
settings = Settings()
