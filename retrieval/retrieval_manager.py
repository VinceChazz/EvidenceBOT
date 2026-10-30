"""Central retrieval manager.

Fans out the user's question to every enabled scientific source concurrently,
records per-source failures, retries with repaired queries where a source
supports it, and returns normalised :class:`Document` objects.

If *every* live source fails (e.g. an air-gapped network) and the offline
sample corpus is enabled, the manager loads the clearly-labelled illustrative
corpus so the pipeline can still be exercised. That corpus is never presented as
live retrieval.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from config import DATA_DIR, Settings
from processing.models import Document
from retrieval.arxiv import ArxivRetriever
from retrieval.base import HttpClient, SourceResult, run_source
from retrieval.crossref import CrossrefRetriever
from retrieval.europe_pmc import EuropePMCRetriever
from retrieval.pubmed import PubMedRetriever
from retrieval.wikipedia import WikipediaRetriever
from utils.logging_config import get_logger

logger = get_logger("retrieval.manager")

SOURCE_ORDER = ["pubmed", "europe_pmc", "arxiv", "crossref", "wikipedia"]
SOURCE_LABELS = {
    "pubmed": "PubMed / NCBI",
    "europe_pmc": "Europe PMC",
    "arxiv": "arXiv",
    "crossref": "Crossref",
    "wikipedia": "Wikipedia (background)",
}
OFFLINE_CORPUS_PATH = os.path.join(DATA_DIR, "offline_sample_corpus.json")


@dataclass
class RetrievalOutcome:
    question: str
    documents: List[Document] = field(default_factory=list)
    source_results: List[SourceResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[SourceResult]:
        return [r for r in self.source_results if r.error]

    @property
    def successes(self) -> List[SourceResult]:
        return [r for r in self.source_results if r.error is None]

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    def summary(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "documents": self.total_documents,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "fallback_used": self.fallback_used,
            "sources": [r.summary() for r in self.source_results],
        }


def load_offline_corpus(path: str = OFFLINE_CORPUS_PATH) -> List[Document]:
    """Load the illustrative offline corpus (never real retrieval)."""

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("offline corpus unavailable: %s", exc)
        return []
    documents: List[Document] = []
    for raw in payload.get("documents", []):
        raw = dict(raw)
        raw.setdefault("is_sample", True)
        raw.setdefault("provenance", "offline_sample")
        raw.setdefault("url", None)
        raw["is_sample"] = True
        try:
            documents.append(Document(**raw))
        except TypeError as exc:  # pragma: no cover - malformed corpus
            logger.warning("skipping malformed offline record: %s", exc)
    return documents


class RetrievalManager:
    """Concurrent multi-source retrieval."""

    def __init__(
        self,
        settings: Settings,
        http: Optional[HttpClient] = None,
        retrievers: Optional[Dict[str, object]] = None,
    ) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)
        if retrievers is not None:
            self.retrievers = retrievers
        else:
            self.retrievers = {
                "pubmed": PubMedRetriever(settings, self.http),
                "europe_pmc": EuropePMCRetriever(settings, self.http),
                "arxiv": ArxivRetriever(settings, self.http),
                "crossref": CrossrefRetriever(settings, self.http),
                "wikipedia": WikipediaRetriever(settings, self.http),
            }

    # ------------------------------------------------------------------
    def available_sources(self) -> List[str]:
        return [s for s in SOURCE_ORDER if s in self.retrievers]

    def _resolve_sources(self, sources: Optional[Iterable[str]]) -> List[str]:
        if not sources:
            return [s for s in self.available_sources() if s in self.settings.enabled_sources]
        wanted = [s for s in sources if s in self.retrievers]
        return [s for s in SOURCE_ORDER if s in wanted]

    def _call_source(self, name: str, question: str) -> SourceResult:
        retriever = self.retrievers[name]
        limit = self.settings.max_results_per_source
        if name == "wikipedia":
            limit = min(limit, 2)
        return run_source(name, lambda: retriever.search(question, limit=limit), question)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        sources: Optional[Iterable[str]] = None,
        allow_fallback: Optional[bool] = None,
    ) -> RetrievalOutcome:
        """Retrieve documents for ``question`` from all requested sources."""

        started = time.perf_counter()
        question = (question or "").strip()
        selected = self._resolve_sources(sources)
        outcome = RetrievalOutcome(question=question)

        if not question:
            outcome.warnings.append("No question supplied.")
            return outcome

        if not selected:
            outcome.warnings.append("No retrieval sources selected.")
            return outcome

        results: List[SourceResult] = []
        workers = max(1, min(self.settings.retrieval_workers, len(selected)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieval") as pool:
            futures = {
                pool.submit(self._call_source, name, question): name for name in selected
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - never crash the run
                    logger.exception("unexpected failure in %s", name)
                    results.append(
                        SourceResult(source=name, error=f"{type(exc).__name__}: {exc}")
                    )

        results.sort(key=lambda r: SOURCE_ORDER.index(r.source) if r.source in SOURCE_ORDER else 99)
        outcome.source_results = results

        documents: List[Document] = []
        for result in results:
            documents.extend(result.documents)
            if result.error:
                outcome.warnings.append(
                    f"{SOURCE_LABELS.get(result.source, result.source)} unavailable: {result.error}"
                )

        fallback_enabled = (
            self.settings.enable_offline_sample_fallback
            if allow_fallback is None
            else allow_fallback
        )
        live_failures = all(r.error for r in results)
        no_documents = not documents

        if no_documents and live_failures and fallback_enabled:
            sample_docs = load_offline_corpus()
            if sample_docs:
                documents = sample_docs
                outcome.fallback_used = True
                outcome.source_results.append(
                    SourceResult(
                        source="offline_sample",
                        documents=sample_docs,
                        note=(
                            "All live sources were unreachable. The illustrative offline "
                            "sample corpus was loaded so the pipeline can be demonstrated."
                        ),
                    )
                )
                outcome.warnings.append(
                    "All live scientific sources were unreachable. Showing the "
                    "illustrative offline sample corpus — these are NOT live results."
                )

        outcome.documents = documents
        outcome.elapsed_seconds = time.perf_counter() - started
        logger.info(
            "retrieval finished: %d documents from %d sources in %.2fs (failures: %d)",
            len(documents),
            len(results),
            outcome.elapsed_seconds,
            len(outcome.failures),
        )
        return outcome

    def retrieve_documents(self, question: str, **kwargs) -> List[Document]:
        return self.retrieve(question, **kwargs).documents


def build_default_manager(settings: Optional[Settings] = None) -> RetrievalManager:
    return RetrievalManager(settings or Settings())
