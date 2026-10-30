"""Shared fixtures: offline settings, fake HTTP layer, fake LLM, passage builders.

No test performs a network call, so the suite never depends on paid APIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from generation.llm import LLMClient  # noqa: E402
from processing.models import Document, Passage  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture()
def settings() -> Settings:
    """Deterministic settings that never touch a real API."""

    return Settings(
        openai_api_key=None,
        embedding_backend="local",
        cache_enabled=False,
        max_results_per_source=5,
        top_k=10,
        candidate_k=20,
        relevance_threshold=0.30,
        minimum_evidence_passages=2,
        enable_offline_sample_fallback=False,
        request_timeout_seconds=2.0,
        request_retries=0,
        llm_temperature=0.0,
    )


@pytest.fixture()
def openai_settings() -> Settings:
    """Settings with a dummy key (the OpenAI client is always mocked)."""

    return Settings(
        openai_api_key="sk-test-not-a-real-key",
        embedding_backend="local",
        cache_enabled=False,
        max_results_per_source=5,
        minimum_evidence_passages=2,
        enable_offline_sample_fallback=False,
    )


class FakeHttp:
    """Drop-in replacement for :class:`retrieval.base.HttpClient`.

    Routes requests by URL substring and can be configured to raise for a host,
    which is how source-failure behaviour is tested.
    """

    def __init__(self, routes: Optional[Dict[str, str]] = None, failures: Optional[Sequence[str]] = None):
        self.routes = routes or {}
        self.failures = list(failures or [])
        self.calls: List[str] = []

    def _route(self, url: str) -> Optional[str]:
        for key, payload in self.routes.items():
            if key in url:
                return payload
        return None

    def get_text(self, url: str, params=None, headers=None, use_cache: bool = True) -> str:
        self.calls.append(url)
        for failing in self.failures:
            if failing in url:
                raise RuntimeError(f"{failing} unavailable (simulated outage)")
        payload = self._route(url)
        if payload is None:
            raise RuntimeError(f"no fixture route for {url}")
        return payload

    def get_json(self, url: str, params=None, headers=None, use_cache: bool = True) -> Any:
        return json.loads(self.get_text(url, params=params, headers=headers, use_cache=use_cache))


@pytest.fixture()
def fake_http() -> FakeHttp:
    return FakeHttp(
        routes={
            "esearch.fcgi": fixture_text("pubmed_esearch.json"),
            "efetch.fcgi": fixture_text("pubmed_medline.txt"),
            "europepmc/webservices/rest/search": fixture_text("europepmc_search.json"),
            "europepmc/webservices/rest/PMC": fixture_text("europepmc_fulltext.xml"),
            "export.arxiv.org": fixture_text("arxiv_response.xml"),
            "api.crossref.org": fixture_text("crossref_response.json"),
            "w/api.php": fixture_text("wikipedia_search.json"),
            "rest_v1/page/summary": fixture_text("wikipedia_summary.json"),
        }
    )


class FakeCompletions:
    """Mimics ``client.chat.completions``; returns queued JSON payloads."""

    def __init__(self, payloads: Sequence[Dict[str, Any]]):
        self.payloads = list(payloads)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        payload = self.payloads[index]
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=80, total_tokens=200),
        )


def fake_llm_client(settings: Settings, payloads: Sequence[Dict[str, Any]]) -> tuple:
    """Return (LLMClient, FakeCompletions) wired together without network."""

    completions = FakeCompletions(payloads)
    stub = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(settings, client=stub), completions


def documents_from_passages(passages) -> List[Document]:
    """Rebuild documents so the pipeline can re-process them from scratch."""

    documents: List[Document] = []
    for passage in passages:
        documents.append(
            make_document(
                document_id=passage.document_id,
                title=passage.title or "Untitled",
                text_passage=passage.text,
                source_name=passage.source_name,
                source_type=passage.source_type,
                publication_year=passage.publication_year,
                authors=list(passage.authors),
                doi=passage.doi,
                pmid=passage.pmid,
                url=passage.url,
                journal=passage.journal,
                is_preprint=passage.is_preprint,
                is_peer_reviewed=passage.is_peer_reviewed,
                is_background=passage.is_background,
                provenance=passage.provenance,
                is_sample=passage.is_sample,
            )
        )
    return documents


def ranked_evidence(settings, passages, question: str, top_k: int = 10, threshold: float = 0.30):
    """Run retrieval→processing→ranking only, returning the evidence set in UI order.

    The LLM sees the same order, so this lets tests build citation-accurate
    payloads instead of guessing numbers.
    """

    from evidence.evidence_evaluator import EvidenceEvaluator
    from processing.document_processor import process_documents
    from ranking.reranker import HybridRanker

    corpus = process_documents(documents_from_passages(passages), settings)
    ranking = HybridRanker(settings).rank(question, corpus.passages, top_k=top_k, threshold=threshold)
    assessment = EvidenceEvaluator(settings).evaluate(
        question, ranking.supporting, background_passages=ranking.background
    )
    return list(assessment.supported_passages)


def payload_from_evidence(evidence, count: int = 2) -> Dict[str, Any]:
    """Build a valid, fully-cited LLM payload using verbatim evidence sentences."""

    answer_parts: List[str] = []
    claims: List[Dict[str, Any]] = []
    for index, passage in enumerate(evidence[:count], start=1):
        sentence = passage.text.split(". ")[0].strip()
        if not sentence.endswith("."):
            sentence += "."
        answer_parts.append(f"{sentence} [{index}]")
        claims.append({"claim": sentence, "source_ids": [index]})
    return {
        "answer": " ".join(answer_parts),
        "supporting_claims": claims,
        "evidence_assessment": "Verbatim evidence statements from the retrieved passages.",
        "limitations": ["Only abstracts were available."],
        "unanswered_questions": [],
        "conflicts": [],
    }


@pytest.fixture()
def make_llm(openai_settings):
    def _factory(payloads):
        return fake_llm_client(openai_settings, payloads)

    return _factory


# ---------------------------------------------------------------------------
# Document / passage builders
# ---------------------------------------------------------------------------


def make_document(
    document_id: str = "doc-1",
    title: str = "Interleukin-6 signalling in rheumatoid arthritis",
    text_passage: str = (
        "Interleukin-6 is a pro-inflammatory cytokine that activates JAK/STAT3 signalling "
        "in inflamed synovium and contributes to the pathogenesis of rheumatoid arthritis."
    ),
    source_name: str = "PubMed",
    source_type: str = "peer_reviewed_article",
    **kwargs,
) -> Document:
    payload = dict(
        document_id=document_id,
        title=title,
        text_passage=text_passage,
        source_name=source_name,
        source_type=source_type,
    )
    payload.update(kwargs)
    return Document(**payload)


def make_passage(
    text: str = (
        "Interleukin-6 is a pro-inflammatory cytokine that activates JAK/STAT3 signalling "
        "in inflamed synovium and contributes to the pathogenesis of rheumatoid arthritis."
    ),
    document_id: str = "doc-1",
    source_name: str = "PubMed",
    source_type: str = "peer_reviewed_article",
    relevance_score: float = 0.7,
    **kwargs,
) -> Passage:
    document = make_document(
        document_id=document_id,
        title=kwargs.pop("title", "Interleukin-6 signalling in rheumatoid arthritis"),
        text_passage=text,
        source_name=source_name,
        source_type=source_type,
        **{k: v for k, v in kwargs.items() if k in {
            "publication_year", "authors", "doi", "pmid", "url", "journal",
            "is_preprint", "is_peer_reviewed", "is_background", "provenance", "is_sample",
        }},
    )
    passage = Passage.from_document(document, text)
    passage.relevance_score = relevance_score
    return passage


@pytest.fixture()
def make_passage_factory():
    return make_passage


@pytest.fixture()
def il6_passages() -> List[Passage]:
    """Three consistent, on-topic IL-6 / rheumatoid-arthritis passages."""

    return [
        make_passage(
            text=(
                "Interleukin-6 (IL-6) is a pleiotropic pro-inflammatory cytokine that activates "
                "JAK/STAT3 signalling in synovial tissue, promotes Th17 cell differentiation and "
                "supports B-cell autoantibody production in rheumatoid arthritis."
            ),
            document_id="doc-il6-1",
            title="Interleukin-6 signalling in rheumatoid arthritis synovium",
            relevance_score=0.82,
            publication_year=2016,
            authors=["Ilse Hanow", "Masaru Tanaka"],
            doi="10.1155/2016/2671924",
            pmid="26719248",
            url="https://pubmed.ncbi.nlm.nih.gov/26719248/",
        ),
        make_passage(
            text=(
                "In a randomised controlled trial, IL-6 receptor blockade with tocilizumab "
                "improved ACR response rates and reduced disease activity scores in rheumatoid "
                "arthritis, although serious infection rates were modestly increased."
            ),
            document_id="doc-il6-2",
            title="Randomised controlled trial of IL-6 receptor blockade in rheumatoid arthritis",
            relevance_score=0.74,
            publication_year=2018,
            authors=["Ada Okafor"],
            doi="10.1136/annrheumdis-2018-0001",
            source_type="randomized_controlled_trial",
        ),
        make_passage(
            text=(
                "A systematic review of IL-6 pathway inhibitors in rheumatoid arthritis found "
                "consistent reductions in disease activity across randomised trials."
            ),
            document_id="doc-il6-3",
            title="Systematic review of IL-6 pathway inhibitors in rheumatoid arthritis",
            relevance_score=0.69,
            publication_year=2021,
            authors=["Sample Author F"],
            source_type="meta_analysis",
            source_name="Europe PMC",
        ),
    ]
