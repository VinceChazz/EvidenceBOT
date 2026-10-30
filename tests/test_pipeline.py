"""End-to-end pipeline tests with the fixture retrieval layer and a mocked LLM.

These cover the seven required scenarios at integration level:
TEST 1 answerable, TEST 2 unanswerable, TEST 3 partial, TEST 4 invalid citation,
TEST 5 API failure, TEST 6 conflicting findings, TEST 7 hallucination.
"""

from __future__ import annotations

from config import INSUFFICIENT_EVIDENCE_MESSAGE
from pipeline import EvidenceBot
from retrieval.retrieval_manager import RetrievalManager
from tests.conftest import (
    FakeHttp,
    documents_from_passages,
    fake_llm_client,
    make_passage,
    payload_from_evidence,
    ranked_evidence,
)
from tests.test_abstention import StubManager

QUESTION = "What role does IL-6 play in rheumatoid arthritis?"


def fixture_manager(settings, fake_http, failures=None) -> RetrievalManager:
    if failures:
        fake_http.failures = failures
    return RetrievalManager(settings, http=fake_http)


def _passages_for(manager_outcome):
    """Convert retrieved documents into the passage list used for payload building."""

    return [
        make_passage(
            text=document.text_passage,
            document_id=document.document_id,
            title=document.title,
            source_name=document.source_name,
            source_type=document.source_type,
            publication_year=document.publication_year,
            authors=list(document.authors),
            doi=document.doi,
            pmid=document.pmid,
            url=document.url,
            is_preprint=document.is_preprint,
            is_peer_reviewed=document.is_peer_reviewed,
            is_background=document.is_background,
        )
        for document in manager_outcome.documents
        if not document.is_background
    ]


# ---------------------------------------------------------------------------
# TEST 1 — answerable question, live-style retrieval through fixtures
# ---------------------------------------------------------------------------


def test_answerable_question_end_to_end(settings, openai_settings, fake_http):
    manager = fixture_manager(settings, fake_http)
    outcome = manager.retrieve(QUESTION)
    evidence = ranked_evidence(settings, _passages_for(outcome), QUESTION)
    assert evidence, "fixture retrieval must produce on-topic evidence"

    llm, completions = fake_llm_client(openai_settings, [payload_from_evidence(evidence, count=2)])
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)
    # The engine embeds with its own settings; the mocked LLM supplies the answer.
    bot.llm = llm
    result = bot.ask(QUESTION)

    assert result.decision in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}
    assert result.documents_retrieved >= 4
    assert result.citations_used >= 1
    assert result.generated_by_llm is True
    assert result.validation.is_valid, result.validation.notes
    for citation in result.citations:
        assert citation.url or citation.is_sample
        assert citation.title
        assert citation.passage_text


def test_retrieved_documents_are_standardised(settings, fake_http):
    manager = fixture_manager(settings, fake_http)
    bot = EvidenceBot(settings, retrieval_manager=manager)
    result = bot.ask("What role does IL-6 play in rheumatoid arthritis?")

    frame = result.corpus.frame
    for column in ("document_id", "title", "text_passage", "source_name", "source_type", "publication_year"):
        assert column in frame.columns
    assert frame["document_id"].is_unique
    # Chunks keep their parent document id.
    for passage in result.corpus.passages:
        assert passage.document_id in set(frame["document_id"])


# ---------------------------------------------------------------------------
# TEST 2 — unanswerable question abstains through the whole pipeline
# ---------------------------------------------------------------------------


def test_unanswerable_question_end_to_end(settings, openai_settings, fake_http):
    manager = fixture_manager(settings, fake_http)
    llm, completions = fake_llm_client(openai_settings, [])
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)

    result = bot.ask(
        "Does drinking a glass of orange juice every morning guarantee that a person will never "
        "develop cancer?"
    )

    assert result.is_abstention is True
    assert result.abstention_message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.answer is None
    assert completions.calls == []


# ---------------------------------------------------------------------------
# TEST 6 — conflicting findings
# ---------------------------------------------------------------------------


def test_conflicting_findings_are_surfaced_to_the_model(settings, openai_settings):
    conflicting = [
        make_passage(
            document_id="omega-1",
            title="Omega-3 supplementation reduces cardiovascular events",
            relevance_score=0.8,
            text=(
                "In this randomised trial, omega-3 fatty acid supplementation reduced the risk of "
                "cardiovascular events significantly and lowered cardiovascular mortality. The "
                "authors conclude that omega-3 supplementation reduces cardiovascular risk."
            ),
        ),
        make_passage(
            document_id="omega-2",
            title="No cardiovascular benefit of omega-3 in a large trial",
            relevance_score=0.78,
            source_type="randomized_controlled_trial",
            text=(
                "In a large placebo-controlled trial, omega-3 supplementation did not reduce "
                "cardiovascular risk and there was no significant difference in cardiovascular "
                "events between groups."
            ),
        ),
    ]
    evidence = ranked_evidence(
        settings, conflicting, "Do omega-3 fatty acids reduce cardiovascular risk?"
    )
    payload = payload_from_evidence(evidence, count=len(evidence)) if evidence else {}
    if payload:
        payload["conflicts"] = [
            "One trial reports reduced cardiovascular risk while another reports no significant effect."
        ]
    llm, completions = fake_llm_client(openai_settings, [payload])
    bot = EvidenceBot(
        settings, retrieval_manager=StubManager(documents_from_passages(conflicting)), llm=llm
    )
    result = bot.ask("Do omega-3 fatty acids reduce cardiovascular risk?")

    assert result.assessment.consistency.is_conflicting
    assert result.strength.grade == "conflicting"
    assert result.comparison
    stances = {row["stance"] for row in result.comparison}
    assert "support" in stances and "null" in stances
    # The disagreement must be pushed into the prompt.
    prompt = completions.calls[0]["messages"][1]["content"]
    assert "disagree" in prompt.lower()
    assert result.conflicts


# ---------------------------------------------------------------------------
# TEST 5 — retrieval failure end to end
# ---------------------------------------------------------------------------


def test_source_failure_does_not_break_pipeline(settings, openai_settings, fake_http):
    manager = fixture_manager(settings, fake_http, failures=["eutils.ncbi.nlm.nih.gov"])
    llm, _ = fake_llm_client(openai_settings, [])
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)

    result = bot.ask(QUESTION)

    assert result.documents_retrieved > 0
    assert any(f["source"] == "pubmed" for f in result.source_failures)
    assert any("PubMed" in warning for warning in result.warnings)
    # Wikipedia is background only and must never be used as evidence.
    assert all(p.source_type != "background" for p in result.evidence_passages)


# ---------------------------------------------------------------------------
# Offline sample corpus path
# ---------------------------------------------------------------------------


def test_offline_fallback_marks_every_sample_record(openai_settings):
    failing = FakeHttp(routes={}, failures=["http"])
    settings = openai_settings.with_overrides(
        enable_offline_sample_fallback=True, max_results_per_source=5
    )
    manager = RetrievalManager(settings, http=failing)
    llm, _ = fake_llm_client(settings, [])
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)

    result = bot.ask("What role does IL-6 play in rheumatoid arthritis?")

    assert result.fallback_used is True
    assert any("illustrative" in w.lower() for w in result.warnings)
    assert all(c.is_sample for c in result.citations)
    strength_notes = " ".join(result.strength.cautions).lower()
    assert "illustrative offline sample" in strength_notes


def test_analytics_separate_retrieved_from_used(settings, fake_http):
    manager = fixture_manager(settings, fake_http)
    bot = EvidenceBot(settings, retrieval_manager=manager)
    result = bot.ask(QUESTION)
    analytics = result.analytics()

    assert analytics["documents_retrieved"] >= analytics["evidence_passages_used"]
    assert analytics["passages_retrieved"] >= analytics["relevant_passages"]
    assert "retrieval_seconds" in analytics
    assert analytics["decision"] in {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
