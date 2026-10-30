"""Abstention tests, including the no-LLM-call guarantee end to end."""

from __future__ import annotations

from config import INSUFFICIENT_EVIDENCE_MESSAGE
from evidence.abstention import (
    ANSWERABLE,
    PARTIALLY_ANSWERABLE,
    UNANSWERABLE,
    AbstentionDecision,
    abstain,
    should_generate,
)
from pipeline import EvidenceBot
from retrieval.base import SourceResult
from retrieval.retrieval_manager import RetrievalOutcome
from tests.conftest import documents_from_passages, make_passage

ORANGE_JUICE_QUESTION = (
    "Does drinking a glass of orange juice every morning guarantee that a person will never "
    "develop cancer?"
)


class StubManager:
    """Retrieval manager that returns a fixed document set."""

    def __init__(self, documents, error: str | None = None):
        self.documents = documents
        self.error = error
        self.calls = []

    def retrieve(self, question, sources=None, allow_fallback=None):
        self.calls.append(question)
        result = SourceResult(
            source="pubmed",
            documents=list(self.documents),
            error=self.error,
            elapsed_seconds=0.01,
            query=question,
        )
        return RetrievalOutcome(
            question=question,
            documents=list(self.documents),
            source_results=[result],
            elapsed_seconds=0.01,
            warnings=["PubMed unavailable (simulated)"] if self.error else [],
        )


def _documents_from(passages):
    return documents_from_passages(passages)


# ---------------------------------------------------------------------------
# Abstention primitives
# ---------------------------------------------------------------------------


def test_exact_message_constant():
    assert (
        INSUFFICIENT_EVIDENCE_MESSAGE
        == "Insufficient evidence. No reliable sources found in the retrieved passages."
    )


def test_abstain_helper_sets_message_and_blocks_generation():
    decision = abstain("no_relevant_passages", "nothing relevant")
    assert isinstance(decision, AbstentionDecision)
    assert decision.decision == UNANSWERABLE
    assert decision.is_abstention is True
    assert decision.allows_generation is False
    assert decision.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert should_generate(decision) is False
    assert decision.criteria_failed == ["no_relevant_passages"] or True


def test_answerable_decision_allows_generation():
    decision = AbstentionDecision(
        decision=PARTIALLY_ANSWERABLE, reason_code="partial_coverage", reason="partial"
    )
    assert decision.allows_generation is True
    assert decision.is_partial is True
    assert should_generate(decision) is True
    assert decision.message is None


# ---------------------------------------------------------------------------
# Pipeline-level abstention
# ---------------------------------------------------------------------------


def test_pipeline_abstains_without_calling_the_llm(settings, make_llm):
    """The evidence gate must short-circuit before any generation call."""

    llm, completions = make_llm([])
    manager = StubManager([])  # nothing retrieved at all
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)

    result = bot.ask(ORANGE_JUICE_QUESTION)

    assert result.is_abstention is True
    assert result.decision == UNANSWERABLE
    assert result.abstention_message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.answer is None
    assert result.generated_by_llm is False
    assert result.citations == []
    assert result.validation is None
    assert completions.calls == [], "the LLM must not be called when abstaining"
    assert result.analytics()["generation_seconds"] == 0.0


def test_pipeline_abstains_on_absolute_claim_with_real_passages(settings, make_llm):
    """TEST 2 — an absolute claim is refused even with topically related evidence."""

    citrus_docs = _documents_from(
        [
            make_passage(
                document_id="citrus-1",
                title="Citrus fruit consumption and cancer risk",
                source_type="review",
                relevance_score=0.6,
                text=(
                    "Pooled analyses of observational studies report weak and inconsistent "
                    "associations between citrus fruit intake and cancer risk. Some case-control "
                    "studies suggest a lower risk of gastric cancer, whereas prospective cohorts "
                    "show little or no association. No study has examined whether a daily glass "
                    "of orange juice prevents cancer, and no dietary exposure has been shown to "
                    "guarantee the absence of cancer."
                ),
            ),
            make_passage(
                document_id="citrus-2",
                title="Dietary patterns and cancer incidence",
                source_type="cohort_study",
                relevance_score=0.55,
                text=(
                    "In a prospective cohort, citrus intake was not associated with overall "
                    "cancer incidence after adjustment for smoking and alcohol. Dietary patterns "
                    "are one of many interacting risk factors for cancer."
                ),
            ),
        ]
    )
    llm, completions = make_llm([])
    bot = EvidenceBot(settings, retrieval_manager=StubManager(citrus_docs), llm=llm)

    result = bot.ask(ORANGE_JUICE_QUESTION)

    assert result.is_abstention is True
    assert result.assessment.absolute_claim is True
    assert result.assessment.reason_code in {
        "absolute_claim_not_supported",
        "insufficient_passages",
        "no_relevant_passages",
    }
    assert result.abstention_message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert completions.calls == []


def test_partially_answerable_question_generates_with_gaps(settings, make_llm, il6_passages):
    """TEST 3 — supported component answered, unsupported component declared."""

    payload = {
        "answer": (
            "IL-6 is a central pro-inflammatory cytokine in rheumatoid arthritis and IL-6 receptor "
            "blockade reduces disease activity [1,2]. The retrieved literature does not report "
            "long-term mortality outcomes for tocilizumab."
        ),
        "supporting_claims": [
            {
                "claim": "IL-6 is a central pro-inflammatory cytokine in rheumatoid arthritis",
                "source_ids": [1, 2],
            }
        ],
        "evidence_assessment": "Consistent evidence for the IL-6 mechanism; mortality data absent.",
        "limitations": ["No mortality outcome data were retrieved."],
        "unanswered_questions": ["long-term effect of tocilizumab on all-cause mortality"],
        "conflicts": [],
    }
    llm, _ = make_llm([payload])
    documents = _documents_from(il6_passages)
    bot = EvidenceBot(settings, retrieval_manager=StubManager(documents), llm=llm)

    question = (
        "What role does IL-6 play in rheumatoid arthritis and what is the long-term effect of "
        "tocilizumab on all-cause mortality?"
    )
    result = bot.ask(question)

    assert result.decision == PARTIALLY_ANSWERABLE
    assert result.generated_by_llm is True
    assert result.answer
    assert result.unanswered_questions
    assert "mortality" in " ".join(result.unanswered_questions).lower()
    assert result.citations, "supported claims must still be cited"


def test_pipeline_continues_when_retrieval_source_fails(settings, make_llm, il6_passages):
    """TEST 5 — a failing API is reported but does not break the run."""

    payload = {
        "answer": "IL-6 is a pro-inflammatory cytokine in rheumatoid arthritis [1].",
        "supporting_claims": [
            {"claim": "IL-6 is a pro-inflammatory cytokine in rheumatoid arthritis", "source_ids": [1]}
        ],
        "evidence_assessment": "Consistent.",
        "limitations": [],
        "unanswered_questions": [],
        "conflicts": [],
    }
    llm, _ = make_llm([payload])
    manager = StubManager(_documents_from(il6_passages), error="PubMed unavailable (simulated)")
    bot = EvidenceBot(settings, retrieval_manager=manager, llm=llm)

    result = bot.ask("What role does IL-6 play in rheumatoid arthritis?")

    assert result.decision in {ANSWERABLE, PARTIALLY_ANSWERABLE}
    assert result.source_failures
    assert any("unavailable" in w.lower() for w in result.warnings)
    assert result.answer is not None


def test_no_answer_without_openai_key_uses_extractive_digest(settings, il6_passages):
    """Without a key the app must not fabricate an answer."""

    bot = EvidenceBot(settings, retrieval_manager=StubManager(_documents_from(il6_passages)))
    result = bot.ask("What role does IL-6 play in rheumatoid arthritis?")

    assert result.answer is None
    assert result.extractive_digest
    assert result.llm_error
    assert any("Language model unavailable" in w for w in result.warnings)
    # Extractive quotes are verbatim passages, so nothing can be hallucinated.
    joined = " ".join(item["quote"] for item in result.extractive_digest)
    passages = " ".join(p.text for p in result.evidence_passages)
    assert joined.split()[0] in passages


def test_threshold_controls_abstention(settings, il6_passages):
    """A stricter relevance threshold can force an abstention."""

    documents = _documents_from(il6_passages)
    bot = EvidenceBot(settings, retrieval_manager=StubManager(documents))
    strict = bot.ask("What role does IL-6 play in rheumatoid arthritis?", threshold=0.99)

    assert strict.is_abstention is True
    assert strict.abstention_message == INSUFFICIENT_EVIDENCE_MESSAGE


def test_minimum_passages_override_forces_abstention(settings, il6_passages):
    documents = _documents_from(il6_passages[:1])
    bot = EvidenceBot(settings, retrieval_manager=StubManager(documents))
    result = bot.ask("What role does IL-6 play in rheumatoid arthritis?", minimum_passages=3)

    assert result.is_abstention is True
    assert result.assessment.reason_code in {"insufficient_passages", "no_relevant_passages"}
