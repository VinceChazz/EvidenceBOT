"""Generation tests: prompts, structured output, repair loop and LLM failures."""

from __future__ import annotations

from evidence.evidence_evaluator import EvidenceEvaluator
from generation.extractive import build_extractive_digest
from generation.llm import LLMClient
from generation.prompts import (
    SYSTEM_PROMPT,
    build_regeneration_prompt,
    build_user_prompt,
    format_evidence_block,
)
from pipeline import EvidenceBot
from tests.conftest import (
    fake_llm_client,
    make_passage,
    payload_from_evidence,
    ranked_evidence,
)
from tests.test_abstention import StubManager, _documents_from

QUESTION = "What role does IL-6 play in rheumatoid arthritis?"

VALID_PAYLOAD = {
    "answer": (
        "IL-6 is a pro-inflammatory cytokine that activates JAK/STAT3 signalling in inflamed "
        "synovium and supports B-cell autoantibody production in rheumatoid arthritis [1]. "
        "IL-6 receptor blockade with tocilizumab improved ACR response rates and reduced "
        "disease activity scores [2]."
    ),
    "supporting_claims": [
        {
            "claim": "IL-6 activates JAK/STAT3 signalling in inflamed synovium in rheumatoid arthritis",
            "source_ids": [1],
        },
        {"claim": "IL-6 receptor blockade with tocilizumab improved ACR response rates", "source_ids": [2]},
    ],
    "evidence_assessment": "Consistent, direct evidence from a mechanistic article and an RCT.",
    "limitations": ["Long-term safety data are limited."],
    "unanswered_questions": [],
    "conflicts": [],
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_system_prompt_encodes_the_ten_requirements():
    lowered = SYSTEM_PROMPT.lower()
    assert "exclusively" in lowered and "retrieved evidence" in lowered
    assert "never invent" in lowered
    assert "[1]" in SYSTEM_PROMPT and "inline" in lowered
    assert "conflicting" in lowered
    assert "limitations" in lowered
    assert "unanswered" in lowered
    assert "pretrained" in lowered
    assert "json" in lowered


def test_evidence_block_is_numbered_and_carries_metadata(settings, il6_passages):
    block = format_evidence_block(il6_passages)
    assert block.startswith("[1]")
    assert "[2]" in block and "[3]" in block
    assert "doi: 10.1155/2016/2671924" in block
    assert "pmid: 26719248" in block
    assert "type: randomized controlled trial" in block.lower()
    assert "type: meta analysis" in block.lower()
    assert "PASSAGE:" in block


def test_preprint_and_sample_flags_reach_the_prompt(settings):
    preprint = make_passage(
        document_id="pp-1",
        source_name="arXiv",
        source_type="preprint",
        is_preprint=True,
        is_peer_reviewed=False,
        is_sample=True,
        text="A preprint reporting IL-6 blockade effects in rheumatoid arthritis cohorts.",
    )
    block = format_evidence_block([preprint])
    assert "PREPRINT" in block
    assert "ILLUSTRATIVE SAMPLE" in block


def test_user_prompt_includes_gate_result_and_constraints(settings, il6_passages):
    assessment = EvidenceEvaluator(settings).evaluate(
        "What role does IL-6 play in rheumatoid arthritis and what is the effect on mortality?",
        il6_passages,
    )
    prompt = build_user_prompt(assessment and QUESTION, il6_passages, assessment, partial=True)
    assert QUESTION in prompt
    assert "EVIDENCE GATE RESULT" in prompt
    assert "PARTIALLY_ANSWERABLE" in prompt
    assert "RETRIEVED EVIDENCE" in prompt
    assert "unanswered" in prompt.lower()


def test_regeneration_prompt_lists_problems(settings, il6_passages):
    prompt = build_regeneration_prompt(
        "old answer",
        {
            "invalid_citations": [9],
            "unsupported_claims": [{"claim": "a fabricated claim"}],
            "uncited_claims": ["an uncited claim"],
            "invented_urls": ["https://fake.example"],
        },
        il6_passages,
    )
    assert "9" in prompt
    assert "fake.example" in prompt
    assert "a fabricated claim" in prompt
    assert "an uncited claim" in prompt
    assert "EVIDENCE (unchanged" in prompt


# ---------------------------------------------------------------------------
# LLM client behaviour
# ---------------------------------------------------------------------------


def test_llm_client_parses_json(openai_settings):
    llm, completions = fake_llm_client(openai_settings, [VALID_PAYLOAD])
    result = llm.generate_json("system", "user")

    assert result.ok is True
    assert result.data["answer"].startswith("IL-6")
    assert result.model == openai_settings.openai_model
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["temperature"] == openai_settings.llm_temperature


def test_llm_client_strips_markdown_fences(openai_settings):
    fenced = "```json\n" + __import__("json").dumps(VALID_PAYLOAD) + "\n```"
    llm, _ = fake_llm_client(openai_settings, [fenced])
    result = llm.generate_json("system", "user", use_cache=False)
    assert result.ok is True
    assert result.data["answer"]


def test_llm_client_reports_missing_key(settings):
    llm = LLMClient(settings)
    assert llm.is_available is False
    result = llm.generate_json("system", "user")
    assert result.ok is False
    assert "OPENAI_API_KEY" in (result.error or "")


def test_llm_client_survives_api_exception(openai_settings):
    class Exploding:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("upstream 503")

    llm = LLMClient(openai_settings, client=Exploding())
    result = llm.generate_json("system", "user", use_cache=False)
    assert result.ok is False
    assert "upstream 503" in (result.error or "")


# ---------------------------------------------------------------------------
# Pipeline generation
# ---------------------------------------------------------------------------


def test_pipeline_generates_verified_answer(settings, il6_passages, openai_settings):
    evidence = ranked_evidence(settings, il6_passages, QUESTION)
    payload = payload_from_evidence(evidence, count=2)
    llm, completions = fake_llm_client(openai_settings, [payload])
    bot = EvidenceBot(openai_settings, retrieval_manager=StubManager(_documents_from(il6_passages)), llm=llm)
    result = bot.ask(QUESTION)

    assert result.generated_by_llm is True
    assert result.answer
    assert "[1]" in result.answer and "[2]" in result.answer
    assert result.validation is not None
    assert result.validation.is_valid, result.validation.notes
    assert result.validation.unsupported_claims == []
    assert result.citations_used >= 2
    assert result.claims
    assert result.claims[0]["source_ids"]
    assert completions.calls, "the LLM is called once the gate approves"
    # Citation cards must match the numbers used in the answer.
    assert {c.number for c in result.citations} >= {1, 2}


def test_pipeline_repairs_invalid_citation(settings, il6_passages, openai_settings):
    """Invalid ids trigger one repair attempt that must be used instead."""

    bad = {
        "answer": "IL-6 is a pro-inflammatory cytokine in rheumatoid arthritis [9].",
        "supporting_claims": [{"claim": "IL-6 is a pro-inflammatory cytokine in rheumatoid arthritis", "source_ids": [9]}],
        "evidence_assessment": "ok",
        "limitations": [],
        "unanswered_questions": [],
        "conflicts": [],
    }
    evidence = ranked_evidence(settings, il6_passages, QUESTION)
    good = payload_from_evidence(evidence, count=2)
    llm, completions = fake_llm_client(openai_settings, [bad, good])
    bot = EvidenceBot(openai_settings, retrieval_manager=StubManager(_documents_from(il6_passages)), llm=llm)
    result = bot.ask(QUESTION)

    assert result.generate_attempts == 2
    assert len(completions.calls) == 2
    assert result.validation.is_valid is True, result.validation.notes
    assert "[9]" not in (result.answer or "")
    assert result.generated_by_llm is True


def test_pipeline_flags_unverifiable_answer(settings, il6_passages, openai_settings):
    """TEST 7 end to end — a persistently hallucinating model cannot look verified."""

    evidence = ranked_evidence(settings, il6_passages, QUESTION)
    verbatim = evidence[0].text.split(". ")[0].strip()
    hallucination = {
        "answer": (
            f"{verbatim} [1]. "
            "A landmark trial proved that IL-6 blockade permanently cures rheumatoid arthritis "
            "in 98 percent of patients [9]."
        ),
        "supporting_claims": [
            {"claim": "IL-6 blockade permanently cures rheumatoid arthritis", "source_ids": [9]}
        ],
        "evidence_assessment": "Definitive.",
        "limitations": [],
        "unanswered_questions": [],
        "conflicts": [],
    }
    llm, completions = fake_llm_client(openai_settings, [hallucination])
    bot = EvidenceBot(openai_settings, retrieval_manager=StubManager(_documents_from(il6_passages)), llm=llm)
    result = bot.ask(QUESTION)

    assert result.generate_attempts == openai_settings.max_generation_attempts
    assert result.validation.is_valid is False
    assert 9 in result.validation.invalid_citations
    assert "[9]" not in (result.answer or "")
    assert any("could not be fully citation-verified" in w for w in result.warnings)
    # The fabricated sentence survives only as unverified prose; its id is gone and
    # the response is explicitly marked as not verified.
    assert result.validation.verification_label.startswith("Citations could not be fully verified")


def test_generation_failure_falls_back_to_extractive_digest(settings, il6_passages, openai_settings):
    class Broken:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("network down")

    llm = LLMClient(openai_settings, client=Broken())
    bot = EvidenceBot(openai_settings, retrieval_manager=StubManager(_documents_from(il6_passages)), llm=llm)
    result = bot.ask(QUESTION)

    assert result.answer is None
    assert result.extractive_digest
    assert result.llm_error
    assert any("Answer generation failed" in w for w in result.warnings)


def test_extractive_digest_quotes_real_passages(il6_passages):
    items = build_extractive_digest(QUESTION, il6_passages)
    assert len(items) == len(il6_passages)
    assert items[0]["citation"] == 1
    assert items[0]["quote"]
    assert items[0]["quote"] in il6_passages[0].text


def test_result_serialises_for_history(settings, il6_passages, openai_settings):
    import json

    evidence = ranked_evidence(settings, il6_passages, QUESTION)
    llm, _ = fake_llm_client(openai_settings, [payload_from_evidence(evidence, count=2)])
    bot = EvidenceBot(openai_settings, retrieval_manager=StubManager(_documents_from(il6_passages)), llm=llm)
    payload = bot.ask(QUESTION).to_dict()

    assert payload["decision"] in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}
    assert payload["analytics"]["citations_used"] >= 1
    assert payload["citations"][0]["number"] == 1
    assert json.dumps(payload, default=str)  # must be JSON-serialisable
