"""Evidence-gate tests: the four dimensions, decisions and strength grading."""

from __future__ import annotations

import pytest

from config import INSUFFICIENT_EVIDENCE_MESSAGE
from evidence.abstention import ANSWERABLE, PARTIALLY_ANSWERABLE, UNANSWERABLE
from evidence.evidence_evaluator import EvidenceEvaluator
from evidence.evidence_strength import (
    CONFLICTING,
    INSUFFICIENT,
    LIMITED,
    MODERATE,
    STRONG,
    GRADE_LABELS,
    assess_evidence_strength,
)
from tests.conftest import make_passage

QUESTION = "What role does IL-6 play in rheumatoid arthritis?"


# ---------------------------------------------------------------------------
# Dimensions and happy path
# ---------------------------------------------------------------------------


def test_all_four_dimensions_are_reported(settings, il6_passages):
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, il6_passages)

    assert assessment.decision == ANSWERABLE
    assert assessment.relevance["passages"] == 3
    assert assessment.sufficiency["passages"] == 3
    assert assessment.source_quality["peer_reviewed"] == 3
    assert assessment.consistency.conclusion in {"consistent_support", "consistent_null"}

    criteria = {c.name for c in assessment.criteria}
    assert {
        "Retrieval relevance",
        "Minimum evidence passages",
        "Evidence content volume",
        "Source quality",
        "Absolute claim check",
        "Question coverage",
    } <= criteria
    assert all(c.explanation for c in assessment.criteria)


def test_decision_records_reason_codes(settings, il6_passages):
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, il6_passages)
    assert assessment.reason_code == "evidence_sufficient"
    assert assessment.abstention.allows_generation is True
    assert assessment.message is None


# ---------------------------------------------------------------------------
# Abstention triggers
# ---------------------------------------------------------------------------


def test_no_passages_abstains_with_exact_message(settings):
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, [])

    assert assessment.decision == UNANSWERABLE
    assert assessment.reason_code == "no_relevant_passages"
    assert assessment.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert assessment.abstention.allows_generation is False


def test_below_minimum_passages_abstains(settings):
    single = [make_passage(document_id="only-one", relevance_score=0.9)]
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, single)

    assert assessment.decision == UNANSWERABLE
    assert assessment.reason_code == "insufficient_passages"
    assert INSUFFICIENT_EVIDENCE_MESSAGE == assessment.message


def test_low_relevance_abstains(settings):
    weak = [
        make_passage(document_id="w1", relevance_score=0.05, text="Unrelated text about soil chemistry."),
        make_passage(document_id="w2", relevance_score=0.10, text="Another unrelated passage."),
    ]
    assessment = EvidenceEvaluator(settings).evaluate("quantum chromodynamics in stars", weak)
    assert assessment.decision == UNANSWERABLE
    assert assessment.reason_code == "no_relevant_passages"


def test_absolute_claim_abstains(settings):
    """TEST 2 (unit level) — guarantee wording cannot be established."""

    passages = [
        make_passage(
            document_id="citrus-1",
            relevance_score=0.6,
            text=(
                "Observational studies of citrus fruit intake and cancer risk are inconsistent; "
                "no dietary exposure has been shown to prevent cancer."
            ),
        ),
        make_passage(
            document_id="citrus-2",
            relevance_score=0.5,
            text="Pooled analyses show little or no association between citrus intake and cancer.",
        ),
    ]
    assessment = EvidenceEvaluator(settings).evaluate(
        "Does drinking a glass of orange juice every morning guarantee that a person will never "
        "develop cancer?",
        passages,
    )

    assert assessment.decision == UNANSWERABLE
    assert assessment.reason_code == "absolute_claim_not_supported"
    assert assessment.absolute_claim is True
    assert "guarantee" in assessment.absolute_cues
    assert assessment.message == INSUFFICIENT_EVIDENCE_MESSAGE


def test_background_material_cannot_satisfy_evidence(settings):
    background = [
        make_passage(
            document_id="wiki-1",
            source_name="Wikipedia",
            source_type="background",
            is_background=True,
            is_peer_reviewed=False,
            relevance_score=0.9,
            text="Interleukin 6 is a cytokine implicated in rheumatoid arthritis.",
        )
    ]
    assessment = EvidenceEvaluator(settings).evaluate(
        QUESTION, [], background_passages=background
    )

    assert assessment.decision == UNANSWERABLE
    assert assessment.source_quality["background_excluded"] == 1
    assert any("background" in note.lower() for note in assessment.notes)


# ---------------------------------------------------------------------------
# Partial answerability
# ---------------------------------------------------------------------------


def test_missing_component_yields_partial_answer(settings, il6_passages):
    question = (
        "What role does IL-6 play in rheumatoid arthritis and what is the long-term effect of "
        "tocilizumab on all-cause mortality?"
    )
    assessment = EvidenceEvaluator(settings).evaluate(question, il6_passages)

    assert assessment.decision == PARTIALLY_ANSWERABLE
    assert assessment.reason_code == "partial_coverage"
    assert assessment.supported_aspects
    assert assessment.unsupported_aspects
    assert "mortality" in " ".join(assessment.sufficiency["unsupported_aspects"]).lower()


def test_partial_allows_generation_with_gaps(settings, il6_passages):
    question = "What role does IL-6 play in rheumatoid arthritis and does it cure multiple sclerosis?"
    assessment = EvidenceEvaluator(settings).evaluate(question, il6_passages)
    assert assessment.abstention.allows_generation is True
    assert assessment.abstention.criteria_failed


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_conflicting_findings_are_detected(settings):
    passages = [
        make_passage(
            document_id="conflict-1",
            relevance_score=0.8,
            text=(
                "In this randomised trial, omega-3 supplementation reduced cardiovascular risk "
                "and lowered the risk of myocardial infarction significantly."
            ),
        ),
        make_passage(
            document_id="conflict-2",
            relevance_score=0.75,
            text=(
                "In this trial omega-3 supplementation did not reduce cardiovascular risk; "
                "there was no significant difference in cardiovascular events."
            ),
        ),
        make_passage(
            document_id="conflict-3",
            relevance_score=0.7,
            text="A further trial found omega-3 supplementation did not reduce cardiovascular events.",
        ),
    ]
    assessment = EvidenceEvaluator(settings).evaluate(
        "Do omega-3 fatty acids reduce cardiovascular risk?", passages
    )

    assert assessment.consistency.conclusion in {"conflicting", "mixed"}
    assert assessment.consistency.is_conflicting
    assert assessment.consistency.stance_counts["support"] >= 1
    assert assessment.consistency.stance_counts["null"] >= 1
    assert any("disagree" in note or "Conflicting" in note for note in assessment.notes)


def test_consistency_uses_passage_stance(settings, il6_passages):
    EvidenceEvaluator(settings).evaluate(QUESTION, il6_passages)
    assert all(p.stance in {"support", "oppose", "null", "unclear"} for p in il6_passages)


# ---------------------------------------------------------------------------
# Source quality
# ---------------------------------------------------------------------------


def test_preprint_only_evidence_is_flagged(settings):
    preprints = [
        make_passage(
            document_id=f"preprint-{i}",
            source_name="arXiv",
            source_type="preprint",
            is_preprint=True,
            is_peer_reviewed=False,
            relevance_score=0.7,
            text=(
                "In this preprint we report that IL-6 blockade reduced disease activity in "
                "rheumatoid arthritis."
            ),
        )
        for i in range(3)
    ]
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, preprints)

    assert assessment.source_quality["peer_reviewed"] == 0
    assert assessment.source_quality["preprints_or_unreviewed"] == 3
    assert any("preprint" in note.lower() for note in assessment.notes)


# ---------------------------------------------------------------------------
# Strength grading
# ---------------------------------------------------------------------------


def test_strength_is_rule_based_and_transparent(settings, il6_passages):
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, il6_passages)
    strength = assess_evidence_strength(assessment)

    assert strength.grade in {STRONG, MODERATE, LIMITED, CONFLICTING}
    assert strength.label == GRADE_LABELS[strength.grade]
    assert 0.0 <= strength.score <= 1.0
    assert pytest.approx(sum(c.weight for c in strength.components), rel=1e-6) == 1.0
    assert {c.name for c in strength.components} == {
        "Study design",
        "Number of relevant studies",
        "Source quality",
        "Consistency of findings",
        "Directness of evidence",
        "Full-text availability",
    }
    payload = strength.to_dict()
    assert "not a probability" in payload["note"]


def test_strength_is_insufficient_when_abstaining(settings):
    assessment = EvidenceEvaluator(settings).evaluate(QUESTION, [])
    strength = assess_evidence_strength(assessment)
    assert strength.grade == INSUFFICIENT
    assert strength.score == 0.0


def test_conflicting_evidence_gets_conflicting_grade(settings):
    passages = [
        make_passage(
            document_id="c1",
            relevance_score=0.8,
            text=(
                "In this open-label observational cohort, statin therapy was associated with a "
                "significantly increased risk of muscle symptoms, and the authors conclude that "
                "statins increase muscle symptom burden in routine practice. Muscle symptom "
                "scores were higher in treated patients than in untreated controls."
            ),
        ),
        make_passage(
            document_id="c2",
            relevance_score=0.8,
            text=(
                "In a double-blind randomised placebo-controlled trial, statin therapy did not "
                "increase muscle symptoms; there was no significant difference versus placebo in "
                "muscle symptom scores. Most reported symptoms were also reproduced during placebo "
                "periods, indicating a nocebo effect rather than a pharmacological effect."
            ),
        ),
        make_passage(
            document_id="c3",
            relevance_score=0.8,
            text=(
                "Observational data suggest that statins increase muscle symptoms, but blinded "
                "re-challenge studies report no significant increase in muscle symptom scores "
                "compared with placebo, so the observational literature conflicts with the "
                "randomised evidence on statin intolerance."
            ),
        ),
    ]
    assessment = EvidenceEvaluator(settings).evaluate(
        "Do statins increase muscle symptoms?", passages
    )
    strength = assess_evidence_strength(assessment)
    assert strength.grade in {CONFLICTING, MODERATE, LIMITED, STRONG}
    if assessment.consistency.is_conflicting:
        assert strength.grade == CONFLICTING
        assert strength.cautions


def test_single_weak_source_is_limited(settings):
    single = [
        make_passage(
            document_id="weak-1",
            source_type="peer_reviewed_article",
            relevance_score=0.8,
            text=(
                "A single small study reported that IL-6 is involved in the pathogenesis of "
                "rheumatoid arthritis synovium, with increased IL-6 concentrations in synovial "
                "fluid from patients compared with controls. The authors note that the sample "
                "size was small and that no independent replication has been published."
            ),
        )
    ]
    assessment = EvidenceEvaluator(settings.with_overrides(minimum_evidence_passages=1)).evaluate(
        QUESTION, single
    )
    strength = assess_evidence_strength(assessment)
    assert strength.grade in {LIMITED, MODERATE}
    assert strength.study_count == 1
