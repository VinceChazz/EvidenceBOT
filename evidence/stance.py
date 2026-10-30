"""Rule-based stance detection for retrieved passages.

For each passage EvidenceBot asks: does it report an effect *in the direction
the question asks about*, *the opposite direction*, or *no effect*? This is
lexical and deterministic — it is a transparent signal for the consistency
dimension, never a claim about scientific truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from processing.models import Passage
from utils.helpers import clean_text, split_sentences

#: Phrases that indicate a positive/directional finding.
POSITIVE_CUES = (
    "was associated with", "were associated with", "associated with a lower",
    "associated with lower", "associated with higher", "associated with increased",
    "reduced", "reduces", "reduce", "increase", "increased", "improved", "improves",
    "effective", "efficacy", "benefit", "beneficial", "significantly", "significant",
    "caused", "causes", "promotes", "drives", "contributes to", "risk factor",
    "protective", "lower risk", "higher risk", "we found", "demonstrated",
    "showed", "concluded", "conclusion",
)

#: Phrases that indicate a null / inconclusive finding.
NULL_CUES = (
    "no significant", "not significant", "no association", "no evidence",
    "did not", "was not associated", "were not associated", "no reduction",
    "no difference", "no effect", "did not reduce", "did not increase",
    "failed to", "inconclusive", "remains unclear", "remains unknown",
    "cannot be determined", "not possible", "no study has", "conflicting",
    "attenuated", "no protective", "little or no", "does not",
)

#: Phrases that indicate a decreasing/protective direction.
NEGATIVE_DIRECTION_CUES = (
    "reduced", "reduces", "reduce", "decreased", "decreases", "lower", "lowered",
    "inhibited", "inhibits", "suppressed", "protective",
)

#: Verbs in the question that imply an expected increase / causal direction.
INCREASE_EXPECTATION_CUES = (
    "cause", "causes", "increase", "increases", "worsen", "worsens", "risk factor",
    "drive", "drives", "promote", "promotes", "contribute",
)

#: Verbs in the question that imply an expected reduction direction.
DECREASE_EXPECTATION_CUES = (
    "reduce", "reduces", "decrease", "decreases", "prevent", "prevents", "lower",
    "protective", "improve", "improves", "treat", "cure",
)


@dataclass
class StanceResult:
    stance: str  # "support" | "oppose" | "null" | "unclear"
    evidence_sentence: Optional[str] = None
    matched_cues: List[str] = field(default_factory=list)
    confidence: float = 0.0


def _find_cues(text: str, cues: Sequence[str]) -> List[str]:
    lowered = text.lower()
    return [cue for cue in cues if cue in lowered]


def question_expectation(question: str) -> str:
    """Infer the direction the question asks about.

    Returns "increase" (question expects a positive/causal association),
    "decrease" (question expects a protective/reducing effect) or "any".
    """

    q = (question or "").lower()
    increase = _find_cues(q, INCREASE_EXPECTATION_CUES)
    decrease = _find_cues(q, DECREASE_EXPECTATION_CUES)
    if increase and not decrease:
        return "increase"
    if decrease and not increase:
        return "decrease"
    return "any"


def detect_stance(text: str, question: str = "") -> StanceResult:
    """Classify a passage's reported finding relative to the question."""

    text = clean_text(text)
    if not text:
        return StanceResult(stance="unclear")

    null_cues = _find_cues(text, NULL_CUES)
    positive_cues = _find_cues(text, POSITIVE_CUES)
    negative_cues = _find_cues(text, NEGATIVE_DIRECTION_CUES)
    expected = question_expectation(question)

    sentences = split_sentences(text)
    best_sentence = None
    best_score = 0.0
    for sentence in sentences:
        score = len(_find_cues(sentence, NULL_CUES)) * 1.5 + len(
            _find_cues(sentence, POSITIVE_CUES)
        )
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if null_cues and len(null_cues) >= max(1, len(positive_cues) - 1):
        # A null result dominates when the paper explicitly reports no effect.
        return StanceResult(
            stance="null",
            evidence_sentence=best_sentence,
            matched_cues=null_cues[:4],
            confidence=0.7,
        )

    if not positive_cues:
        return StanceResult(stance="unclear", evidence_sentence=best_sentence, confidence=0.2)

    if expected == "any":
        return StanceResult(
            stance="support",
            evidence_sentence=best_sentence,
            matched_cues=positive_cues[:4],
            confidence=0.5,
        )

    has_negative_direction = bool(negative_cues)
    if expected == "increase":
        stance = "oppose" if has_negative_direction else "support"
    elif expected == "decrease":
        stance = "support" if has_negative_direction else "oppose"
    else:  # pragma: no cover - defensive
        stance = "support"

    return StanceResult(
        stance=stance,
        evidence_sentence=best_sentence,
        matched_cues=(negative_cues if has_negative_direction else positive_cues)[:4],
        confidence=0.55,
    )


@dataclass
class ConsistencyResult:
    conclusion: str = "inconclusive"  # consistent_support | consistent_null | conflicting | mixed | inconclusive
    stance_counts: Dict[str, int] = field(default_factory=dict)
    agreement_ratio: float = 0.0
    conflicting_passages: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_conflicting(self) -> bool:
        return self.conclusion in {"conflicting", "mixed"}

    @property
    def label(self) -> str:
        return {
            "consistent_support": "Consistent supporting findings",
            "consistent_null": "Consistent null (no-effect) findings",
            "conflicting": "Conflicting findings",
            "mixed": "Mixed findings (positive and null)",
            "inconclusive": "Inconclusive / insufficient to compare",
        }.get(self.conclusion, self.conclusion)


def assess_consistency(passages: Sequence[Passage], question: str) -> ConsistencyResult:
    """Compare reported findings across the supporting passages."""

    result = ConsistencyResult()
    if not passages:
        result.notes.append("No passages available to compare.")
        return result

    counts = {"support": 0, "oppose": 0, "null": 0, "unclear": 0}
    for passage in passages:
        stance = detect_stance(passage.text, question)
        passage.stance = stance.stance
        counts[stance.stance] += 1
    result.stance_counts = counts

    comparable = counts["support"] + counts["oppose"] + counts["null"]
    if comparable == 0:
        result.conclusion = "inconclusive"
        result.agreement_ratio = 0.0
        result.notes.append("No directional findings were detected in the passages.")
        return result

    if counts["support"] and counts["oppose"]:
        result.conclusion = "conflicting"
        result.agreement_ratio = max(counts["support"], counts["oppose"]) / comparable
        result.notes.append(
            "At least one study reports the opposite direction of effect to another study."
        )
    elif counts["support"] and counts["null"]:
        result.conclusion = "mixed"
        result.agreement_ratio = counts["support"] / comparable
        result.notes.append(
            "Some studies report a positive finding while others report no significant effect."
        )
    elif counts["oppose"] and counts["null"]:
        result.conclusion = "mixed"
        result.agreement_ratio = counts["oppose"] / comparable
        result.notes.append("Directional findings are mixed with null results.")
    elif counts["support"]:
        result.conclusion = "consistent_support"
        result.agreement_ratio = counts["support"] / comparable
    elif counts["oppose"]:
        result.conclusion = "consistent_support"
        # Consistently opposite to the question's implied direction is still a
        # consistent evidence base — it simply answers the question "no".
        result.agreement_ratio = counts["oppose"] / comparable
        result.notes.append(
            "Studies consistently report the opposite direction to the one implied by the question."
        )
    else:
        result.conclusion = "consistent_null"
        result.agreement_ratio = counts["null"] / comparable
        result.notes.append(
            "The retrieved studies consistently report no significant effect. "
            "This is not proof that the effect does not exist."
        )

    if result.is_conflicting:
        result.conflicting_passages = [
            p.passage_id for p in passages if p.stance in {"support", "oppose"}
        ]
    return result


def findings_table(passages: Sequence[Passage], question: str) -> List[Dict[str, object]]:
    """Build the evidence-comparison table used by the UI."""

    rows: List[Dict[str, object]] = []
    for passage in passages:
        stance = detect_stance(passage.text, question)
        rows.append(
            {
                "publication": passage.title or passage.document_id,
                "source": passage.source_name,
                "study_type": passage.source_type.replace("_", " "),
                "year": passage.publication_year,
                "main_finding": clean_text(stance.evidence_sentence or passage.text)[:320],
                "stance": stance.stance,
                "supporting_or_contradicting": {
                    "support": "supports the question's direction",
                    "oppose": "contradicts the question's direction",
                    "null": "reports no significant effect",
                    "unclear": "unclear / descriptive only",
                }[stance.stance],
                "limitations": _limitations(passage),
            }
        )
    return rows


def _limitations(passage: Passage) -> str:
    notes: List[str] = []
    if passage.is_preprint:
        notes.append("Preprint — not peer reviewed.")
    if not passage.is_peer_reviewed:
        notes.append("Not peer reviewed.")
    if passage.provenance == "title_only":
        notes.append("Metadata only (no abstract available).")
    if passage.source_type in {"cohort_study", "case_control_study", "observational_study"}:
        notes.append("Observational design — cannot establish causation.")
    if passage.source_type in {"review", "systematic_review", "meta_analysis"}:
        notes.append("Evidence is indirect (secondary analysis of other studies).")
    if passage.is_sample:
        notes.append("Illustrative offline sample record.")
    return " ".join(notes) or "Not reported in the retrieved passage."
