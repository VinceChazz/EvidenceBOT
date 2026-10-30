"""Prompts for grounded, citation-only answer generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from evidence.evidence_evaluator import EvidenceAssessment
from processing.models import Passage
from utils.helpers import truncate

SYSTEM_PROMPT = """You are EvidenceBot, a scientific evidence synthesis assistant.

ABSOLUTE RULES — violating any of them makes your answer invalid:
1. Answer EXCLUSIVELY using the retrieved evidence passages supplied in the user message.
2. Never introduce scientific claims that are not present in the supplied evidence.
3. Never invent publications, authors, dates, DOIs, URLs or citation numbers. You may only
   cite source ids that appear in the evidence block.
4. Use inline numbered citations in square brackets, e.g. [1], [2] or [1,3].
5. Cite every substantive scientific claim. Use multiple citations when several sources agree.
6. If the evidence contains conflicting findings, state the disagreement explicitly and do not
   present either side as universally established.
7. Distinguish established findings from preliminary findings (preprints, small or
   observational studies, single studies).
8. State the limitations of the available literature.
9. If part of the question cannot be answered from the evidence, say so explicitly instead of
   guessing. Never fill gaps with your pretrained knowledge.
10. Do not report confidence percentages. Evidence strength is assessed separately by the system.

STYLE
- Neutral, precise, scientific register. No marketing language.
- Prefer concrete statements, effect directions and study designs over vague summaries.
- Keep the answer under 250 words unless the question requires more.

OUTPUT
Return a single JSON object with exactly these keys:
{
  "answer": "string — the answer with inline [n] citations",
  "supporting_claims": [{"claim": "string", "source_ids": [1, 2]}],
  "evidence_assessment": "string — one or two sentences about how strong/direct/consistent the evidence is",
  "limitations": ["string"],
  "unanswered_questions": ["string"],
  "conflicts": ["string — explicit description of any disagreement between studies; empty if none"]
}
Return JSON only. No markdown fences, no commentary outside the JSON object."""


def format_evidence_block(passages: Sequence[Passage], max_chars: int = 1800) -> str:
    """Render the numbered evidence block (numbers match the UI citation cards)."""

    if not passages:
        return "(no evidence passages)"
    blocks: List[str] = []
    seen_documents: Dict[str, int] = {}
    for index, passage in enumerate(passages, start=1):
        authors = ", ".join(passage.authors[:3]) + (" et al." if len(passage.authors) > 3 else "")
        meta_bits = [
            f"source: {passage.source_name}",
            f"type: {passage.source_type.replace('_', ' ')}",
            f"year: {passage.publication_year or 'n.d.'}",
        ]
        if authors:
            meta_bits.append(f"authors: {authors}")
        if passage.doi:
            meta_bits.append(f"doi: {passage.doi}")
        if passage.pmid:
            meta_bits.append(f"pmid: {passage.pmid}")
        if passage.is_preprint:
            meta_bits.append("PREPRINT — not peer reviewed")
        if passage.is_sample:
            meta_bits.append("ILLUSTRATIVE SAMPLE RECORD — not a real publication")
        duplicate = seen_documents.get(passage.document_id)
        if duplicate:
            meta_bits.append(f"same publication as [{duplicate}]")
        seen_documents.setdefault(passage.document_id, index)
        blocks.append(
            f"[{index}] {passage.title or '(untitled)'}\n"
            f"    ({'; '.join(meta_bits)})\n"
            f"    PASSAGE: {truncate(passage.text, max_chars)}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    passages: Sequence[Passage],
    assessment: EvidenceAssessment,
    partial: bool = False,
) -> str:
    """User message containing the question, the evidence and the constraints."""

    supported = assessment.supported_aspects
    missing = assessment.unsupported_aspects
    consistency = assessment.consistency

    instructions: List[str] = []
    if partial:
        instructions.append(
            "The evidence gate returned PARTIALLY_ANSWERABLE. Answer ONLY the parts of the "
            "question that the evidence supports, and explicitly list the unanswered parts in "
            "'unanswered_questions'. Do not speculate about the missing parts."
        )
    if consistency.is_conflicting:
        instructions.append(
            "The retrieved studies disagree. Describe the disagreement explicitly in 'conflicts' "
            "and avoid presenting a single conclusion as established."
        )
    if missing:
        instructions.append("Question components without supporting evidence: " + "; ".join(missing))
    if supported:
        instructions.append("Question components with supporting evidence: " + "; ".join(supported))
    if assessment.absolute_claim:
        instructions.append(
            "The question contains absolute/universal wording. Do not affirm an absolute claim; "
            "report only what the studies actually show."
        )
    if assessment.source_quality.get("peer_reviewed", 0) == 0:
        instructions.append(
            "No peer-reviewed source is available: label the evidence as preliminary/preprint."
        )
    if any(p.is_sample for p in passages):
        instructions.append(
            "The evidence comes from an ILLUSTRATIVE offline sample corpus, not real published "
            "literature. Prefix the answer with a note that these are sample records."
        )

    return (
        f"RESEARCH QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE ({len(passages)} passage(s), numbered):\n"
        f"{format_evidence_block(passages)}\n\n"
        f"EVIDENCE GATE RESULT: {assessment.decision} ({assessment.reason_code})\n"
        f"CONSISTENCY OF FINDINGS: {consistency.label}\n"
        f"ADDITIONAL CONSTRAINTS:\n- " + "\n- ".join(instructions or ["none"]) + "\n\n"
        "Produce the JSON object described in the system prompt. Cite only the numbers above."
    )


def build_regeneration_prompt(
    previous: str,
    issues: Dict[str, Any],
    passages: Sequence[Passage],
) -> str:
    """Ask the model to repair a response that failed citation validation."""

    invalid = issues.get("invalid_citations") or []
    unsupported = issues.get("unsupported_claims") or []
    uncited = issues.get("uncited_claims") or []
    invented = issues.get("invented_urls") or []

    lines = [
        "Your previous response failed citation verification and must be rewritten.",
        "",
        "PROBLEMS DETECTED:",
    ]
    if invalid:
        lines.append(f"- These citation ids do not exist: {invalid}. Only ids 1-{len(passages)} exist.")
    if invented:
        lines.append(f"- These URLs were not retrieved and must not appear: {invented}")
    for item in unsupported[:5]:
        claim = item.get("claim") if isinstance(item, dict) else str(item)
        lines.append(
            f"- This claim is not supported by the passage it cites and must be removed or "
            f"rephrased strictly from the evidence: {truncate(str(claim), 220)}"
        )
    for claim in uncited[:5]:
        lines.append(f"- This claim has no citation and needs one: {truncate(str(claim), 220)}")
    lines.extend(
        [
            "",
            "PREVIOUS RESPONSE:",
            truncate(previous, 4000),
            "",
            "EVIDENCE (unchanged, cite only these ids):",
            format_evidence_block(passages),
            "",
            "Rewrite the complete JSON object. Remove every claim you cannot support with a "
            "cited passage. Keep valid citations. Return JSON only.",
        ]
    )
    return "\n".join(lines)


def json_schema_description() -> str:
    return json.dumps(
        {
            "answer": "string",
            "supporting_claims": [{"claim": "string", "source_ids": ["int"]}],
            "evidence_assessment": "string",
            "limitations": ["string"],
            "unanswered_questions": ["string"],
            "conflicts": ["string"],
        },
        indent=2,
    )
