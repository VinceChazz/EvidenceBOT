"""Extractive evidence digest.

Used only when the answer LLM is unavailable. It does **not** synthesise or
paraphrase: it quotes the strongest verbatim sentence from each cited passage
with its citation number, so nothing can be hallucinated. The UI labels it
explicitly as an extractive digest rather than a generated answer.
"""

from __future__ import annotations

from typing import List, Sequence

from evidence.stance import detect_stance
from processing.models import Passage
from utils.helpers import clean_text, truncate


def build_extractive_digest(
    question: str, passages: Sequence[Passage], max_items: int = 5
) -> List[dict]:
    """Return verbatim evidence quotes, one per citation number."""

    items: List[dict] = []
    for index, passage in enumerate(passages[:max_items], start=1):
        stance = detect_stance(passage.text, question)
        quote = clean_text(stance.evidence_sentence or passage.text)
        if not quote:
            continue
        items.append(
            {
                "citation": index,
                "quote": truncate(quote, 420),
                "title": passage.title,
                "source": passage.source_name,
                "year": passage.publication_year,
                "stance": stance.stance,
            }
        )
    return items


def render_extractive_markdown(items: Sequence[dict]) -> str:
    if not items:
        return "_No quotable evidence sentence was found._"
    lines = [
        "**Extractive evidence digest** — verbatim sentences from the retrieved "
        "passages (no language model was used to write this):",
        "",
    ]
    for item in items:
        lines.append(f"- {item['quote']} [{item['citation']}]")
    return "\n".join(lines)
