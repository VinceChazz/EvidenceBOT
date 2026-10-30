"""Shared text-processing and formatting helpers."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Stop words / vocabulary
# ---------------------------------------------------------------------------

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "cannot", "could", "couldn", "did", "didn",
    "do", "does", "doesn", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i",
    "if", "in", "into", "is", "isn", "it", "its", "itself", "just", "me", "might",
    "more", "most", "mustn", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan", "she", "should", "shouldn", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn", "we", "were", "weren", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won", "would", "wouldn", "you",
    "your", "yours", "yourself", "yourselves", "also", "may", "many", "much", "use",
    "used", "using", "one", "two", "study", "studies", "result", "results", "paper",
    "papers", "research", "evidence", "question", "role", "play", "plays",
}

#: Words that make a question an absolute/universal claim.
ABSOLUTE_CLAIM_CUES: Set[str] = {
    "guarantee", "guarantees", "guaranteed", "never", "always", "every", "all",
    "completely", "totally", "cure", "cures", "cured", "eliminate", "eliminates",
    "prevent", "prevents", "impossible", "definitively", "prove", "proves", "proof",
    "universally", "100%", "certainly", "undoubtedly",
}

CAUSAL_CUES: Set[str] = {
    "cause", "causes", "caused", "causing", "lead", "leads", "reduce", "reduces",
    "increase", "increases", "improve", "improves", "effect", "effects", "impact",
    "affect", "affects", "prevent", "prevents", "treat", "treats", "risk",
}

QUESTION_STARTERS = {
    "what", "how", "why", "does", "do", "is", "are", "can", "could", "should",
    "would", "which", "when", "who", "will",
}


def strip_html(text: Optional[str]) -> str:
    """Remove HTML/JATS markup and decode entities."""

    if not text:
        return ""
    # Drop tags but keep sentence boundaries.
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    cleaned = re.sub(r"</(p|sec|div|title|abstract|jats:p|jats:sec)>", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return clean_text(cleaned)


def clean_text(text: Optional[str]) -> str:
    """Normalise whitespace/unicode artefacts from API payloads."""

    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ").replace("\u2009", " ").replace("\u2019", "'")
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def clean_multiline(text: Optional[str]) -> str:
    """Normalise whitespace but preserve line breaks (for Markdown answers)."""

    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in value.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines.append(line)
    # Collapse >2 blank lines.
    value = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return value.strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


#: Keeps hyphenated biomedical terms ("il-6", "tnf-alpha", "sars-cov-2") intact.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def simple_stem(token: str) -> str:
    """Very light suffix stripping (enough to match plurals/inflections)."""

    for suffix in ("ational", "iveness", "ization", "isation", "ations", "ation",
                   "ities", "ively", "ement", "ness", "ments", "ities", "ical",
                   "ies", "ing", "ers", "ed", "es", "ly", "al", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str, keep_stopwords: bool = False) -> List[str]:
    """Return stemmed content terms (stop words removed by default)."""

    terms: List[str] = []
    for token in tokenize(text):
        if token.isdigit() or len(token) < 2:
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        if "-" in token:
            # Keep the hyphenated form, and the collapsed form (IL6 / IL-6).
            terms.append(token)
            collapsed = token.replace("-", "")
            if len(collapsed) >= 3 and collapsed not in STOPWORDS:
                terms.append(collapsed)
            continue
        terms.append(simple_stem(token))
    return terms


def unique_terms(text: str) -> Set[str]:
    return set(content_terms(text))


def term_overlap(source_text: str, target_terms: Iterable[str]) -> float:
    """Share of ``target_terms`` present in ``source_text`` (0-1)."""

    targets = {t for t in target_terms if t}
    if not targets:
        return 0.0
    present = unique_terms(source_text)
    return len(targets & present) / len(targets)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def stable_hash(*parts: Any) -> str:
    payload = "||".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def normalise_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    value = doi.strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value)
    return value or None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM response."""

    if not text:
        return None
    text = text.strip()
    # Strip Markdown code fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def split_question_components(question: str) -> List[str]:
    """Split a compound research question into its sub-questions.

    Only splits on explicit coordination or question marks, never inside a
    single clause, so wording like "role of IL-6 in rheumatoid arthritis" stays
    intact.
    """

    text = (question or "").strip().rstrip("?")
    if not text:
        return []
    parts = re.split(r"\?|;\s*|\s+and\s+(?=(?:does|do|is|are|can|could|what|how|why|whether)\b)",
                     text, flags=re.I)
    components = [clean_text(p).strip(" ,.") for p in parts if p and len(clean_text(p)) > 8]
    return components or [clean_text(text)]


def question_concepts(question: str) -> List[str]:
    """Key concepts of a question (deduplicated, order preserved)."""

    seen: List[str] = []
    for term in content_terms(question):
        if term not in seen:
            seen.append(term)
    return seen


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def year_sort_value(value: Any) -> int:
    parsed = safe_int(value)
    return parsed if parsed else 0


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
