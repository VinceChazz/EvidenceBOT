"""Citation validation and claim-to-source verification.

Every citation produced by the answer LLM is checked against the passages that
were actually retrieved:

* citation numbers must exist in the evidence set,
* each number must map to the correct publication metadata,
* the cited passage must lexically support the sentence that cites it,
* the answer must not contain URLs or DOIs that were not retrieved.

Invalid citation markers are stripped before anything is displayed, and a
response that cannot be verified is never presented as fully evidence-verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from config import Settings
from processing.models import Passage
from utils.helpers import clean_text, content_terms, split_sentences, unique_terms
from utils.logging_config import get_logger

logger = get_logger("evidence.citations")

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*[,;–—-]\s*\d+)*)\]")
URL_PATTERN = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]]+", re.IGNORECASE)

#: Sentences that are structural rather than substantive.
NON_CLAIM_MARKERS = (
    "insufficient evidence",
    "evidence assessment",
    "limitations",
    "unanswered",
    "sources",
    "references",
    "confidence",
)


#: Matches markers written *after* the sentence's full stop, e.g. "… arthritis. [1]"
_TRAILING_MARKERS = re.compile(
    r"([.!?])\s*((?:\[\d+(?:\s*[,;–—-]\s*\d+)*\]\s*)+)(?=\s|$)"
)


def normalise_citation_placement(text: str) -> str:
    """Move markers that follow a full stop to before it.

    ``"… arthritis. [1] Next sentence."`` becomes ``"… arthritis [1]. Next sentence."``
    so each marker stays attached to the claim it supports.
    """

    def _swap(match: re.Match) -> str:
        markers = " ".join(match.group(2).split())
        return f" {markers}{match.group(1)} "

    normalised = _TRAILING_MARKERS.sub(_swap, text or "")
    return re.sub(r"[ \t]{2,}", " ", normalised).strip()


def parse_citation_markers(text: str) -> List[int]:
    """Extract all citation numbers from a text, expanding ``[1-3]`` ranges."""

    numbers: List[int] = []
    for match in CITATION_PATTERN.finditer(text or ""):
        group = match.group(1)
        for chunk in re.split(r"[,;]", group):
            chunk = chunk.strip()
            if not chunk:
                continue
            range_match = re.match(r"^(\d+)\s*[–—-]\s*(\d+)$", chunk)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if 0 < start <= end <= start + 20:
                    numbers.extend(range(start, end + 1))
                continue
            if chunk.isdigit():
                numbers.append(int(chunk))
    # De-duplicate while keeping order.
    seen: Set[int] = set()
    ordered: List[int] = []
    for number in numbers:
        if number not in seen:
            seen.add(number)
            ordered.append(number)
    return ordered


def strip_citation_markers(text: str, remove: Iterable[int]) -> str:
    """Remove specific citation numbers from a text, keeping valid ones."""

    remove_set = {int(n) for n in remove}

    def _replace(match: re.Match) -> str:
        kept: List[str] = []
        for chunk in re.split(r"[,;]", match.group(1)):
            chunk = chunk.strip()
            if chunk.isdigit() and int(chunk) in remove_set:
                continue
            kept.append(chunk)
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    cleaned = CITATION_PATTERN.sub(_replace, text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    return cleaned.strip()


@dataclass
class CitationRecord:
    number: int
    passage_id: str
    document_id: str
    title: str
    source_name: str
    source_type: str
    authors: List[str]
    publication_year: Optional[int]
    doi: Optional[str]
    pmid: Optional[str]
    url: Optional[str]
    journal: Optional[str]
    passage_text: str
    relevance_score: float
    is_preprint: bool
    is_sample: bool
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "title": self.title,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "authors": self.authors,
            "publication_year": self.publication_year,
            "doi": self.doi,
            "pmid": self.pmid,
            "url": self.url,
            "journal": self.journal,
            "relevance_score": round(self.relevance_score, 4),
            "is_preprint": self.is_preprint,
            "is_sample": self.is_sample,
            "provenance": self.provenance,
        }


@dataclass
class ClaimCheck:
    claim: str
    citations: List[int]
    support_score: float
    supported: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "citations": self.citations,
            "support_score": round(self.support_score, 3),
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass
class CitationValidationReport:
    is_valid: bool = True
    citation_map: Dict[int, CitationRecord] = field(default_factory=dict)
    used_citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    unsupported_claims: List[ClaimCheck] = field(default_factory=list)
    uncited_claims: List[str] = field(default_factory=list)
    claim_checks: List[ClaimCheck] = field(default_factory=list)
    invented_urls: List[str] = field(default_factory=list)
    invented_dois: List[str] = field(default_factory=list)
    sanitized_answer: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def citation_count(self) -> int:
        return len(self.used_citations)

    @property
    def all_claims_supported(self) -> bool:
        return not self.unsupported_claims and not self.uncited_claims

    @property
    def verification_label(self) -> str:
        if self.is_valid and self.all_claims_supported:
            return "Fully evidence-verified"
        if self.is_valid:
            return "Citations verified; some claims could not be cross-checked"
        return "Citations could not be fully verified"

    def needs_regeneration(self, max_unsupported_ratio: float = 0.34) -> bool:
        if self.invalid_citations or self.invented_urls or self.invented_dois:
            return True
        total = max(1, len(self.claim_checks))
        return len(self.unsupported_claims) / total > max_unsupported_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "verification_label": self.verification_label,
            "used_citations": self.used_citations,
            "invalid_citations": self.invalid_citations,
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "uncited_claims": self.uncited_claims,
            "invented_urls": self.invented_urls,
            "invented_dois": self.invented_dois,
            "claims_checked": len(self.claim_checks),
            "notes": self.notes,
        }


def build_citation_records(evidence: Sequence[Passage]) -> List[CitationRecord]:
    """Number the evidence passages 1..n in the order they will be shown."""

    records: List[CitationRecord] = []
    for index, passage in enumerate(evidence, start=1):
        records.append(
            CitationRecord(
                number=index,
                passage_id=passage.passage_id,
                document_id=passage.document_id,
                title=passage.title or "(untitled)",
                source_name=passage.source_name,
                source_type=passage.source_type,
                authors=list(passage.authors),
                publication_year=passage.publication_year,
                doi=passage.doi,
                pmid=passage.pmid,
                url=passage.url,
                journal=passage.journal,
                passage_text=passage.text,
                relevance_score=passage.relevance_score,
                is_preprint=passage.is_preprint,
                is_sample=passage.is_sample,
                provenance=passage.provenance,
            )
        )
    return records


def _is_substantive_claim(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(marker in lowered for marker in NON_CLAIM_MARKERS):
        return False
    if len(sentence.split()) < 6:
        return False
    return len(content_terms(sentence)) >= 4


NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|mg|ml|iu|nmol|years?|patients?)?\b", re.IGNORECASE)


def _years_only(token: str) -> bool:
    digits = re.sub(r"\D", "", token)
    return len(digits) == 4 and 1900 <= int(digits) <= 2100


def _unsupported_numbers(claim: str, cited_texts: Sequence[str]) -> List[str]:
    """Specific numeric values in a claim that no cited passage mentions."""

    haystack = " ".join(cited_texts)
    haystack_numbers = {re.sub(r"\s+", "", m) for m in NUMBER_PATTERN.findall(haystack)}
    missing: List[str] = []
    for match in NUMBER_PATTERN.findall(claim):
        token = re.sub(r"\s+", "", match)
        if _years_only(token) or len(re.sub(r"\D", "", token)) < 2:
            continue
        if token not in haystack_numbers:
            missing.append(match.strip())
    return missing


#: Claim vocabularies that assert an absolute or causal relationship. If the model
#: uses one of these but the cited passage never does, the claim overstates the
#: evidence and is rejected.
STRONG_CLAIM_GROUPS = (
    ("cure", "cures", "cured", "curing"),
    ("cause", "causes", "caused", "causing"),
    ("prove", "proves", "proved", "proof"),
    ("guarantee", "guarantees", "guaranteed"),
    ("eliminate", "eliminates", "eliminated"),
    ("prevent", "prevents", "prevented"),
    ("always", "never", "completely", "totally", "definitively", "certainly"),
    ("established", "definitive", "conclusive"),
)


def _unsupported_strength_claims(claim: str, cited_texts: Sequence[str]) -> List[str]:
    """Absolute/causal wording in a claim that the cited passage never uses."""

    claim_lower = claim.lower()
    haystack = " ".join(cited_texts).lower()
    flagged: List[str] = []
    for group in STRONG_CLAIM_GROUPS:
        used = [word for word in group if re.search(rf"\b{re.escape(word)}\b", claim_lower)]
        if not used:
            continue
        if any(re.search(rf"\b{re.escape(word)}\b", haystack) for word in group):
            continue
        flagged.append(used[0])
    return flagged


def _claim_support_score(claim: str, cited_texts: Sequence[str]) -> float:
    claim_terms = set(content_terms(claim))
    if not claim_terms:
        return 0.0
    support_terms: Set[str] = set()
    for text in cited_texts:
        support_terms |= unique_terms(text)
    return len(claim_terms & support_terms) / len(claim_terms)


def validate_citations(
    answer_text: str,
    evidence: Sequence[Passage],
    settings: Optional[Settings] = None,
    extra_claims: Optional[Sequence[str]] = None,
) -> CitationValidationReport:
    """Validate every citation and claim in a generated answer."""

    settings = settings or Settings()
    answer_text = normalise_citation_placement(answer_text or "")
    report = CitationValidationReport(sanitized_answer=answer_text or "")
    records = build_citation_records(evidence)
    report.citation_map = {record.number: record for record in records}
    valid_numbers = set(report.citation_map)

    allowed_urls = {
        (record.url or "").rstrip("/").lower()
        for record in records
        if record.url
    }
    allowed_dois = {(record.doi or "").lower() for record in records if record.doi}

    # ------------------------------------------------------------- citations
    cited = parse_citation_markers(answer_text or "")
    report.invalid_citations = sorted({n for n in cited if n not in valid_numbers})
    report.used_citations = sorted({n for n in cited if n in valid_numbers})

    if report.invalid_citations:
        report.notes.append(
            "The response cited source id(s) that were never retrieved: "
            + ", ".join(str(n) for n in report.invalid_citations)
            + ". They were removed."
        )
        report.is_valid = False

    # ------------------------------------------------- URLs / DOIs invented?
    for url in URL_PATTERN.findall(answer_text or ""):
        token = url.rstrip(".,;").rstrip("/").lower()
        if token not in allowed_urls:
            report.invented_urls.append(url)
    for doi in DOI_PATTERN.findall(answer_text or ""):
        if doi.rstrip(".,;").lower() not in allowed_dois:
            report.invented_dois.append(doi)
    if report.invented_urls or report.invented_dois:
        report.is_valid = False
        report.notes.append(
            "The response contained URLs/DOIs that do not belong to any retrieved record."
        )

    # ------------------------------------------------------------ claim scan
    claim_texts: List[str] = []
    for sentence in split_sentences(clean_text(answer_text or "")):
        if _is_substantive_claim(sentence):
            claim_texts.append(sentence)
    for claim in extra_claims or []:
        claim = clean_text(claim)
        if claim and _is_substantive_claim(claim):
            claim_texts.append(claim)

    for claim in claim_texts:
        numbers = [n for n in parse_citation_markers(claim) if n in valid_numbers]
        if not numbers:
            report.uncited_claims.append(claim)
            report.claim_checks.append(
                ClaimCheck(
                    claim=claim,
                    citations=[],
                    support_score=0.0,
                    supported=False,
                    reason="No valid citation attached to a substantive claim.",
                )
            )
            continue
        cited_texts = [report.citation_map[n].passage_text for n in numbers]
        score = _claim_support_score(claim, cited_texts)
        missing_numbers = _unsupported_numbers(claim, cited_texts)
        overstated = _unsupported_strength_claims(claim, cited_texts)
        supported = (
            score >= settings.claim_support_threshold
            and not missing_numbers
            and not overstated
        )
        if supported:
            reason = "Cited passage supports the claim's key terms."
        elif overstated:
            reason = (
                "The claim asserts a causal/absolute relationship (" + ", ".join(overstated)
                + ") that the cited passage does not state."
            )
        elif missing_numbers:
            reason = (
                "The claim states specific value(s) not present in the cited passage: "
                + ", ".join(missing_numbers[:4])
            )
        else:
            reason = "Cited passage does not contain the claim's key terms."
        check = ClaimCheck(
            claim=claim,
            citations=numbers,
            support_score=score,
            supported=supported,
            reason=reason,
        )
        report.claim_checks.append(check)
        if not supported:
            report.unsupported_claims.append(check)

    # ------------------------------------------------------------- sanitise
    if report.invalid_citations:
        report.sanitized_answer = strip_citation_markers(
            answer_text or "", report.invalid_citations
        )
    if report.invented_urls:
        for url in report.invented_urls:
            report.sanitized_answer = report.sanitized_answer.replace(url, "[link removed]")

    if report.uncited_claims:
        report.notes.append(
            f"{len(report.uncited_claims)} substantive claim(s) had no citation."
        )
    if report.unsupported_claims:
        report.notes.append(
            f"{len(report.unsupported_claims)} claim(s) were not supported by their cited passage."
        )
        report.is_valid = False

    logger.info(
        "citation validation: %d citations, %d invalid, %d unsupported, %d uncited",
        len(report.used_citations),
        len(report.invalid_citations),
        len(report.unsupported_claims),
        len(report.uncited_claims),
    )
    return report


def citation_map_frame_rows(records: Sequence[CitationRecord]) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in records]
