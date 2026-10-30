"""Deduplication of publications across sources.

The same paper is frequently returned by PubMed, Europe PMC and Crossref with
different identifiers. EvidenceBot merges those records while preserving the
strongest metadata and the best available abstract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

from processing.models import Document
from utils.helpers import normalise_doi
from utils.logging_config import get_logger

logger = get_logger("processing.dedup")

#: Lower rank wins when merging duplicate records (primary biomedical source first).
SOURCE_PRIORITY = {
    "PubMed": 0,
    "Europe PMC": 1,
    "Crossref": 2,
    "arXiv": 3,
    "Wikipedia": 4,
}

TITLE_SIMILARITY_THRESHOLD = 0.90


def _norm_title(title: str) -> str:
    value = (title or "").lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _title_tokens(title: str) -> set:
    return {t for t in _norm_title(title).split() if len(t) > 2 and not t.isdigit()}


def titles_match(a: str, b: str) -> bool:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    tokens_a, tokens_b = _title_tokens(a), _title_tokens(b)
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
        if overlap >= 0.95:
            return True
    return SequenceMatcher(None, na, nb).ratio() >= TITLE_SIMILARITY_THRESHOLD


@dataclass
class DeduplicationReport:
    input_count: int = 0
    output_count: int = 0
    duplicates_removed: int = 0
    merged_groups: List[List[str]] = field(default_factory=list)

    @property
    def duplicate_rate(self) -> float:
        if not self.input_count:
            return 0.0
        return self.duplicates_removed / self.input_count

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "duplicates_removed": self.duplicates_removed,
            "merged_groups": len(self.merged_groups),
        }


def _quality_score(doc: Document) -> Tuple[int, int, int, int, int]:
    """Sort key: higher is better metadata quality."""

    return (
        1 if doc.is_peer_reviewed else 0,
        1 if not doc.is_background else 0,
        1 if len(doc.text_passage or "") > 200 else 0,
        len(doc.text_passage or ""),
        1 if doc.doi or doc.pmid else 0,
    )


def merge_documents(primary: Document, secondary: Document) -> Document:
    """Merge ``secondary`` into ``primary``, filling gaps without overwriting."""

    merged = Document(**primary.to_dict())
    merged.authors = merged.authors or []

    if not merged.doi and secondary.doi:
        merged.doi = secondary.doi
    if not merged.pmid and secondary.pmid:
        merged.pmid = secondary.pmid
    if not merged.arxiv_id and secondary.arxiv_id:
        merged.arxiv_id = secondary.arxiv_id
    if not merged.publication_year and secondary.publication_year:
        merged.publication_year = secondary.publication_year
    if not merged.journal and secondary.journal:
        merged.journal = secondary.journal
    if not merged.publisher and secondary.publisher:
        merged.publisher = secondary.publisher
    if not merged.url and secondary.url:
        merged.url = secondary.url
    if len(secondary.authors) > len(merged.authors):
        merged.authors = list(secondary.authors)
    if len(secondary.text_passage) > len(merged.text_passage):
        merged.text_passage = secondary.text_passage
        merged.provenance = secondary.provenance
    merged.has_full_text = merged.has_full_text or secondary.has_full_text
    # Peer-reviewed status is sticky: a preprint duplicate never downgrades a
    # peer-reviewed record, and a peer-reviewed duplicate upgrades a preprint.
    merged.is_peer_reviewed = merged.is_peer_reviewed or secondary.is_peer_reviewed
    if secondary.is_peer_reviewed and merged.source_type == "preprint":
        merged.source_type = secondary.source_type
    merged.is_preprint = merged.source_type == "preprint"
    merged.is_background = merged.is_background and secondary.is_background
    merged.extra = {**secondary.extra, **merged.extra}
    merged.extra.setdefault("also_found_in", [])
    also = set(merged.extra.get("also_found_in", []))
    also.add(secondary.source_name)
    merged.extra["also_found_in"] = sorted(also)
    return merged


def deduplicate(documents: Sequence[Document]) -> Tuple[List[Document], DeduplicationReport]:
    """Remove duplicate publications, keeping the highest-quality record."""

    report = DeduplicationReport(input_count=len(documents))
    if not documents:
        report.output_count = 0
        return [], report

    # Sort by source priority then quality so the best record seeds each group.
    ordered = sorted(
        documents,
        key=lambda d: (
            SOURCE_PRIORITY.get(d.source_name, 9),
            -_quality_score(d)[0],
            -len(d.text_passage or ""),
        ),
    )

    groups: List[List[Document]] = []
    index: Dict[str, int] = {}

    for doc in ordered:
        doi = normalise_doi(doc.doi)
        pmid = str(doc.pmid).strip() if doc.pmid else None
        arxiv = doc.arxiv_id.strip().lower() if doc.arxiv_id else None

        candidates = []
        if doi and f"doi:{doi}" in index:
            candidates.append(index[f"doi:{doi}"])
        if pmid and f"pmid:{pmid}" in index:
            candidates.append(index[f"pmid:{pmid}"])
        if arxiv and f"arxiv:{arxiv}" in index:
            candidates.append(index[f"arxiv:{arxiv}"])

        if not candidates:
            norm_title = _norm_title(doc.title)
            if norm_title and f"title:{norm_title}" in index:
                candidates.append(index[f"title:{norm_title}"])
            else:
                for idx, group in enumerate(groups):
                    if any(titles_match(doc.title, member.title) for member in group):
                        candidates.append(idx)
                        break

        if candidates:
            group_index = candidates[0]
            groups[group_index].append(doc)
            report.merged_groups.append([d.document_id for d in groups[group_index]])
        else:
            groups.append([doc])
            group_index = len(groups) - 1

        if doi:
            index[f"doi:{doi}"] = group_index
        if pmid:
            index[f"pmid:{pmid}"] = group_index
        if arxiv:
            index[f"arxiv:{arxiv}"] = group_index
        norm_title = _norm_title(doc.title)
        if norm_title:
            index[f"title:{norm_title}"] = group_index

    unique: List[Document] = []
    for group in groups:
        base = group[0]
        for other in group[1:]:
            base = merge_documents(base, other)
        unique.append(base)

    report.output_count = len(unique)
    report.duplicates_removed = report.input_count - report.output_count
    if report.duplicates_removed:
        logger.info(
            "deduplication removed %d of %d records",
            report.duplicates_removed,
            report.input_count,
        )
    return unique, report
