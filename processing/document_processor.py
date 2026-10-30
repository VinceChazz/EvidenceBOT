"""Document standardisation: cleaning, DataFrame assembly, dedup and chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from config import Settings
from processing.chunking import chunk_documents
from processing.deduplication import DeduplicationReport, deduplicate
from processing.models import Document, Passage
from utils.helpers import clean_text, normalise_doi
from utils.logging_config import get_logger

logger = get_logger("processing.documents")

DATA_COLUMNS = [
    "document_id",
    "title",
    "text_passage",
    "source_name",
    "source_type",
    "publication_year",
    "authors",
    "doi",
    "pmid",
    "arxiv_id",
    "url",
    "journal",
    "publisher",
    "is_preprint",
    "is_background",
    "is_peer_reviewed",
    "has_full_text",
    "provenance",
    "is_sample",
    "note",
]


@dataclass
class ProcessedCorpus:
    """Standardised corpus ready for ranking."""

    documents: List[Document]
    passages: List[Passage]
    frame: pd.DataFrame
    passage_frame: pd.DataFrame
    dedup_report: DeduplicationReport = field(default_factory=DeduplicationReport)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def passage_count(self) -> int:
        return len(self.passages)

    def passages_as_frame(self) -> pd.DataFrame:
        return self.passage_frame


def clean_document(document: Document) -> Document:
    """Normalise text and metadata of a single document."""

    document.title = clean_text(document.title)
    document.text_passage = clean_text(document.text_passage)
    document.doi = normalise_doi(document.doi)
    document.pmid = str(document.pmid).strip() if document.pmid else None
    if document.pmid:
        document.pmid = "".join(ch for ch in document.pmid if ch.isdigit()) or None
    document.authors = [
        clean_text(a) for a in (document.authors or []) if a and clean_text(a)
    ][:12]
    if document.url:
        document.url = document.url.strip()
    # Re-derive the boolean flags so downstream logic can trust them.
    document.is_preprint = document.is_preprint or document.source_type == "preprint"
    document.is_background = document.is_background or document.source_type == "background"
    document.is_peer_reviewed = (
        not document.is_preprint and not document.is_background and document.is_peer_reviewed
    )
    if document.text_passage and len(document.text_passage) < 25 and document.title:
        document.text_passage = document.title
    return document


def documents_to_frame(documents: Sequence[Document]) -> pd.DataFrame:
    """Build the unified Pandas DataFrame of documents."""

    if not documents:
        return pd.DataFrame(columns=DATA_COLUMNS)
    rows = []
    for doc in documents:
        row = doc.to_record()
        rows.append({column: row.get(column) for column in DATA_COLUMNS})
    frame = pd.DataFrame(rows, columns=DATA_COLUMNS)
    frame["publication_year"] = pd.to_numeric(frame["publication_year"], errors="coerce")
    return frame


def passages_to_frame(passages: Sequence[Passage]) -> pd.DataFrame:
    if not passages:
        return pd.DataFrame()
    frame = pd.DataFrame([p.to_dict() for p in passages])
    return frame


def process_documents(
    documents: Iterable[Document],
    settings: Settings,
    deduplicate_documents: bool = True,
) -> ProcessedCorpus:
    """Clean, deduplicate, store in a DataFrame and chunk the corpus."""

    cleaned = [clean_document(doc) for doc in documents]
    cleaned = [doc for doc in cleaned if doc.title or doc.text_passage]
    report = DeduplicationReport(input_count=len(cleaned), output_count=len(cleaned))
    if deduplicate_documents:
        cleaned, report = deduplicate(cleaned)

    passages = chunk_documents(
        cleaned,
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
        min_chunk_words=settings.min_chunk_words,
    )
    corpus = ProcessedCorpus(
        documents=cleaned,
        passages=passages,
        frame=documents_to_frame(cleaned),
        passage_frame=passages_to_frame(passages),
        dedup_report=report,
    )
    logger.info(
        "processed corpus: %d documents, %d passages (dupes removed: %d)",
        corpus.document_count,
        corpus.passage_count,
        report.duplicates_removed,
    )
    return corpus


def source_breakdown(documents: Sequence[Document]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for doc in documents:
        counts[doc.source_name] = counts.get(doc.source_name, 0) + 1
    return counts
