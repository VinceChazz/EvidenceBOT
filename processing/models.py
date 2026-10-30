"""Unified document / passage data structures used across EvidenceBot.

Every retrieval source is normalised into :class:`Document` objects, which are
then split into :class:`Passage` chunks. Every chunk keeps the identifier and
metadata of the document it came from, so a citation can always be traced back
to a real retrieved record.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Source taxonomy
# ---------------------------------------------------------------------------

#: Source types ordered from strongest to weakest for evidence purposes.
SOURCE_TYPE_WEIGHTS: Dict[str, float] = {
    "systematic_review": 1.00,
    "meta_analysis": 1.00,
    "randomized_controlled_trial": 1.00,
    "clinical_trial": 0.90,
    "cohort_study": 0.80,
    "case_control_study": 0.75,
    "observational_study": 0.70,
    "peer_reviewed_article": 0.85,
    "review": 0.80,
    "conference_paper": 0.55,
    "preprint": 0.45,
    "book_chapter": 0.50,
    "background": 0.15,  # encyclopaedic background, never sufficient on its own
    "unknown": 0.40,
}

#: Types that are accepted as scientific evidence for sufficiency purposes.
EVIDENCE_GRADE_SOURCE_TYPES = {
    "systematic_review",
    "meta_analysis",
    "randomized_controlled_trial",
    "clinical_trial",
    "cohort_study",
    "case_control_study",
    "observational_study",
    "peer_reviewed_article",
    "review",
    "conference_paper",
    "preprint",
    "book_chapter",
    "unknown",
}

REVIEW_LIKE_TYPES = {"systematic_review", "meta_analysis", "review"}


def _clean_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


@dataclass
class Document:
    """A single standardised publication record."""

    document_id: str
    title: str
    text_passage: str
    source_name: str
    source_type: str
    publication_year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    is_preprint: bool = False
    is_background: bool = False
    is_peer_reviewed: bool = True
    has_full_text: bool = False
    #: Free-form provenance (e.g. "abstract", "full_text", "offline_sample").
    provenance: str = "abstract"
    #: True only for the synthetic offline sample corpus.
    is_sample: bool = False
    note: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def source_type_weight(self) -> float:
        return SOURCE_TYPE_WEIGHTS.get(self.source_type, SOURCE_TYPE_WEIGHTS["unknown"])

    @property
    def is_review_like(self) -> bool:
        return self.source_type in REVIEW_LIKE_TYPES

    @property
    def is_scientific_evidence(self) -> bool:
        """Background material (e.g. Wikipedia) can never establish a conclusion."""

        return (not self.is_background) and self.source_type in EVIDENCE_GRADE_SOURCE_TYPES

    @property
    def display_source(self) -> str:
        return self.source_name

    @property
    def citation_label(self) -> str:
        authors = ", ".join(self.authors[:3]) if self.authors else "Unknown authors"
        if len(self.authors) > 3:
            authors += " et al."
        year = self.publication_year or "n.d."
        return f"{authors} ({year})"

    def identity_keys(self) -> List[str]:
        keys: List[str] = []
        if self.doi:
            keys.append("doi:" + self.doi.strip().lower())
        if self.pmid:
            keys.append("pmid:" + str(self.pmid).strip())
        if self.arxiv_id:
            keys.append("arxiv:" + str(self.arxiv_id).strip().lower())
        if self.title:
            keys.append("title:" + re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip())
        return keys

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_record(self) -> Dict[str, Any]:
        """Flattened representation used in the Pandas DataFrame."""

        data = asdict(self)
        data["authors"] = "; ".join(self.authors)
        return data


@dataclass
class Passage:
    """A chunk of a document, carrying its parent document metadata."""

    passage_id: str
    document_id: str
    text: str
    source_name: str
    source_type: str
    chunk_index: int = 0
    title: Optional[str] = None
    publication_year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    is_preprint: bool = False
    is_background: bool = False
    is_peer_reviewed: bool = True
    is_sample: bool = False
    provenance: str = "abstract"
    #: Scores are attached by the ranking stage.
    dense_score: float = 0.0
    bm25_score: float = 0.0
    lexical_score: float = 0.0
    relevance_score: float = 0.0
    #: Stance detected by the evidence evaluator ("support", "null", "oppose", "unclear").
    stance: str = "unclear"

    @classmethod
    def from_document(
        cls, document: Document, text: str, chunk_index: int = 0
    ) -> "Passage":
        passage_id = f"{document.document_id}::c{chunk_index}"
        return cls(
            passage_id=passage_id,
            document_id=document.document_id,
            text=text,
            source_name=document.source_name,
            source_type=document.source_type,
            chunk_index=chunk_index,
            title=document.title,
            publication_year=document.publication_year,
            authors=list(document.authors),
            doi=document.doi,
            pmid=document.pmid,
            arxiv_id=document.arxiv_id,
            url=document.url,
            journal=document.journal,
            is_preprint=document.is_preprint,
            is_background=document.is_background,
            is_peer_reviewed=document.is_peer_reviewed,
            is_sample=document.is_sample,
            provenance=document.provenance,
        )

    @property
    def source_type_weight(self) -> float:
        return SOURCE_TYPE_WEIGHTS.get(self.source_type, SOURCE_TYPE_WEIGHTS["unknown"])

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["authors"] = "; ".join(self.authors)
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stable_document_id(source_name: str, identifier: str, title: str) -> str:
    """Deterministic id so the same paper gets the same id across runs."""

    basis = f"{source_name}|{identifier}|{title}".lower()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def documents_to_payload(documents: List[Document]) -> str:
    return json.dumps([d.to_dict() for d in documents], ensure_ascii=False)
