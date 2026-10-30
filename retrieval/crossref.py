"""Crossref REST API retrieval (metadata + coverage beyond PubMed)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import Settings
from processing.models import Document, stable_document_id
from retrieval.base import HttpClient, RetrievalError
from utils.helpers import clean_text, normalise_doi, safe_int, strip_html
from utils.logging_config import get_logger

logger = get_logger("retrieval.crossref")

BASE = "https://api.crossref.org/works"

#: Crossref ``type`` -> EvidenceBot source_type.
TYPE_MAP = {
    "journal-article": "peer_reviewed_article",
    "proceedings-article": "conference_paper",
    "posted-content": "preprint",
    "book-chapter": "book_chapter",
    "book": "book_chapter",
    "report": "peer_reviewed_article",
    "dataset": "unknown",
    "peer-review": "unknown",
}


def _title_from_item(item: Dict[str, Any]) -> str:
    titles = item.get("title") or []
    if isinstance(titles, list) and titles:
        return clean_text(strip_html(titles[0]))
    if isinstance(titles, str):
        return clean_text(strip_html(titles))
    return ""


def _abstract_from_item(item: Dict[str, Any]) -> str:
    abstract = item.get("abstract")
    if not abstract:
        return ""
    return clean_text(strip_html(abstract))


class CrossrefRetriever:
    name = "Crossref"

    def __init__(self, settings: Settings, http: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)

    # ------------------------------------------------------------------
    def search(self, question: str, limit: Optional[int] = None) -> List[Document]:
        limit = limit or self.settings.max_results_per_source
        cleaned = re.sub(r'[?*"()\[\]]', " ", question or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return []
        params = {
            "query.bibliographic": cleaned,
            "rows": str(limit),
            "select": "DOI,title,author,issued,container-title,publisher,type,abstract,URL,subject,is-referenced-by-count",
            "mailto": self.settings.crossref_mailto,
        }
        payload = self.http.get_json(BASE, params=params)
        if not isinstance(payload, dict):
            raise RetrievalError("Unexpected Crossref response")
        message = payload.get("message", {}) or {}
        items = message.get("items", []) or []
        documents = [doc for doc in (self._to_document(i) for i in items) if doc]
        logger.info("Crossref returned %d documents", len(documents))
        return documents[:limit]

    # ------------------------------------------------------------------
    def _to_document(self, item: Dict[str, Any]) -> Optional[Document]:
        title = _title_from_item(item)
        abstract = _abstract_from_item(item)
        if not title:
            return None
        doi = normalise_doi(item.get("DOI"))
        year = None
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        if issued and issued[0]:
            year = safe_int(issued[0][0])
        authors: List[str] = []
        for author in item.get("author", []) or []:
            given = clean_text(author.get("given") or "")
            family = clean_text(author.get("family") or "")
            name = f"{given} {family}".strip() or clean_text(author.get("name") or "")
            if name:
                authors.append(name)
        container = item.get("container-title") or []
        journal = clean_text(container[0]) if isinstance(container, list) and container else None
        crossref_type = str(item.get("type") or "").lower()
        source_type = TYPE_MAP.get(crossref_type, "peer_reviewed_article")
        is_preprint = source_type == "preprint"

        return Document(
            document_id=stable_document_id("Crossref", doi or title, title),
            title=title,
            # Crossref abstracts are frequently missing; the title still carries
            # topical signal, but we flag it so evidence grading can down-weight it.
            text_passage=abstract or title,
            source_name="Crossref",
            source_type=source_type,
            publication_year=year,
            authors=authors[:12],
            doi=doi,
            url=(item.get("URL") or (f"https://doi.org/{doi}" if doi else None)),
            journal=journal,
            publisher=clean_text(item.get("publisher") or "") or None,
            is_preprint=is_preprint,
            is_peer_reviewed=not is_preprint,
            has_full_text=False,
            provenance="abstract" if abstract else "title_only",
            note=None if abstract else "Metadata record without abstract.",
            extra={
                "crossref_type": crossref_type,
                "is_referenced_by_count": item.get("is-referenced-by-count"),
                "subjects": (item.get("subject") or [])[:6],
            },
        )
