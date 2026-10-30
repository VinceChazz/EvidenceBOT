"""arXiv API retrieval.

Only quantitative-biology / bioinformatics / ML / biomedical-AI material is
relevant to EvidenceBot, so result categories are filtered and preprints are
explicitly labelled as preprints (never as peer-reviewed literature).
"""

from __future__ import annotations

import re
from typing import List, Optional
from xml.etree import ElementTree as ET

from config import Settings
from processing.models import Document, stable_document_id
from retrieval.base import HttpClient
from utils.helpers import clean_text, safe_int
from utils.logging_config import get_logger

logger = get_logger("retrieval.arxiv")

BASE = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

#: Categories that EvidenceBot considers in scope.
RELEVANT_CATEGORIES = {
    "q-bio", "q-bio.QM", "q-bio.BM", "q-bio.GN", "q-bio.MN", "q-bio.TO",
    "q-bio.CB", "q-bio.NC", "cs.LG", "cs.AI", "cs.CL", "stat.ML", "stat.AP",
    "physics.med-ph", "eess.IV", "cs.CV",
}


class ArxivRetriever:
    name = "arXiv"

    def __init__(self, settings: Settings, http: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)

    # ------------------------------------------------------------------
    def _query(self, question: str) -> str:
        cleaned = re.sub(r'[?*"()\[\]:]', " ", question or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # arXiv's search wants field prefixes; all: searches every field.
        return f"all:{cleaned}" if cleaned else ""

    def search(
        self, question: str, limit: Optional[int] = None, restrict_categories: bool = True
    ) -> List[Document]:
        limit = limit or self.settings.max_results_per_source
        params = {
            "search_query": self._query(question),
            "start": "0",
            # Fetch extra because out-of-scope categories are filtered out.
            "max_results": str(min(50, limit * 3)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        xml_text = self.http.get_text(BASE, params=params)
        documents = self._parse(xml_text, restrict_categories=restrict_categories, limit=limit)
        logger.info("arXiv returned %d in-scope documents", len(documents))
        return documents

    # ------------------------------------------------------------------
    def _parse(self, xml_text: str, restrict_categories: bool, limit: int) -> List[Document]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"invalid arXiv XML: {exc}") from exc

        documents: List[Document] = []
        for entry in root.findall(f"{ATOM}entry"):
            title = clean_text(entry.findtext(f"{ATOM}title") or "")
            abstract = clean_text(entry.findtext(f"{ATOM}summary") or "")
            if not title and not abstract:
                continue
            raw_id = entry.findtext(f"{ATOM}id") or ""
            arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1]) if raw_id else None
            categories = [
                cat.attrib.get("term", "")
                for cat in entry.findall(f"{ATOM}category")
            ]
            primary = entry.find(f"{ARXIV_NS}primary_category")
            if primary is not None:
                categories.insert(0, primary.attrib.get("term", ""))
            if restrict_categories and categories:
                in_scope = any(
                    cat in RELEVANT_CATEGORIES
                    or cat.split(".")[0] in RELEVANT_CATEGORIES
                    for cat in categories
                )
                if not in_scope:
                    continue
            authors = [
                clean_text(a.findtext(f"{ATOM}name") or "")
                for a in entry.findall(f"{ATOM}author")
            ]
            published = entry.findtext(f"{ATOM}published") or entry.findtext(
                f"{ATOM}updated"
            )
            year = safe_int(published)
            doi = None
            doi_node = entry.find(f"{ARXIV_NS}doi")
            if doi_node is not None and doi_node.text:
                doi = doi_node.text.strip().lower()
            documents.append(
                Document(
                    document_id=stable_document_id("arXiv", arxiv_id or title, title),
                    title=title or abstract[:160],
                    text_passage=abstract or title,
                    source_name="arXiv",
                    source_type="preprint",
                    publication_year=year,
                    authors=[a for a in authors if a][:12],
                    doi=doi,
                    pmid=None,
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                    journal="arXiv (preprint server)",
                    is_preprint=True,
                    is_peer_reviewed=False,
                    provenance="preprint_abstract",
                    note="Preprint — not peer reviewed.",
                    extra={"categories": categories[:6]},
                )
            )
            if len(documents) >= limit:
                break
        return documents
