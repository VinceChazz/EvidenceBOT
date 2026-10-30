"""Europe PMC REST API retrieval (expands biomedical coverage beyond PubMed)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import Settings
from processing.models import Document, stable_document_id
from retrieval.base import HttpClient, RetrievalError
from utils.helpers import clean_text, normalise_doi, safe_int, strip_html
from utils.logging_config import get_logger

logger = get_logger("retrieval.europe_pmc")

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARTICLE_URL = "https://europepmc.org/article/{source}/{id}"
OPEN_ACCESS_XML = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)


def _publication_type(record: Dict[str, Any]) -> str:
    """Map Europe PMC metadata to EvidenceBot's source taxonomy."""

    pub_types = record.get("pubTypeList", {}).get("pubType", []) or []
    if isinstance(pub_types, str):
        pub_types = [pub_types]
    joined = " ".join(str(p).lower() for p in pub_types)
    title = str(record.get("title", "")).lower()
    if "systematic review" in joined or "systematic review" in title:
        return "systematic_review"
    if "meta-analysis" in joined or "meta analysis" in title:
        return "meta_analysis"
    if "randomized controlled trial" in joined or "randomised controlled trial" in joined:
        return "randomized_controlled_trial"
    if "clinical trial" in joined:
        return "clinical_trial"
    if "cohort" in joined or "cohort" in title:
        return "cohort_study"
    if "review" in joined or "review" in title:
        return "review"
    if "preprint" in joined:
        return "preprint"
    return "peer_reviewed_article"


class EuropePMCRetriever:
    name = "Europe PMC"

    def __init__(self, settings: Settings, http: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)

    # ------------------------------------------------------------------
    def _query(self, question: str) -> str:
        # Europe PMC supports free text; restrict to life-science records.
        cleaned = re.sub(r'[?*"()\[\]]', " ", question or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def search_records(self, question: str, limit: int) -> List[Dict[str, Any]]:
        params = {
            "query": self._query(question),
            "format": "json",
            "pageSize": str(limit),
            "resultType": "core",
            "sort": "CITED desc",
        }
        payload = self.http.get_json(BASE, params=params)
        if not isinstance(payload, dict):
            raise RetrievalError("Unexpected Europe PMC response")
        result_list = payload.get("resultList", {}) or {}
        return list(result_list.get("result", []) or [])

    def fetch_full_text(self, pmcid: Optional[str]) -> Optional[str]:
        """Download open-access full text XML for a PMC record (best effort)."""

        if not pmcid:
            return None
        try:
            xml_text = self.http.get_text(OPEN_ACCESS_XML.format(pmcid=pmcid))
            from utils.helpers import strip_html as _strip

            text = _strip(xml_text)
            text = clean_text(text)
            if len(text) < 500:
                return None
            return text[: self.settings.max_fulltext_chars]
        except Exception as exc:  # noqa: BLE001 - full text is optional
            logger.debug("full text unavailable for %s: %s", pmcid, exc)
            return None

    def search(
        self, question: str, limit: Optional[int] = None, full_text_budget: Optional[int] = None
    ) -> List[Document]:
        limit = limit or self.settings.max_results_per_source
        records = self.search_records(question, limit)
        documents: List[Document] = []
        fulltext_used = 0
        budget = self.settings.max_fulltext_fetches if full_text_budget is None else full_text_budget

        for record in records:
            try:
                doc = self._to_document(record)
            except Exception as exc:  # noqa: BLE001
                logger.debug("skipping malformed Europe PMC record: %s", exc)
                continue
            if doc is None:
                continue
            pmcid = record.get("pmcid")
            if doc.extra.get("is_open_access") and fulltext_used < budget and pmcid:
                full_text = self.fetch_full_text(pmcid)
                if full_text:
                    doc.text_passage = f"{doc.text_passage}\n\n{full_text}".strip()
                    doc.has_full_text = True
                    doc.provenance = "abstract+full_text"
                    fulltext_used += 1
            documents.append(doc)
        logger.info(
            "Europe PMC returned %d documents (%d with full text)", len(documents), fulltext_used
        )
        return documents[:limit]

    # ------------------------------------------------------------------
    def _to_document(self, record: Dict[str, Any]) -> Optional[Document]:
        title = clean_text(record.get("title") or "")
        abstract = clean_text(
            strip_html(record.get("abstractText") or "") or record.get("abstractText") or ""
        )
        if not title and not abstract:
            return None
        pmid = str(record.get("pmid")) if record.get("pmid") else None
        doi = normalise_doi(record.get("doi"))
        source = record.get("source") or "MED"
        record_id = record.get("id") or pmid or doi or title
        year = safe_int(record.get("pubYear") or record.get("firstPublicationDate"))

        author_string = record.get("authorString") or ""
        authors = [
            clean_text(a).strip(",.")
            for a in re.split(r",", author_string)
            if clean_text(a).strip(",.")
        ][:12]

        pmcid = record.get("pmcid")
        is_open_access = str(record.get("isOpenAccess", "")).upper() == "Y"
        url = ARTICLE_URL.format(source=source, id=record_id)
        if doi and not pmid:
            url = f"https://doi.org/{doi}"

        pub_type = _publication_type(record)
        is_preprint = str(record.get("source", "")).upper() == "PPR" or pub_type == "preprint"
        journal = (record.get("journalInfo", {}) or {}).get("journal", {}) or {}
        journal_title = clean_text(journal.get("title") or "")

        return Document(
            document_id=stable_document_id("Europe PMC", str(record_id), title),
            title=title or abstract[:160],
            text_passage=abstract or title,
            source_name="Europe PMC",
            source_type=pub_type,
            publication_year=year,
            authors=authors,
            doi=doi,
            pmid=pmid,
            url=url,
            journal=journal_title or None,
            publisher=None,
            is_preprint=is_preprint,
            is_peer_reviewed=not is_preprint,
            has_full_text=False,
            provenance="abstract",
            extra={
                "pmcid": pmcid,
                "is_open_access": is_open_access,
                "europe_pmc_source": source,
                "cited_by_count": record.get("citedByCount"),
            },
        )
