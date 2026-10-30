"""PubMed / NCBI E-utilities retrieval.

Biopython is used for the record model: abstracts are fetched in MEDLINE format
and parsed with :mod:`Bio.Medline`, which needs no external DTD. (Biopython's
``Bio.Entrez.read`` XML parser requires NCBI's DTD, which it downloads at parse
time; that fails behind proxies, so XML output is handled by an ElementTree
fallback in this module.) ``Bio.Entrez`` is still configured with the tool,
e-mail and optional API key so any direct Entrez usage is compliant.

HTTP itself goes through :class:`retrieval.base.HttpClient`, which provides
timeouts, retries, per-host rate limiting and response caching.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from Bio import Entrez, Medline

from config import Settings
from processing.models import Document, stable_document_id
from retrieval.base import HttpClient, RetrievalError
from utils.helpers import clean_text, normalise_doi, question_concepts, safe_int
from utils.logging_config import get_logger

logger = get_logger("retrieval.pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

#: PubMed publication type -> EvidenceBot source taxonomy.
PUBLICATION_TYPE_MAP = {
    "systematic review": "systematic_review",
    "meta-analysis": "meta_analysis",
    "meta analysis": "meta_analysis",
    "randomized controlled trial": "randomized_controlled_trial",
    "randomised controlled trial": "randomized_controlled_trial",
    "clinical trial": "clinical_trial",
    "clinical trial, phase ii": "clinical_trial",
    "clinical trial, phase iii": "clinical_trial",
    "cohort study": "cohort_study",
    "observational study": "observational_study",
    "case reports": "observational_study",
    "review": "review",
    "journal article": "peer_reviewed_article",
    "guideline": "review",
    "practice guideline": "review",
}


#: Most specific publication type wins ("Review" outranks "Journal Article").
PUBLICATION_TYPE_PRIORITY = [
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "randomized controlled trial",
    "randomised controlled trial",
    "clinical trial, phase iii",
    "clinical trial, phase ii",
    "clinical trial",
    "cohort study",
    "observational study",
    "case reports",
    "practice guideline",
    "guideline",
    "review",
    "journal article",
]


def map_publication_types(publication_types: List[str]) -> str:
    """Map a PubMed ``PT`` list onto the most specific source type."""

    lowered = {pt.strip().lower() for pt in publication_types if pt}
    for pub_type in PUBLICATION_TYPE_PRIORITY:
        if pub_type in lowered:
            return PUBLICATION_TYPE_MAP[pub_type]
    if any("review" in pt for pt in lowered):
        return "review"
    if any("trial" in pt for pt in lowered):
        return "clinical_trial"
    return "peer_reviewed_article"


def build_pubmed_queries(question: str) -> List[str]:
    """Fallback ladder of PubMed queries for a natural-language question."""

    cleaned = re.sub(r"[?*\"()\[\]]", " ", question or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    base = cleaned
    concepts = question_concepts(question)
    broad = " OR ".join(concepts[:6]) if concepts else base
    queries = [base]
    if broad and broad.lower() != base.lower():
        queries.append(broad)
    if concepts:
        queries.append(" AND ".join(concepts[:3]))
    seen, out = set(), []
    for query in queries:
        if query and query.lower() not in seen:
            seen.add(query.lower())
            out.append(query)
    return out


class PubMedRetriever:
    """Search PubMed via E-utilities and normalise the results."""

    name = "PubMed"

    def __init__(self, settings: Settings, http: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)
        try:
            Entrez.email = settings.ncbi_email
            Entrez.tool = settings.ncbi_tool
            if settings.ncbi_api_key:
                Entrez.api_key = settings.ncbi_api_key
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    def _common_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "db": "pubmed",
            "tool": self.settings.ncbi_tool,
            "email": self.settings.ncbi_email,
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def search_ids(self, query: str, limit: int) -> List[str]:
        params = self._common_params()
        params.update(
            {"term": query, "retmax": str(limit), "retmode": "json", "sort": "relevance"}
        )
        payload = self.http.get_json(f"{EUTILS}/esearch.fcgi", params=params)
        try:
            ids = payload["esearchresult"]["idlist"]
        except (KeyError, TypeError) as exc:
            raise RetrievalError("unexpected PubMed esearch response") from exc
        return [str(i) for i in ids][:limit]

    # ------------------------------------------------------------------
    def fetch_medline(self, ids: List[str]) -> List[Document]:
        """Fetch records in MEDLINE format and parse with Biopython."""

        params = self._common_params()
        params.update({"id": ",".join(ids), "rettype": "medline", "retmode": "text"})
        text = self.http.get_text(f"{EUTILS}/efetch.fcgi", params=params)
        documents: List[Document] = []
        handle = StringIO(text)
        try:
            for record in Medline.parse(handle):
                document = self._document_from_medline(record)
                if document:
                    documents.append(document)
        except Exception as exc:  # noqa: BLE001 - fall back to XML parsing
            logger.debug("MEDLINE parse failed (%s); trying XML", exc)
            return []
        return documents

    def fetch_xml(self, ids: List[str]) -> List[Document]:
        """Fetch PubMed XML and parse it with ElementTree (no DTD required)."""

        params = self._common_params()
        params.update({"id": ",".join(ids), "retmode": "xml"})
        xml_text = self.http.get_text(f"{EUTILS}/efetch.fcgi", params=params)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise RetrievalError(f"invalid PubMed XML: {exc}") from exc
        documents = []
        for node in root.findall(".//PubmedArticle"):
            document = self._document_from_xml(node)
            if document:
                documents.append(document)
        return documents

    # ------------------------------------------------------------------
    def search(self, question: str, limit: Optional[int] = None) -> List[Document]:
        limit = limit or self.settings.max_results_per_source
        last_error: Optional[Exception] = None
        for query in build_pubmed_queries(question):
            try:
                ids = self.search_ids(query, limit)
            except RetrievalError as exc:
                raise RetrievalError(f"PubMed esearch failed: {exc}") from exc
            if not ids:
                continue
            try:
                documents = self.fetch_medline(ids)
                if not documents:
                    documents = self.fetch_xml(ids)
            except RetrievalError as exc:
                last_error = exc
                logger.warning("PubMed fetch failed: %s", exc)
                continue
            if documents:
                logger.info("PubMed query %r -> %d documents", query, len(documents))
                return documents[:limit]
        if last_error is not None:
            raise RetrievalError(f"PubMed search failed: {last_error}")
        return []

    # ------------------------------------------------------------------
    def _document_from_medline(self, record: Dict[str, Any]) -> Optional[Document]:
        try:
            title = clean_text(record.get("TI") or "")
            abstract = clean_text(record.get("AB") or "")
            if not title and not abstract:
                return None
            pmid = re.sub(r"\D", "", str(record.get("PMID") or "")) or None
            doi = None
            for aid in record.get("AID") or []:
                match = re.match(r"(10\.[^\s]+)\s*\[doi\]", str(aid), re.IGNORECASE)
                if match:
                    doi = normalise_doi(match.group(1))
                    break
            if not doi:
                for lid in record.get("LID") or []:
                    match = re.match(r"(10\.[^\s]+)\s*\[doi\]", str(lid), re.IGNORECASE)
                    if match:
                        doi = normalise_doi(match.group(1))
                        break
            authors = []
            for author in record.get("AU") or []:
                name = clean_text(str(author))
                if not name:
                    continue
                if "," in name:  # "Hanow, Ilse"
                    last, fore = [part.strip() for part in name.split(",", 1)]
                    name = f"{fore} {last}".strip()
                authors.append(name)
            publication_types = record.get("PT") or []
            if isinstance(publication_types, str):
                publication_types = [publication_types]
            year = safe_int(record.get("DP"))
            return self._build(
                title=title,
                abstract=abstract,
                pmid=pmid,
                doi=doi,
                authors=authors,
                journal=clean_text(record.get("JT") or ""),
                year=year,
                publication_types=[str(pt) for pt in publication_types],
            )
        except Exception as exc:  # noqa: BLE001 - skip malformed records
            logger.debug("failed to normalise MEDLINE record: %s", exc)
            return None

    def _document_from_xml(self, node: ET.Element) -> Optional[Document]:
        try:
            title = clean_text("".join(node.findtext(".//Article/ArticleTitle") or ""))
            abstract = clean_text(
                " ".join(
                    (text.text or "")
                    for text in node.findall(".//Article/Abstract/AbstractText")
                )
            )
            if not title and not abstract:
                return None
            pmid = clean_text(node.findtext(".//MedlineCitation/PMID") or "")
            pmid = re.sub(r"\D", "", pmid) or None
            doi = None
            for article_id in node.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if article_id.get("IdType") == "doi" and article_id.text:
                    doi = normalise_doi(article_id.text)
                    break
            authors = []
            for author in node.findall(".//Article/AuthorList/Author"):
                last = clean_text(author.findtext("LastName") or "")
                fore = clean_text(author.findtext("ForeName") or author.findtext("Initials") or "")
                collective = clean_text(author.findtext("CollectiveName") or "")
                name = f"{fore} {last}".strip() or collective
                if name:
                    authors.append(name)
            publication_types = [
                clean_text(pt.text or "")
                for pt in node.findall(".//Article/PublicationTypeList/PublicationType")
            ]
            year = safe_int(
                node.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
                or node.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
                or node.findtext(".//Article/ArticleDate/Year")
            )
            return self._build(
                title=title,
                abstract=abstract,
                pmid=pmid,
                doi=doi,
                authors=authors,
                journal=clean_text(node.findtext(".//Article/Journal/Title") or ""),
                year=year,
                publication_types=publication_types,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("failed to normalise PubMed XML record: %s", exc)
            return None

    def _build(
        self,
        title: str,
        abstract: str,
        pmid: Optional[str],
        doi: Optional[str],
        authors: List[str],
        journal: str,
        year: Optional[int],
        publication_types: List[str],
    ) -> Document:
        source_type = map_publication_types(publication_types)
        url = (
            PUBMED_URL.format(pmid=pmid)
            if pmid
            else (f"https://doi.org/{doi}" if doi else None)
        )
        return Document(
            document_id=stable_document_id("PubMed", pmid or doi or title, title),
            title=title or abstract[:160],
            text_passage=abstract or title,
            source_name="PubMed",
            source_type=source_type,
            publication_year=year,
            authors=authors[:12],
            doi=doi,
            pmid=pmid,
            url=url,
            journal=journal or None,
            is_peer_reviewed=True,
            has_full_text=False,
            provenance="abstract",
            extra={"publication_types": publication_types},
        )
