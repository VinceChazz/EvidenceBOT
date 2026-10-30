"""Wikipedia retrieval for *background* information only.

Wikipedia is deliberately never treated as peer-reviewed evidence: every
document produced here has ``is_background=True`` and ``source_type="background"``,
and the evidence evaluator excludes such documents from sufficiency checks.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config import Settings
from processing.models import Document, stable_document_id
from retrieval.base import HttpClient
from utils.helpers import clean_text, truncate
from utils.logging_config import get_logger

logger = get_logger("retrieval.wikipedia")

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaRetriever:
    name = "Wikipedia"

    def __init__(self, settings: Settings, http: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http = http or HttpClient(settings)

    # ------------------------------------------------------------------
    def search(self, question: str, limit: Optional[int] = None) -> List[Document]:
        limit = min(limit or 2, 3)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": re.sub(r"[?]", " ", question or "").strip(),
            "srlimit": str(limit),
            "format": "json",
            "utf8": "1",
        }
        payload = self.http.get_json(SEARCH_URL, params=params)
        hits = ((payload or {}).get("query", {}) or {}).get("search", []) or []
        documents: List[Document] = []
        for hit in hits:
            title = clean_text(hit.get("title") or "")
            if not title:
                continue
            summary = self._summary(title)
            if not summary:
                summary = clean_text(hit.get("snippet") or "")
                summary = re.sub(r"<[^>]+>", "", summary)
            if not summary:
                continue
            documents.append(
                Document(
                    document_id=stable_document_id("Wikipedia", title, title),
                    title=f"{title} — Wikipedia (background)",
                    text_passage=truncate(clean_text(summary), 2500),
                    source_name="Wikipedia",
                    source_type="background",
                    publication_year=None,
                    authors=["Wikipedia contributors"],
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    journal="Wikipedia",
                    is_background=True,
                    is_peer_reviewed=False,
                    provenance="encyclopaedia_summary",
                    note=(
                        "Background material only. Wikipedia cannot satisfy the "
                        "scientific evidence requirement and is never cited as "
                        "evidence for a biomedical conclusion."
                    ),
                )
            )
        logger.info("Wikipedia returned %d background documents", len(documents))
        return documents

    def _summary(self, title: str) -> str:
        try:
            payload = self.http.get_json(SUMMARY_URL.format(title=title.replace(" ", "_")))
        except Exception as exc:  # noqa: BLE001
            logger.debug("wikipedia summary failed for %s: %s", title, exc)
            return ""
        if not isinstance(payload, dict):
            return ""
        return clean_text(payload.get("extract") or "")
