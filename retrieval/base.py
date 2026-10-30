"""Shared HTTP plumbing for the retrieval sources.

Provides:
* a cached, retrying, rate-limited GET helper,
* uniform error and result objects,
* per-host rate limiting so PubMed / Europe PMC / arXiv / Crossref limits are respected.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

from config import Settings
from processing.models import Document
from utils.cache import TTLCache
from utils.logging_config import get_logger

logger = get_logger("retrieval.http")

USER_AGENT = (
    "EvidenceBot/1.0 (scientific evidence retrieval; "
    "https://github.com/VinceChazz/EvidenceBOT) Python-requests"
)

#: Minimum seconds between requests to the same host (NCBI allows 3/s w/o key).
_HOST_MIN_INTERVAL: Dict[str, float] = {
    "eutils.ncbi.nlm.nih.gov": 0.35,
    "www.ebi.ac.uk": 0.25,
    "export.arxiv.org": 3.0,
    "api.crossref.org": 0.2,
    "en.wikipedia.org": 0.2,
}

_RATE_LOCK = threading.Lock()
_LAST_REQUEST: Dict[str, float] = {}
#: Hosts that recently refused connections are short-circuited to keep the UI snappy.
_HOST_DOWN: Dict[str, float] = {}
HOST_COOLDOWN_SECONDS = 120.0


class RetrievalError(RuntimeError):
    """Raised when a retrieval source cannot return usable data."""


def short_error(exc: BaseException | str, limit: int = 180) -> str:
    """Compact error text for the UI: no URLs, no request dumps."""

    text = str(exc)
    text = re.sub(r"https?://\S+", "[url]", text)
    text = re.sub(r"\(Caused by [^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


@dataclass
class SourceResult:
    """Outcome of one retrieval source."""

    source: str
    documents: List[Document] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    query: str = ""
    note: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def count(self) -> int:
        return len(self.documents)

    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        if not self.documents:
            return "empty"
        return "ok"

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "documents": self.count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
            "note": self.note,
        }


class HttpClient:
    """Cached + rate-limited HTTP GET helper."""

    def __init__(self, settings: Settings, cache: Optional[TTLCache] = None) -> None:
        self.settings = settings
        self.timeout = max(1.0, settings.request_timeout_seconds)
        self.retries = max(0, settings.request_retries)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        self.cache = cache or TTLCache(
            "http",
            ttl_seconds=settings.cache_ttl_seconds,
            enabled=settings.cache_enabled,
        )

    # ------------------------------------------------------------------
    def _throttle(self, host: str) -> None:
        interval = _HOST_MIN_INTERVAL.get(host, 0.1)
        with _RATE_LOCK:
            last = _LAST_REQUEST.get(host, 0.0)
            wait = interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            _LAST_REQUEST[host] = time.time()

    @staticmethod
    def _check_host(host: str) -> None:
        with _RATE_LOCK:
            down_since = _HOST_DOWN.get(host)
        if down_since and time.time() - down_since < HOST_COOLDOWN_SECONDS:
            raise RetrievalError(f"{host} is unreachable (recent connection failure)")

    @staticmethod
    def _mark_host(host: str, down: bool) -> None:
        with _RATE_LOCK:
            if down:
                _HOST_DOWN[host] = time.time()
            else:
                _HOST_DOWN.pop(host, None)

    def get_text(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ) -> str:
        params = params or {}
        cache_key = "GET|" + url + "|" + repr(sorted(params.items())) + "|" + repr(headers)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("cache hit %s", url)
                return cached

        host = urlparse(url).netloc
        self._check_host(host)
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                self._throttle(host)
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    raise RetrievalError(
                        f"HTTP {response.status_code} from {host}"
                    )
                if response.status_code >= 400:
                    raise RetrievalError(
                        f"HTTP {response.status_code} from {host}: {response.text[:200]}"
                    )
                text = response.text
                self._mark_host(host, False)
                if use_cache:
                    self.cache.set(cache_key, text)
                return text
            except (requests.RequestException, RetrievalError) as exc:
                last_error = exc
                if attempt < self.retries:
                    backoff = min(4.0, 0.6 * (2 ** attempt))
                    logger.warning(
                        "request failed (%s/%s) %s: %s — retrying in %.1fs",
                        attempt + 1,
                        self.retries + 1,
                        host,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
        self._mark_host(host, True)
        raise RetrievalError(f"{host} unreachable: {short_error(last_error)}") from last_error

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ) -> Any:
        import json

        text = self.get_text(url, params=params, headers=headers, use_cache=use_cache)
        try:
            return json.loads(text)
        except ValueError as exc:
            raise RetrievalError(f"Invalid JSON from {urlparse(url).netloc}: {exc}") from exc


def run_source(
    name: str, func: Callable[[], List[Document]], query: str
) -> SourceResult:
    """Execute a retrieval function and capture failures without raising."""

    started = time.perf_counter()
    try:
        documents = func() or []
        elapsed = time.perf_counter() - started
        logger.info("%s returned %d documents in %.2fs", name, len(documents), elapsed)
        return SourceResult(
            source=name, documents=documents, elapsed_seconds=elapsed, query=query
        )
    except Exception as exc:  # noqa: BLE001 - one failing source must not break the run
        elapsed = time.perf_counter() - started
        message = short_error(f"{type(exc).__name__}: {exc}")
        logger.warning("%s failed after %.2fs: %s", name, elapsed, message)
        return SourceResult(
            source=name, documents=[], error=message, elapsed_seconds=elapsed, query=query
        )
