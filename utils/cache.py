"""Small TTL cache with optional on-disk persistence.

Used for external API responses and embeddings so that repeated questions do
not hammer PubMed / Europe PMC / arXiv / Crossref (and do not cost money on the
embedding API).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from config import CACHE_DIR
from utils.helpers import stable_hash
from utils.logging_config import get_logger

logger = get_logger("cache")


class TTLCache:
    """Thread-safe in-memory cache with optional JSON persistence."""

    def __init__(
        self,
        namespace: str,
        ttl_seconds: int = 3600,
        enabled: bool = True,
        persist: bool = True,
        directory: Optional[str] = None,
    ) -> None:
        self.namespace = namespace
        self.ttl = max(0, int(ttl_seconds))
        self.enabled = enabled
        self.persist = persist
        self.directory = directory or os.path.join(CACHE_DIR, namespace)
        self._memory: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        if self.enabled and self.persist:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except OSError:  # pragma: no cover - read-only filesystem
                self.persist = False

    # ------------------------------------------------------------------
    def _path(self, key: str) -> str:
        return os.path.join(self.directory, stable_hash(key) + ".json")

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            hit = self._memory.get(key)
            if hit and (self.ttl <= 0 or now - hit[0] < self.ttl):
                return hit[1]
            if hit:
                self._memory.pop(key, None)
        if not self.persist:
            return None
        path = self._path(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if self.ttl > 0 and now - float(payload.get("stored_at", 0)) >= self.ttl:
                return None
            value = payload.get("value")
        except (OSError, ValueError):
            return None
        with self._lock:
            self._memory[key] = (now, value)
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            self._memory[key] = (now, value)
        if not self.persist:
            return
        try:
            with open(self._path(key), "w", encoding="utf-8") as fh:
                json.dump({"stored_at": now, "value": value}, fh)
        except (OSError, TypeError, ValueError):
            pass

    def get_or_compute(self, key: str, factory: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        if value is not None:
            self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
        if not self.persist:
            return
        try:
            for name in os.listdir(self.directory):
                if name.endswith(".json"):
                    os.remove(os.path.join(self.directory, name))
        except OSError:  # pragma: no cover
            pass


def clear_all_caches() -> None:
    for namespace in ("http", "embeddings", "llm", "demo"):
        TTLCache(namespace).clear()
    logger.info("all caches cleared")
