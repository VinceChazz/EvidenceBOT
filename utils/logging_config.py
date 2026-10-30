"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Iterable

_CONFIGURED = False

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s]+)"),
]


class RedactionFilter(logging.Filter):
    """Remove API keys from anything that reaches a log handler."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = message
        for pattern in _SECRET_PATTERNS:
            if pattern.groups >= 2 and pattern.pattern.lower().startswith("(?i)(api"):
                redacted = pattern.sub(r"\1***", redacted)
            else:
                redacted = pattern.sub("***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO", stream=None) -> logging.Logger:
    """Configure the root EvidenceBot logger once."""

    global _CONFIGURED
    logger = logging.getLogger("evidencebot")
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        handler.addFilter(RedactionFilter())
        logger.addHandler(handler)
        logger.setLevel(numeric)
        logger.propagate = False
        _CONFIGURED = True
    else:
        logger.setLevel(numeric)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the EvidenceBot logger."""

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(f"evidencebot.{name}")


def redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("***", redacted)
    return redacted


def log_source_errors(logger: logging.Logger, errors: Iterable) -> None:
    for err in errors:
        logger.warning("source failure: %s", redact(str(err)))
