"""Abstention policy helpers.

The abstention decision is produced by the evidence evaluator and is binding:
the generation layer receives it as an input and can never overturn it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import INSUFFICIENT_EVIDENCE_MESSAGE

ANSWERABLE = "ANSWERABLE"
PARTIALLY_ANSWERABLE = "PARTIALLY_ANSWERABLE"
UNANSWERABLE = "UNANSWERABLE"


@dataclass
class AbstentionDecision:
    """The final, immutable outcome of the evidence stage."""

    decision: str
    reason_code: str
    reason: str
    message: Optional[str] = None
    criteria_failed: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.criteria_failed is None:
            self.criteria_failed = []

    @property
    def is_abstention(self) -> bool:
        return self.decision == UNANSWERABLE

    @property
    def allows_generation(self) -> bool:
        return self.decision in {ANSWERABLE, PARTIALLY_ANSWERABLE}

    @property
    def is_partial(self) -> bool:
        return self.decision == PARTIALLY_ANSWERABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "message": self.message,
            "criteria_failed": list(self.criteria_failed),
        }


def abstain(reason_code: str, reason: str, criteria_failed: Optional[List[str]] = None) -> AbstentionDecision:
    """Create an abstention carrying the exact required user-facing message."""

    return AbstentionDecision(
        decision=UNANSWERABLE,
        reason_code=reason_code,
        reason=reason,
        message=INSUFFICIENT_EVIDENCE_MESSAGE,
        criteria_failed=criteria_failed or [],
    )


def should_generate(decision: AbstentionDecision) -> bool:
    """Single source of truth used by the pipeline before any LLM call."""

    return decision.allows_generation


def insufficient_evidence_message() -> str:
    return INSUFFICIENT_EVIDENCE_MESSAGE
