"""OpenAI chat-completion wrapper for grounded answer generation.

The client is injectable so the whole pipeline can be tested with mocked
responses and never depends on paid API calls in CI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import Settings
from utils.cache import TTLCache
from utils.helpers import extract_json_object, stable_hash
from utils.logging_config import get_logger, redact

logger = get_logger("generation.llm")


class LLMUnavailable(RuntimeError):
    """Raised when no LLM can be used (missing key, API error, invalid output)."""


@dataclass
class LLMResult:
    """Outcome of one generation call."""

    data: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.data)


class LLMClient:
    """Thin, retrying wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        settings: Settings,
        client: Optional[object] = None,
        cache: Optional[TTLCache] = None,
    ) -> None:
        self.settings = settings
        self.model = settings.openai_model
        self._client = client
        self._client_error: Optional[str] = None
        self.cache = cache or TTLCache(
            "llm", ttl_seconds=max(settings.cache_ttl_seconds, 3600), enabled=settings.cache_enabled
        )
        if self._client is None and settings.openai_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url or None,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=1,
                )
            except Exception as exc:  # noqa: BLE001
                self._client_error = redact(str(exc))
                logger.warning("OpenAI client could not be created: %s", self._client_error)

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        if self._client is not None:
            return None
        if self._client_error:
            return self._client_error
        return "OPENAI_API_KEY is not configured."

    # ------------------------------------------------------------------
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        use_cache: bool = True,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """Call the model and parse a JSON object from the response."""

        if not self.is_available:
            return LLMResult(error=self.unavailable_reason, model=self.model)

        cache_key = stable_hash(self.model, system_prompt, user_prompt)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return LLMResult(
                    data=cached.get("data", {}),
                    raw_text=cached.get("raw_text", ""),
                    model=self.model,
                    elapsed_seconds=0.0,
                )

        started = time.perf_counter()
        last_error: Optional[str] = None
        # Some models (reasoning models) reject ``temperature``/``max_tokens`` and need
        # ``max_completion_tokens`` instead, so the call parameters are adjusted on retry.
        use_max_completion_tokens = False
        drop_temperature = False
        for attempt in range(3):
            try:
                request: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "timeout": self.settings.llm_timeout_seconds,
                }
                if not drop_temperature:
                    request["temperature"] = (
                        self.settings.llm_temperature if temperature is None else temperature
                    )
                if use_max_completion_tokens:
                    request["max_completion_tokens"] = self.settings.llm_max_output_tokens
                else:
                    request["max_tokens"] = self.settings.llm_max_output_tokens
                response = self._client.chat.completions.create(**request)  # type: ignore[union-attr]
                raw = response.choices[0].message.content or ""
                data = extract_json_object(raw)
                if not data:
                    last_error = "Model returned no parsable JSON object."
                    logger.warning("generation attempt %d: %s", attempt + 1, last_error)
                    continue
                elapsed = time.perf_counter() - started
                usage = {}
                try:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                except Exception:  # pragma: no cover - usage optional
                    pass
                result = LLMResult(
                    data=data,
                    raw_text=raw,
                    model=self.model,
                    elapsed_seconds=elapsed,
                    usage=usage,
                )
                if use_cache:
                    self.cache.set(
                        cache_key,
                        {"data": data, "raw_text": raw},
                    )
                logger.info(
                    "generation ok in %.2fs (%s tokens)",
                    elapsed,
                    usage.get("total_tokens", "?"),
                )
                return result
            except Exception as exc:  # noqa: BLE001 - surface as a clean error
                last_error = redact(f"{type(exc).__name__}: {exc}")
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    use_max_completion_tokens = True
                elif "max_completion_tokens" in message and "not supported" in message:
                    use_max_completion_tokens = False
                if "temperature" in message:
                    drop_temperature = True
                logger.warning("generation attempt %d failed: %s", attempt + 1, last_error)
                time.sleep(0.5 * (attempt + 1))

        return LLMResult(
            error=last_error or "Unknown generation failure",
            model=self.model,
            elapsed_seconds=time.perf_counter() - started,
        )


def build_llm_client(settings: Settings, client: Optional[object] = None) -> LLMClient:
    return LLMClient(settings, client=client)


def generation_status(settings: Settings) -> Dict[str, Any]:
    return {
        "openai_key_configured": settings.has_openai_key,
        "model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
        "embedding_backend": settings.resolved_embedding_backend(),
        "base_url": settings.openai_base_url or "https://api.openai.com/v1",
    }
