"""Minimal client for local or hosted OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError
from src.evaluations.evaluation_config import GenerationConfig, ServerConfig

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_CODES = {str(code) for code in _RETRYABLE_STATUS_CODES} | {
    "provider_unavailable",
    "rate_limit_exceeded",
}


def _is_retryable_error(value: Any) -> bool:
    """Recognize explicit transient errors, including nested provider metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"code", "status", "status_code", "type", "reason", "error_type"}:
                if isinstance(item, (str, int)) and str(item) in _RETRYABLE_ERROR_CODES:
                    return True
            if _is_retryable_error(item):
                return True
    elif isinstance(value, list):
        return any(_is_retryable_error(item) for item in value)
    return False


class _EndpointError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class CompletionResult:
    text: str
    response: dict[str, Any]
    reasoning_content: str | None = None


class _ResponseMessage(BaseModel):
    reasoning: str | None = None
    reasoning_content: str | None = None
    # Reasoning models can omit a final response while they are still producing
    # hidden reasoning, in which case OpenAI-compatible servers return null.
    content: str | None = None


class _ResponseChoice(BaseModel):
    message: _ResponseMessage


class _ChatCompletionResponse(BaseModel):
    choices: list[_ResponseChoice]
    usage: dict[str, Any] | None = None


class VLLMClient:
    def __init__(
        self,
        server: ServerConfig,
        *,
        model: str,
        generation: GenerationConfig,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.endpoint = f"{server.base_url.rstrip('/')}/chat/completions"
        self.api_key = server.api_key
        self.timeout_seconds = server.timeout_seconds
        self.model = model
        self.generation = generation
        self.max_retries = max_retries

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
        request_overrides: dict[str, Any] | None = None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.generation.temperature is not None:
            payload["temperature"] = self.generation.temperature
        if self.generation.top_p is not None:
            payload["top_p"] = self.generation.top_p
        payload[self.generation.max_tokens_field] = self.generation.max_tokens
        if self.generation.seed is not None:
            payload["seed"] = self.generation.seed
        if self.generation.structured_output and response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "arc_high_level_actions",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        payload.update(self.generation.request_overrides)
        if request_overrides is not None:
            payload.update(request_overrides)

        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        for attempt in range(self.max_retries + 1):
            try:
                return self._complete_request(request)
            except _EndpointError as error:
                if not error.retryable or attempt == self.max_retries:
                    raise
                delay = min(2 ** (attempt + 1), 30)
                logger.warning(
                    "Endpoint failure; retry %s/%s in %ss with the same request: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    error,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _complete_request(self, request: Request) -> CompletionResult:
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise _EndpointError(
                f"OpenAI-compatible endpoint returned HTTP {error.code}: {body}",
                retryable=error.code in _RETRYABLE_STATUS_CODES,
            ) from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise _EndpointError(
                    f"OpenAI-compatible request to {self.endpoint} timed out "
                    f"(socket timeout: {self.timeout_seconds}s): {error.reason}",
                    retryable=True,
                ) from error
            raise RuntimeError(
                f"could not reach OpenAI-compatible endpoint at {self.endpoint}: "
                f"{error}"
            ) from error
        except TimeoutError as error:
            raise _EndpointError(
                f"OpenAI-compatible request to {self.endpoint} timed out "
                f"(socket timeout: {self.timeout_seconds}s): {error}",
                retryable=True,
            ) from error

        try:
            response_payload = json.loads(response_body)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned invalid JSON: "
                f"{response_body[:2000].decode('utf-8', errors='replace')}"
            ) from error

        # Hosted gateways can return an error envelope even with HTTP 200.
        if (
            isinstance(response_payload, dict)
            and response_payload.get("error") is not None
        ):
            details = {"error": response_payload["error"]}
            if "id" in response_payload:
                details["id"] = response_payload["id"]
            raise _EndpointError(
                "OpenAI-compatible endpoint returned an API error: "
                f"{json.dumps(details, ensure_ascii=False)}",
                retryable=_is_retryable_error(response_payload["error"]),
            )

        try:
            parsed = _ChatCompletionResponse.model_validate(response_payload)
            text = parsed.choices[0].message.content or ""
        except (ValidationError, IndexError) as error:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned an unexpected response shape: "
                f"{json.dumps(response_payload, ensure_ascii=False)[:2000]}"
            ) from error
        return CompletionResult(
            text=text,
            response=parsed.model_dump(),
            reasoning_content=(
                parsed.choices[0].message.reasoning_content
                or parsed.choices[0].message.reasoning
            ),
        )
