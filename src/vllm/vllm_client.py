"""Minimal client for local or hosted OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError
from src.evaluations.evaluation_config import GenerationConfig, ServerConfig


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
    ) -> None:
        self.endpoint = f"{server.base_url.rstrip('/')}/chat/completions"
        self.api_key = server.api_key
        self.timeout_seconds = server.timeout_seconds
        self.model = model
        self.generation = generation

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
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible endpoint returned HTTP {error.code}: {body}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"could not reach OpenAI-compatible endpoint at {self.endpoint}: "
                f"{error}"
            ) from error

        try:
            parsed = _ChatCompletionResponse.model_validate_json(response_body)
            text = parsed.choices[0].message.content or ""
        except (ValidationError, IndexError) as error:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned an unexpected response shape"
            ) from error
        return CompletionResult(
            text=text,
            response=parsed.model_dump(),
            reasoning_content=(
                parsed.choices[0].message.reasoning_content
                or parsed.choices[0].message.reasoning
            ),
        )
