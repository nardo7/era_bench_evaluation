import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from src.evaluations.evaluation_config import GenerationConfig, ServerConfig
from src.vllm.vllm_client import VLLMClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body.read()


def test_posts_openai_compatible_structured_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"actions": []}',
                            "reasoning_content": "The target output is empty.",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(timeout_seconds=12),
        model="test-model",
        generation=GenerationConfig(
            max_tokens=123,
            request_overrides={
                "stop": ["x"],
                "mm_processor_kwargs": {"fps": 1.0},
            },
        ),
    )

    result = client.complete(
        [{"role": "user", "content": "test"}],
        response_schema={"type": "object"},
    )

    assert result.text == '{"actions": []}'
    assert result.reasoning_content == "The target output is empty."
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["max_completion_tokens"] == 123
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["stop"] == ["x"]
    assert captured["payload"]["mm_processor_kwargs"] == {"fps": 1.0}
    assert captured["authorization"] is None


def test_adds_authorization_only_when_api_key_is_configured(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse({"choices": [{"message": {"content": '{"actions": []}'}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(api_key="secret"),
        model="test-model",
        generation=GenerationConfig(),
    )

    client.complete([{"role": "user", "content": "test"}])

    assert captured["authorization"] == "Bearer secret"


def test_accepts_null_content_from_a_reasoning_model(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return _FakeResponse({"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(), model="qwen3-8b-thinking", generation=GenerationConfig()
    )

    result = client.complete([{"role": "user", "content": "test"}])

    assert result.text == ""
    assert result.reasoning_content is None


def test_accepts_openrouter_reasoning_field(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "reasoning": "Reasoning returned by OpenRouter.",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )

    result = client.complete([{"role": "user", "content": "test"}])

    assert result.reasoning_content == "Reasoning returned by OpenRouter."


def test_uses_configured_max_tokens_field_and_omits_null_seed(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(),
        model="test-model",
        generation=GenerationConfig(
            max_tokens=123,
            max_tokens_field="max_tokens",
            seed=None,
            request_overrides={"video_fps": 1.0, "video_max_frames": 64},
        ),
    )

    client.complete([{"role": "user", "content": "test"}])

    assert captured["payload"]["max_tokens"] == 123
    assert "max_completion_tokens" not in captured["payload"]
    assert "seed" not in captured["payload"]
    assert captured["payload"]["video_fps"] == 1.0
    assert captured["payload"]["video_max_frames"] == 64


def test_omits_null_generation_parameters(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(),
        model="hosted-model",
        generation=GenerationConfig(temperature=None, top_p=None, seed=None),
    )

    client.complete([{"role": "user", "content": "test"}])

    assert "temperature" not in captured["payload"]
    assert "top_p" not in captured["payload"]
    assert "seed" not in captured["payload"]


def test_retries_http_200_provider_error_with_identical_request(monkeypatch, caplog) -> None:
    requests = []
    delays = []

    def fake_urlopen(request, timeout):
        requests.append(request.data)
        if len(requests) == 1:
            return _FakeResponse(
                {
                    "id": "gen-provider-error",
                    "error": {
                        "message": "Provider unavailable",
                        "metadata": {"raw": {"error": {"code": "provider_unavailable"}}},
                    },
                }
            )
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", delays.append)
    client = VLLMClient(
        ServerConfig(),
        model="hosted-model",
        generation=GenerationConfig(
            request_overrides={
                "provider": {"only": ["openai"], "allow_fallbacks": False}
            }
        ),
    )

    assert client.complete([{"role": "user", "content": "test"}]).text == "ok"
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert delays == [2]
    assert "provider_unavailable" in caplog.text
    assert "gen-provider-error" in caplog.text


def test_provider_error_exhaustion_preserves_details(monkeypatch) -> None:
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        return _FakeResponse(
            {
                "id": "gen-failed",
                "error": {
                    "code": 503,
                    "message": "No capacity",
                    "metadata": {"provider_name": "OpenAI"},
                },
            }
        )

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", delays.append)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )

    with pytest.raises(RuntimeError, match="No capacity") as caught:
        client.complete([])

    assert len(attempts) == 4
    assert delays == [2, 4, 8]
    assert "gen-failed" in str(caught.value)
    assert "OpenAI" in str(caught.value)
    assert "503" in str(caught.value)


@pytest.mark.parametrize("status", [429, 502, 503])
def test_retries_transient_http_errors(monkeypatch, status) -> None:
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url, status, "Unavailable", {}, BytesIO(b"provider busy")
            )
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", delays.append)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )

    assert client.complete([]).text == "ok"
    assert len(attempts) == 2
    assert delays == [2]


@pytest.mark.parametrize("status", [400, 401, 403, 402])
def test_does_not_retry_permanent_http_errors(monkeypatch, status) -> None:
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise HTTPError(
            request.full_url, status, "Rejected", {}, BytesIO(b"request rejected")
        )

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )

    with pytest.raises(RuntimeError, match=f"HTTP {status}: request rejected"):
        client.complete([])
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"error": {"code": 400, "message": "Invalid schema"}}, "Invalid schema"),
        ({"unexpected": "payload"}, "unexpected response shape"),
        ({"choices": []}, "unexpected response shape"),
    ],
)
def test_does_not_retry_permanent_api_errors_or_malformed_responses(
    monkeypatch, payload, message
) -> None:
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        return _FakeResponse(payload)

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )

    with pytest.raises(RuntimeError, match=message):
        client.complete([])
    assert len(attempts) == 1


@pytest.mark.parametrize("stage", ["headers", "wrapped", "body"])
def test_retries_socket_timeouts_with_identical_request(monkeypatch, stage) -> None:
    requests = []
    delays = []

    class TimedOutResponse(_FakeResponse):
        def read(self):
            raise TimeoutError("The read operation timed out")

    def fake_urlopen(request, timeout):
        requests.append((request.data, request.full_url, timeout))
        if len(requests) == 1:
            if stage == "body":
                return TimedOutResponse({})
            error = TimeoutError("The read operation timed out")
            if stage == "wrapped":
                raise URLError(error)
            raise error
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", delays.append)
    client = VLLMClient(
        ServerConfig(timeout_seconds=300), model="hosted-model",
        generation=GenerationConfig(request_overrides={
            "provider": {"only": ["openai"], "allow_fallbacks": False}
        }),
    )

    assert client.complete([]).text == "ok"
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert delays == [2]


def test_repeated_timeouts_exhaust_retry_budget(monkeypatch) -> None:
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", delays.append)
    client = VLLMClient(
        ServerConfig(timeout_seconds=300), model="hosted-model",
        generation=GenerationConfig(),
    )

    with pytest.raises(RuntimeError, match="socket timeout: 300s") as caught:
        client.complete([])
    assert len(attempts) == 4
    assert delays == [2, 4, 8]
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_non_timeout_url_errors_are_not_retried(monkeypatch) -> None:
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise URLError("certificate verify failed")

    monkeypatch.setattr("src.vllm.vllm_client.urlopen", fake_urlopen)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )
    with pytest.raises(RuntimeError, match="certificate verify failed"):
        client.complete([])
    assert len(attempts) == 1


def test_retries_provider_unavailable_in_error_type_metadata(monkeypatch) -> None:
    responses = iter([
        {"error": {"metadata": {"error_type": "provider_unavailable"}}},
        {"choices": [{"message": {"content": "ok"}}]},
    ])
    monkeypatch.setattr(
        "src.vllm.vllm_client.urlopen", lambda *args, **kwargs: _FakeResponse(next(responses))
    )
    monkeypatch.setattr("src.vllm.vllm_client.time.sleep", lambda delay: None)
    client = VLLMClient(
        ServerConfig(), model="hosted-model", generation=GenerationConfig()
    )
    assert client.complete([]).text == "ok"
