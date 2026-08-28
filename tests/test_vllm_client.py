import json
from io import BytesIO

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
