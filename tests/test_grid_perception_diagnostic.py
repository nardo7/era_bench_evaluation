from src.evaluations.evaluation_config import GenerationConfig
from src.evaluations.grid_perception_diagnostic import (
    DemonstrationSceneThenStructuredObservation,
    _evaluate,
    _messages,
    _processor_overrides,
)
from src.vllm.vllm_client import CompletionResult


def test_omits_empty_vllm_processor_overrides() -> None:
    class _Config:
        generation = GenerationConfig()

    assert _processor_overrides(_Config(), fps=None) == {}


def test_processor_overrides_preserve_non_fps_values_and_set_requested_fps() -> None:
    class _Config:
        generation = GenerationConfig(
            request_overrides={"mm_processor_kwargs": {"max_pixels": 1024, "fps": 1}}
        )

    assert _processor_overrides(_Config(), fps=2.0) == {
        "mm_processor_kwargs": {"max_pixels": 1024, "fps": 2.0}
    }


def test_scene_then_structured_requires_one_scene_description_before_grids() -> None:
    observation = {
        "observation_type": "demonstration",
        "input_grid": [[1, 0]],
        "output_grid": [[0]],
        "media": [],
    }

    messages = _messages(observation, response_format="scene_then_structured")
    instruction = messages[1]["content"][-1]["text"]

    assert list(
        DemonstrationSceneThenStructuredObservation.model_json_schema()["properties"]
    ) == [
        "scene_description",
        "input_grid",
        "output_grid",
    ]
    assert "one integrated description" in instruction
    assert "do not create separate descriptions for each grid" in instruction
    assert "scene_description, then input_grid and output_grid" in instruction


def test_scene_then_structured_persists_description_and_scores_grids() -> None:
    class _Client:
        def complete(self, messages, *, response_schema, request_overrides):
            assert messages[0]["role"] == "system"
            assert response_schema is not None
            assert list(response_schema["properties"]) == [
                "scene_description",
                "input_grid",
                "output_grid",
            ]
            return CompletionResult(
                text=(
                    '{"scene_description":"The left grid is visible.",'
                    '"input_grid":[[1,0]],"output_grid":[[0]]}'
                ),
                response={"usage": {"total_tokens": 1}},
            )

    result = _evaluate(
        _Client(),
        {
            "observation_type": "demonstration",
            "input_grid": [[1, 0]],
            "output_grid": [[0]],
            "media": [],
        },
        response_format="scene_then_structured",
        request_overrides={},
    )

    assert result["scene_description"] == "The left grid is visible."
    assert result["metrics"]["both_grids_exact_match"] is True
