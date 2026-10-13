import random
import sys
from types import SimpleNamespace

import pytest

from scripts import diagnose_grid_perception, diagnose_scene_description
from src.evaluations.evaluation_config import GenerationConfig
from src.evaluations.grid_perception_diagnostic import (
    DemonstrationSceneThenStructuredObservation,
    _effective_seed,
    _evaluate,
    _messages,
    _processor_overrides,
    _task_indices,
    _update_all_grids_metrics,
)
from src.vllm.vllm_client import CompletionResult


@pytest.fixture
def task_dataset():
    # Interleaved test pairs must stay with their task even when it is limited.
    return SimpleNamespace(
        samples=[
            SimpleNamespace(task_id=task_id)
            for task_id in ("a", "b", "c", "a", "d", "c", "e")
        ]
    )


def test_task_selection_preserves_default_order_and_complete_tasks(task_dataset):
    assert list(_task_indices(task_dataset, None)) == ["a", "b", "c", "d", "e"]
    assert _task_indices(task_dataset, 2) == {"a": [0, 3], "b": [1]}
    assert _task_indices(task_dataset, 2, shuffle_seed=42) == {"a": [0, 3], "b": [1]}


def test_task_shuffle_is_reproducible_and_precedes_limit(task_dataset):
    original_samples = list(task_dataset.samples)
    random_state = random.getstate()
    shuffled = _task_indices(task_dataset, None, shuffle_tasks=True, shuffle_seed=0)
    repeated = _task_indices(task_dataset, None, shuffle_tasks=True, shuffle_seed=0)
    limited = _task_indices(task_dataset, 1, shuffle_tasks=True, shuffle_seed=0)

    assert list(shuffled) == list(repeated)
    assert list(shuffled) != list(_task_indices(task_dataset, None))
    assert list(shuffled) != list(
        _task_indices(task_dataset, None, shuffle_tasks=True, shuffle_seed=42)
    )
    assert list(limited) == list(shuffled)[:1]
    assert list(limited) != ["a"]
    assert shuffled["a"] == [0, 3]
    assert shuffled["c"] == [2, 5]
    assert sorted(index for indices in shuffled.values() for index in indices) == list(
        range(len(original_samples))
    )
    assert task_dataset.samples == original_samples
    assert random.getstate() == random_state


@pytest.mark.parametrize("shuffle_tasks", [False, True])
def test_task_selection_handles_empty_dataset_and_large_limit(
    task_dataset, shuffle_tasks
):
    assert (
        _task_indices(SimpleNamespace(samples=[]), 3, shuffle_tasks=shuffle_tasks) == {}
    )
    assert len(_task_indices(task_dataset, 20, shuffle_tasks=shuffle_tasks)) == 5


@pytest.mark.parametrize(
    "configured_seed,requested_seed,expected_seed",
    [(17, None, 17), (17, 42, 42), (None, None, 0), (None, 42, 42)],
)
def test_run_seed_uses_cli_override_then_config_then_reproducible_default(
    configured_seed, requested_seed, expected_seed
):
    config = SimpleNamespace(generation=GenerationConfig(seed=configured_seed))

    assert _effective_seed(config, requested_seed) == expected_seed


def test_all_grids_exact_combines_shared_demos_with_each_test_pair() -> None:
    totals = {"all_grids_exact": 0}
    demo_context_exact = {"a": True, "b": True}

    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "a", "observation_type": "demonstration"},
        {"both_grids_exact_match": True},
    )
    exact_test = {"input_grid_exact_match": True}
    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "a", "observation_type": "test"},
        exact_test,
    )
    inexact_second_test = {"input_grid_exact_match": False}
    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "a", "observation_type": "test"},
        inexact_second_test,
    )

    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "b", "observation_type": "demonstration"},
        {"both_grids_exact_match": False},
    )
    blocked_test = {"input_grid_exact_match": True}
    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "b", "observation_type": "test"},
        blocked_test,
    )

    assert exact_test["all_grids_exact"] is True
    assert inexact_second_test["all_grids_exact"] is False
    assert blocked_test["all_grids_exact"] is False
    assert totals["all_grids_exact"] == 1


def test_all_grids_exact_treats_request_errors_as_non_matches() -> None:
    totals = {"all_grids_exact": 0}
    demo_context_exact = {"a": True, "b": True}

    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "a", "observation_type": "demonstration"},
        None,
    )
    test_after_demo_error = {"input_grid_exact_match": True}
    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "a", "observation_type": "test"},
        test_after_demo_error,
    )
    _update_all_grids_metrics(
        totals,
        demo_context_exact,
        {"task_id": "b", "observation_type": "test"},
        None,
    )

    assert test_after_demo_error["all_grids_exact"] is False
    assert totals["all_grids_exact"] == 0


@pytest.mark.parametrize(
    "script,runner_name,condition_count",
    [
        (diagnose_grid_perception, "run_grid_perception_diagnostic", 6),
        (diagnose_scene_description, "run_scene_description_diagnostic", 1),
    ],
)
@pytest.mark.parametrize(
    "flags,expected_shuffle,expected_seed",
    [
        ([], False, None),
        (["--shuffle-tasks"], True, None),
        (["--shuffle-tasks", "--seed", "42"], True, 42),
        (["--no-shuffle-tasks"], False, None),
    ],
)
def test_diagnostic_cli_forwards_task_shuffle_to_every_condition(
    monkeypatch,
    script,
    runner_name,
    condition_count,
    flags,
    expected_shuffle,
    expected_seed,
):
    calls = []

    def run_diagnostic(config, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        sys, "argv", [script.__file__, "--config", "config.yaml", *flags]
    )
    monkeypatch.setattr(script, "load_evaluation_config", lambda path: object())
    monkeypatch.setattr(script, runner_name, run_diagnostic)

    script.main()

    assert len(calls) == condition_count
    assert all(call["shuffle_tasks"] is expected_shuffle for call in calls)
    assert all(call["seed"] == expected_seed for call in calls)


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
