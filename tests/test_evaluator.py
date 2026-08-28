import json
from pathlib import Path

import h5py
import numpy as np
from src.evaluations.evaluation_config import (
    DatasetConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    GenerationConfig,
    ModelConfig,
    ServerConfig,
)
from src.evaluations.evaluator import run_evaluation
from src.vllm.vllm_client import CompletionResult


def _write_dataset(dataset_path: Path) -> None:
    action_timeline = np.array([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 1]]], dtype=np.int64)
    with h5py.File(dataset_path, "w") as dataset:
        task_group = dataset.create_group("data").create_group("task_001")
        rule_group = task_group.create_group("rule")
        rule_group.create_dataset(
            "data",
            data=(
                '{"task_id": "task_001", "inferred_rule": "Copy the red input '
                'cube to output column 1.", "task_type": "evaluation", '
                '"reference_claims": []}'
            ),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        grid_metadata = task_group.create_group("arc_grids")
        for episode_name in ("demo_0", "test"):
            pair = grid_metadata.create_group(episode_name)
            pair.create_dataset("input_grid", data=[[2]])
            pair.create_dataset("output_grid", data=[[0, 2]])
        for episode_name in ("demo_0", "test"):
            episode = task_group.create_group(episode_name)
            observations = episode.create_group("obs")
            observations.create_dataset(
                "table_cam",
                data=np.arange(3, dtype=np.uint8)[:, None, None, None]
                * np.ones((1, 4, 4, 3), dtype=np.uint8),
            )
            observations.create_dataset(
                "origin_top_cam",
                data=(10 + np.arange(3, dtype=np.uint8))[:, None, None, None]
                * np.ones((1, 4, 4, 3), dtype=np.uint8),
            )
            episode.create_group("high_level_actions").create_dataset(
                "action_timeline", data=action_timeline
            )


def test_runs_end_to_end_with_a_stubbed_completion(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_dataset(dataset_path)

    def fake_complete(_self, messages, *, response_schema, request_overrides=None):
        content = messages[1]["content"]
        assert sum(part.get("type") == "video_url" for part in content) == 1
        assert sum(part.get("type") == "image_url" for part in content) == 1
        assert request_overrides is None
        assert set(response_schema["required"]) == {"rule_summary", "actions"}
        return CompletionResult(
            text=(
                '{"rule_summary": "Copy the red input cube to output column 1.", "actions": ['
                '{"phase": "grasp", "color_id": 2, "location_type": "reserve", "row": -1, "column": -1},'
                '{"phase": "place", "color_id": 2, "location_type": "output", "row": 0, "column": 1}'
                "]}"
            ),
            response={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
            reasoning_content="Copy the red input cube to output column 1.",
        )

    monkeypatch.setattr("src.evaluations.evaluator.VLLMClient.complete", fake_complete)
    config = EvaluationConfig(
        dataset=DatasetConfig(paths=[dataset_path], frame_stride=2),
        model=ModelConfig(model_id="test", served_name="test"),
        server=ServerConfig(),
        generation=GenerationConfig(
            request_overrides={"mm_processor_kwargs": {"fps": 1.0}}
        ),
        evaluation=EvaluationRunConfig(output_dir=tmp_path / "output"),
    )

    summary = run_evaluation(config)

    assert summary["exact_match_rate"] == 1.0
    assert summary["valid_action_sequence_rate"] == 1.0
    assert summary["colored_cell_f1"] == 1.0
    assert summary["occupancy_exact_match_rate"] == 1.0
    assert summary["occupied_cell_f1"] == 1.0
    record = json.loads(
        (tmp_path / "output/results_actions_embodied_videos_actions.jsonl").read_text()
    )
    assert record["sample_id"] == "task_001:test"
    assert record["prompt_mode"] == "embodied"
    assert record["include_frames"]
    assert record["include_actions"]
    assert record["induce_rule_summary"]
    assert not record["include_textual_grids"]
    assert record["metrics"]["occupancy_exact_match"]
    assert (
        record["predicted_rule_summary"]
        == "Copy the red input cube to output column 1."
    )
    assert record["ground_truth_rule"] == "Copy the red input cube to output column 1."
    assert record["reasoning_content"] == "Copy the red input cube to output column 1."
    assert record["usage"]["prompt_tokens"] == 10


def test_runs_grid_focused_evaluation_without_media(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_dataset(dataset_path)

    def fake_complete(_self, messages, *, response_schema, request_overrides=None):
        content = messages[1]["content"]
        assert all(part["type"] == "text" for part in content)
        assert "scene shows an ARC input grid" not in messages[0]["content"]
        assert "rule_summary" not in str(messages)
        assert request_overrides is None
        assert response_schema["required"] == ["output_grid"]
        return CompletionResult(
            text='{"output_grid": [[0, 2]]}',
            response={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        )

    monkeypatch.setattr("src.evaluations.evaluator.VLLMClient.complete", fake_complete)
    monkeypatch.setattr(
        "src.evaluations.evaluator.frames_to_video_data_url",
        lambda _frames: (_ for _ in ()).throw(AssertionError("video encoding called")),
    )
    monkeypatch.setattr(
        "src.evaluations.evaluator.frame_to_data_url",
        lambda _frame: (_ for _ in ()).throw(AssertionError("image encoding called")),
    )
    config = EvaluationConfig(
        dataset=DatasetConfig(paths=[dataset_path], frame_stride=2),
        model=ModelConfig(model_id="test", served_name="test"),
        server=ServerConfig(),
        generation=GenerationConfig(),
        evaluation=EvaluationRunConfig(
            output_dir=tmp_path / "output",
            prompt_mode="grid_focused",
            include_frames=False,
            induce_rule_summary=False,
        ),
    )

    summary = run_evaluation(config)

    assert summary["exact_match_rate"] == 1.0
    assert summary["valid_grid_rate"] == 1.0
    assert summary["prompt_mode"] == "grid_focused"
    assert not summary["include_frames"]
    assert not summary["include_actions"]
    assert not summary["induce_rule_summary"]
    record = json.loads(
        (
            tmp_path
            / "output/results_grid_grid_focused_grids_without_rule_summary.jsonl"
        ).read_text()
    )
    assert record["prompt_mode"] == "grid_focused"
    assert not record["induce_rule_summary"]
    assert record["predicted_rule_summary"] is None
    assert record["predicted_output_grid"] == [[0, 2]]
    assert (
        tmp_path / "output/summary_grid_grid_focused_grids_without_rule_summary.json"
    ).exists()


def test_runs_grid_focused_evaluation_with_final_images(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_dataset(dataset_path)

    def fake_complete(_self, messages, *, response_schema, request_overrides=None):
        content = messages[1]["content"]
        assert not any(part["type"] == "video_url" for part in content)
        assert sum(part["type"] == "image_url" for part in content) == 2
        assert request_overrides is None
        assert set(response_schema["required"]) == {"output_grid", "rule_summary"}
        return CompletionResult(
            text=(
                '{"rule_summary": "Copy the red input cube to output column 1.", '
                '"output_grid": [[0, 2]]}'
            ),
            response={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        )

    monkeypatch.setattr("src.evaluations.evaluator.VLLMClient.complete", fake_complete)
    monkeypatch.setattr(
        "src.evaluations.evaluator.frames_to_video_data_url",
        lambda _frames: (_ for _ in ()).throw(AssertionError("video encoding called")),
    )
    encoded_frame_values: list[int] = []

    def encode_frame(frame):
        encoded_frame_values.append(int(frame[0, 0, 0]))
        return f"data:image/png;base64,{encoded_frame_values[-1]}"

    monkeypatch.setattr("src.evaluations.evaluator.frame_to_data_url", encode_frame)
    upscale_factors: list[int] = []

    def upscale_frame(frame, *, factor):
        upscale_factors.append(factor)
        return frame

    monkeypatch.setattr("src.evaluations.evaluator.upscale_rgb_frames", upscale_frame)
    config = EvaluationConfig(
        dataset=DatasetConfig(paths=[dataset_path], frame_stride=2),
        model=ModelConfig(model_id="test", served_name="test"),
        server=ServerConfig(),
        generation=GenerationConfig(),
        evaluation=EvaluationRunConfig(
            output_dir=tmp_path / "output",
            prompt_mode="grid_focused",
            include_frames=False,
            include_final_frames=True,
        ),
    )

    summary = run_evaluation(config)

    assert summary["exact_match_rate"] == 1.0
    assert summary["include_final_frames"]
    assert encoded_frame_values == [2, 0]
    assert upscale_factors == [4, 4]
    record = json.loads(
        (
            tmp_path / "output/results_grid_grid_focused_final_frames_grids.jsonl"
        ).read_text()
    )
    assert record["include_final_frames"]


def test_runs_evaluation_with_both_camera_perspectives(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "dataset.hdf5"
    _write_dataset(dataset_path)

    def fake_complete(_self, messages, *, response_schema, request_overrides=None):
        content = messages[1]["content"]
        assert sum(part["type"] == "video_url" for part in content) == 1
        assert sum(part["type"] == "image_url" for part in content) == 4
        assert [part["text"] for part in content if part["type"] == "text"].count(
            "Final frame — camera perspective: table_cam"
        ) == 1
        assert [part["text"] for part in content if part["type"] == "text"].count(
            "Final frame — camera perspective: origin_top_cam"
        ) == 1
        return CompletionResult(
            text=(
                '{"rule_summary": "Copy the red input cube to output column 1.", "actions": ['
                '{"phase": "grasp", "color_id": 2, "location_type": "reserve", "row": -1, "column": -1},'
                '{"phase": "place", "color_id": 2, "location_type": "output", "row": 0, "column": 1}'
                "]}"
            ),
            response={},
        )

    monkeypatch.setattr("src.evaluations.evaluator.VLLMClient.complete", fake_complete)
    config = EvaluationConfig(
        dataset=DatasetConfig(paths=[dataset_path], frame_stride=2),
        model=ModelConfig(model_id="test", served_name="test"),
        server=ServerConfig(),
        generation=GenerationConfig(),
        evaluation=EvaluationRunConfig(
            output_dir=tmp_path / "output",
            cameras=("table_cam", "origin_top_cam"),
            include_final_frames=True,
        ),
    )

    summary = run_evaluation(config)

    assert summary["cameras"] == ["table_cam", "origin_top_cam"]
    record = json.loads(
        (
            tmp_path
            / "output/results_actions_embodied_videos_final_frames_actions_table_cam-origin_top_cam.jsonl"
        ).read_text()
    )
    assert record["cameras"] == ["table_cam", "origin_top_cam"]
