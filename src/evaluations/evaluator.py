"""End-to-end embodied ARC VLM evaluation loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.datasets import (
    ArcTaskDataset,
    uniform_frame_indices,
)
from src.evaluations.evaluation_config import EvaluationConfig
from src.evaluations.evaluation_metrics import (
    SampleScore,
    aggregate_scores,
    expected_actions_from_timeline,
    parse_action_response,
    parse_grid_response,
    score_grid_response,
    score_response,
)
from src.vllm import VLLMClient
from src.vlm_images import (
    frame_to_data_url,
    frames_to_video_data_url,
    upscale_rgb_frames,
)
from src.vlm_preprocessing import prepare_vlm_task
from src.vlm_prompts import (
    build_vlm_messages,
    response_schema_for_output_mode,
)

FINAL_FRAME_IMAGE_SCALE = 4


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate configured task samples and write JSONL plus summary JSON."""
    config.evaluation.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cameras = (
        tuple(dict.fromkeys(("table_cam", *config.evaluation.cameras)))
        if config.evaluation.include_frames
        else config.evaluation.cameras
    )
    dataset = ArcTaskDataset(
        config.dataset.paths,
        expose_test_targets=True,
        frame_index_selector=uniform_frame_indices(config.dataset.frame_stride),
        cameras=dataset_cameras,
    )
    client = VLLMClient(
        config.server,
        model=config.model.served_name,
        generation=config.generation,
    )
    sample_count = len(dataset)
    if config.evaluation.max_samples is not None:
        sample_count = min(sample_count, config.evaluation.max_samples)

    scores: list[SampleScore] = []
    error_count = 0
    artifact_suffix = _artifact_suffix(
        config.evaluation.prompt_mode,
        config.evaluation.include_frames,
        config.evaluation.include_textual_grids,
        config.evaluation.include_actions,
        config.evaluation.induce_rule_summary,
        config.evaluation.induce_scene_descriptions,
        config.evaluation.include_final_frames,
        config.evaluation.cameras,
    )
    results_path = config.evaluation.output_dir / f"results_{artifact_suffix}.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for index in range(sample_count):
            sample = dataset[index]
            sample_id = f"{sample.task_id}:{sample.test_key}"
            started_at = time.monotonic()
            prompt_mode = config.evaluation.prompt_mode
            mode = _output_mode_for_prompt_mode(prompt_mode)
            include_frames = config.evaluation.include_frames
            include_textual_grids = config.evaluation.include_textual_grids
            include_actions = config.evaluation.include_actions
            induce_rule_summary = config.evaluation.induce_rule_summary
            induce_scene_descriptions = config.evaluation.induce_scene_descriptions
            include_final_frames = config.evaluation.include_final_frames
            try:
                if mode == "actions" and sample.test_case.high_level_actions is None:
                    raise ValueError(f"{sample_id} has no expected high-level actions")
                if mode == "grid" and sample.test_case.expected_output_grid is None:
                    raise ValueError(f"{sample_id} has no expected output grid")
                task_input = prepare_vlm_task(
                    sample, camera=config.evaluation.cameras[0]
                )
                demonstration_video_urls = None
                demonstration_final_image_urls = None
                test_image_url = None
                if include_frames:
                    demonstration_video_urls = [
                        frames_to_video_data_url(
                            sample.demonstrations[demo_index].table_cam_rgb_frames
                        )
                        for demo_index in range(len(task_input.demonstrations))
                    ]
                if include_final_frames:
                    demonstration_final_image_urls = [
                        [
                            (
                                camera,
                                frame_to_data_url(
                                    upscale_rgb_frames(
                                        sample.demonstrations[
                                            demo_index
                                        ].frames_for_camera(camera)[-1],
                                        factor=FINAL_FRAME_IMAGE_SCALE,
                                    )
                                ),
                            )
                            for camera in config.evaluation.cameras
                        ]
                        for demo_index in range(len(task_input.demonstrations))
                    ]
                if include_frames or include_final_frames:
                    test_image_scale = (
                        FINAL_FRAME_IMAGE_SCALE if include_final_frames else 1
                    )
                    test_image_url = [
                        (
                            camera,
                            frame_to_data_url(
                                upscale_rgb_frames(
                                    sample.test_case.frames_for_camera(camera)[0],
                                    factor=test_image_scale,
                                )
                            ),
                        )
                        for camera in config.evaluation.cameras
                    ]
                messages = build_vlm_messages(
                    task_input,
                    output_mode=mode,
                    include_frames=include_frames,
                    include_textual_grids=include_textual_grids,
                    include_actions=include_actions,
                    induce_rule_summary=induce_rule_summary,
                    induce_scene_descriptions=induce_scene_descriptions,
                    include_final_frames=include_final_frames,
                    demonstration_video_urls=demonstration_video_urls,
                    demonstration_final_image_urls=demonstration_final_image_urls,
                    test_image_url=test_image_url,
                )
                completion = client.complete(
                    messages,
                    response_schema=response_schema_for_output_mode(
                        mode, induce_rule_summary, induce_scene_descriptions
                    ),
                )
                if mode == "actions":
                    score = score_response(
                        completion.text,
                        sample.test_case.high_level_actions,
                        task_id=sample.task_id,
                    )
                    predicted, _ = parse_action_response(completion.text)
                    scores.append(score)
                elif mode == "grid":
                    score = score_grid_response(
                        completion.text,
                        sample.test_case.expected_output_grid,
                        task_id=sample.task_id,
                    )
                    predicted, _ = parse_grid_response(completion.text)
                    scores.append(score)
                print(
                    f"[{index + 1}/{sample_count}] {sample_id} "
                    f"exact={score.exact_match} valid={score.valid_action_sequence}",
                    flush=True,
                )
                record = {
                    "sample_id": sample_id,
                    "task_id": sample.task_id,
                    "test_key": sample.test_key,
                    "output_mode": mode,
                    "prompt_mode": prompt_mode,
                    "include_frames": include_frames,
                    "include_actions": include_actions,
                    "induce_rule_summary": induce_rule_summary,
                    "induce_scene_descriptions": induce_scene_descriptions,
                    "include_final_frames": include_final_frames,
                    "cameras": list(config.evaluation.cameras),
                    "include_textual_grids": include_textual_grids,
                    "response_text": completion.text,
                    "predicted_rule_summary": getattr(predicted, "rule_summary", None),
                    "predicted_scene_descriptions": getattr(
                        predicted, "scene_descriptions", None
                    ),
                    "reasoning_content": completion.reasoning_content,
                    "usage": completion.response.get("usage"),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                if mode == "actions":
                    record["predicted_actions"] = (
                        [action.model_dump() for action in predicted.actions]
                        if predicted is not None
                        else None
                    )
                    record["expected_actions"] = expected_actions_from_timeline(
                        sample.test_case.high_level_actions
                    )
                    record["ground_truth_rule"] = sample.ground_truth_rule
                if mode == "grid":
                    record["predicted_output_grid"] = (
                        predicted.output_grid if predicted is not None else None
                    )
                    record["expected_output_grid"] = (
                        sample.test_case.expected_output_grid
                    )
                    record["ground_truth_rule"] = sample.ground_truth_rule
                record["metrics"] = (
                    _grid_metrics(score) if mode == "grid" else score.to_dict()
                )
            except Exception as error:
                error_count += 1
                record = {
                    "sample_id": sample_id,
                    "task_id": sample.task_id,
                    "test_key": sample.test_key,
                    "output_mode": mode,
                    "prompt_mode": prompt_mode,
                    "include_frames": include_frames,
                    "include_actions": include_actions,
                    "induce_rule_summary": induce_rule_summary,
                    "induce_scene_descriptions": induce_scene_descriptions,
                    "include_final_frames": include_final_frames,
                    "cameras": list(config.evaluation.cameras),
                    "include_textual_grids": include_textual_grids,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                print(
                    f"[{index + 1}/{sample_count}] {sample_id} error={record['error']}",
                    flush=True,
                )
                if not config.evaluation.continue_on_error:
                    results_file.write(json.dumps(record) + "\n")
                    results_file.flush()
                    raise
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    summary = {
        "model": config.model.served_name,
        "output_mode": _output_mode_for_prompt_mode(config.evaluation.prompt_mode),
        "prompt_mode": config.evaluation.prompt_mode,
        "include_frames": config.evaluation.include_frames,
        "include_actions": config.evaluation.include_actions,
        "induce_rule_summary": config.evaluation.induce_rule_summary,
        "induce_scene_descriptions": config.evaluation.induce_scene_descriptions,
        "include_final_frames": config.evaluation.include_final_frames,
        "cameras": list(config.evaluation.cameras),
        "include_textual_grids": config.evaluation.include_textual_grids,
        "requested_sample_count": sample_count,
        "error_count": error_count,
    }
    summary.update(
        _aggregate_grid_scores(scores)
        if _output_mode_for_prompt_mode(config.evaluation.prompt_mode) == "grid"
        else aggregate_scores(scores)
    )
    summary_path = config.evaluation.output_dir / f"summary_{artifact_suffix}.json"
    _write_json(summary_path, summary)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _grid_metrics(score: SampleScore) -> dict[str, Any]:
    metrics = score.to_dict()
    metrics["valid_grid"] = metrics.pop("valid_action_sequence")
    return metrics


def _aggregate_grid_scores(scores: list[SampleScore]) -> dict[str, Any]:
    summary = aggregate_scores(scores)
    summary["valid_grid_rate"] = summary.pop("valid_action_sequence_rate")
    return summary


def _artifact_suffix(
    prompt_mode: str,
    include_frames: bool,
    include_textual_grids: bool,
    include_actions: bool,
    induce_rule_summary: bool,
    induce_scene_descriptions: bool,
    include_final_frames: bool,
    cameras: tuple[str, ...],
) -> str:
    """Name every prompt-context combination distinctly for comparisons."""
    contexts = [
        name
        for name, enabled in (
            ("videos", include_frames),
            ("final_frames", include_final_frames),
            ("grids", include_textual_grids),
            ("actions", include_actions),
        )
        if enabled
    ]
    suffix_parts = [_output_mode_for_prompt_mode(prompt_mode), prompt_mode, *contexts]
    if cameras != ("table_cam",):
        suffix_parts.append("-".join(cameras))
    suffix = "_".join(suffix_parts)
    if induce_scene_descriptions:
        suffix += "_with_scene_descriptions"
    return suffix if induce_rule_summary else f"{suffix}_without_rule_summary"


def _output_mode_for_prompt_mode(prompt_mode: str) -> str:
    return {"embodied": "actions", "grid_focused": "grid"}[prompt_mode]
