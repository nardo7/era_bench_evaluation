"""Single-episode, full-scene visual grid transcription diagnostic."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import time
from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.datasets import (
    ArcTaskDataset,
    ArcTaskSample,
    uniform_frame_indices,
)
from src.evaluations import EvaluationConfig
from src.vllm import VLLMClient
from src.vlm_images import (
    frame_to_data_url,
    frames_to_mp4_data_url,
    frames_to_video_data_url,
    upscale_rgb_frames,
)
from src.vlm_prompts import RENDERED_SCENE_BACKGROUND_PROMPT

MediaMode = Literal["final_frame", "sampled_frames", "sampled_video"]
ResponseFormat = Literal["structured", "json", "scene_then_structured"]
SUPPORTED_CAMERAS = ("table_cam", "origin_top_cam")
ARCGrid = list[list[Annotated[int, Field(ge=0, le=9)]]]

SYSTEM_PROMPT = f"""You transcribe ARC grids from one complete robot-scene observation.
Your sole task is to reproduce the visible grids exactly as integer arrays.

The ARC input grid is on the left and the ARC output grid is on the right. The
robot and the vertical reserve-cube stacks in the middle are not part of either
grid.

{RENDERED_SCENE_BACKGROUND_PROMPT}
Input and output grids are centered on fixed world-space center points, so their
top-left lattice corners move when their dimensions change. Preserve logical row
order from top to bottom and column order from left to right.

The same physical scene may be shown from one or two synchronized, labelled
camera perspectives. When two are provided, combine their evidence and
transcribe each grid only once.

Use ARC color IDs: 0 black/background, 1 blue, 2 red, 3 green, 4 yellow, 5 gray,
6 magenta, 7 orange, 8 teal, and 9 burgundy. Return every cell, including zeros.
"""

DESCRIPTION_SYSTEM_PROMPT = f"""You are inspecting one complete rendered robot workcell image.
This is a visual-perception diagnostic: describe only what you can actually see.
Do not infer an ARC transformation, solve the task, return grid arrays, or use ARC
numeric color IDs.

The input-grid area is on the left, the robot is in the middle with reserve-cube
columns below it, and the output-grid area is on the right.
The reserve area is only a supply of cubes used to physically construct the
output.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The robot may partly occlude any of these areas. Be explicit about which
positions are visibly occupied, visibly empty, uncertain, or occluded. Use
ordinary color names.
"""

SCENE_THEN_STRUCTURED_SYSTEM_PROMPT = f"""You transcribe ARC grids from one complete robot-scene observation.
Before transcribing the arrays, make one integrated prose inventory of the whole
scene. The inventory is visual evidence for the later transcription, not an ARC
transformation or solution explanation.

The input-grid area is on the left, the robot is in the middle with reserve-cube
columns below it, and the output-grid area is on the right. The reserve area is
only a supply of cubes used to physically construct the output.

{RENDERED_SCENE_BACKGROUND_PROMPT}
The robot may partly occlude any area. In the prose inventory, use ordinary
color names and be explicit about occupied, empty, uncertain, and occluded
positions. Do not use ARC numeric color IDs in that prose inventory.

After the inventory, transcribe the grids with ARC color IDs: 0 black/background,
1 blue, 2 red, 3 green, 4 yellow, 5 gray, 6 magenta, 7 orange, 8 teal, and
9 burgundy. Preserve logical row order from top to bottom and column order from
left to right, and return every cell including zeros.
"""


class DemonstrationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_grid: ARCGrid
    output_grid: ARCGrid


class TestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_grid: ARCGrid


class DemonstrationSceneThenStructuredObservation(BaseModel):
    """A prose visual inventory followed by a strict grid transcription."""

    model_config = ConfigDict(extra="forbid", strict=True)
    scene_description: Annotated[str, Field(min_length=1)]
    input_grid: ARCGrid
    output_grid: ARCGrid


class TestSceneThenStructuredObservation(BaseModel):
    """A prose visual inventory followed by a strict test-input transcription."""

    model_config = ConfigDict(extra="forbid", strict=True)
    scene_description: Annotated[str, Field(min_length=1)]
    input_grid: ARCGrid


def run_grid_perception_diagnostic(
    config: EvaluationConfig,
    *,
    media_mode: MediaMode,
    response_format: ResponseFormat,
    output_dir: Path | None = None,
    max_tasks: int | None = None,
    shuffle_tasks: bool = False,
    seed: int | None = None,
    video_fps: float | None = None,
    cameras: tuple[str, ...] = ("table_cam",),
    image_scale: int = 1,
) -> dict[str, Any]:
    """Evaluate each demo/test scene independently with exact-array scoring."""
    _validate_options(media_mode, response_format, max_tasks, image_scale)
    fps = config.dataset.sampled_fps if video_fps is None else video_fps
    if fps <= 0:
        raise ValueError("video_fps must be positive")
    seed = _effective_seed(config, seed)
    cameras = _normalize_cameras(cameras)
    selector = uniform_frame_indices(config.dataset.frame_stride)
    dataset = ArcTaskDataset(
        config.dataset.paths,
        expose_test_targets=False,
        frame_index_selector=selector,
        cameras=cameras,
    )
    task_indices = _task_indices(
        dataset, max_tasks, shuffle_tasks=shuffle_tasks, shuffle_seed=seed
    )

    destination = output_dir or (
        config.evaluation.output_dir / "grid_perception_diagnostic"
    )
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f"{media_mode}_{response_format}_scale{image_scale}_{'-'.join(cameras)}"
    client = VLLMClient(
        config.server,
        model=config.model.served_name,
        generation=replace(config.generation, seed=seed),
    )
    image_overrides = _processor_overrides(config, fps=None)
    demo_overrides = (
        image_overrides
        if media_mode == "final_frame"
        else _processor_overrides(config, fps=fps)
    )
    totals = {
        "records": 0,
        "valid": 0,
        "demos": 0,
        "demo_inputs": 0,
        "demo_outputs": 0,
        "demo_pairs": 0,
        "tests": 0,
        "test_inputs": 0,
        "exact_grids": 0,
        "expected_grids": 0,
        "all_grids_exact": 0,
    }
    demo_context_exact = {task_id: True for task_id in task_indices}
    error_count = 0
    expected_records = sum(
        len(dataset.samples[indices[0]].demo_keys) + len(indices)
        for indices in task_indices.values()
    )

    results_path = destination / f"results_{suffix}.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        observations = _observations(
            dataset,
            task_indices,
            cameras=cameras,
            media_mode=media_mode,
            fps=fps,
            image_scale=image_scale,
        )
        for completed, observation in enumerate(observations, start=1):
            _count_expected(totals, observation)
            try:
                overrides = (
                    demo_overrides
                    if observation["observation_type"] == "demonstration"
                    else image_overrides
                )
                record = _evaluate(
                    client,
                    observation,
                    response_format=response_format,
                    request_overrides=overrides,
                )
                _update_totals(totals, record["metrics"], observation)
            except Exception as error:
                error_count += 1
                record = {"error": f"{type(error).__name__}: {error}"}
                if not config.evaluation.continue_on_error:
                    _write_record(
                        results_file,
                        _record(
                            observation,
                            record,
                            media_mode,
                            response_format,
                            cameras,
                            fps,
                            image_scale,
                        ),
                    )
                    raise
            _update_all_grids_metrics(
                totals,
                demo_context_exact,
                observation,
                record.get("metrics"),
            )
            combined = _record(
                observation,
                record,
                media_mode,
                response_format,
                cameras,
                fps,
                image_scale,
            )
            _write_record(results_file, combined)
            status = combined.get("metrics", combined.get("error"))
            print(
                f"[{completed}/{expected_records}] {observation['record_id']} {status}",
                flush=True,
            )

    summary = {
        "model": config.model.served_name,
        "diagnostic": "single_episode_grid_perception",
        "media_mode": media_mode,
        "response_format": response_format,
        "cameras": list(cameras),
        "video_fps": fps,
        "image_scale": image_scale,
        "task_count": len(task_indices),
        "shuffle_tasks": shuffle_tasks,
        "seed": seed,
        "record_count": totals["records"],
        "error_count": error_count,
        "valid_response_rate": _rate(totals["valid"], totals["records"]),
        "demonstration_input_grid_exact_match_rate": _rate(
            totals["demo_inputs"], totals["demos"]
        ),
        "demonstration_output_grid_exact_match_rate": _rate(
            totals["demo_outputs"], totals["demos"]
        ),
        "demonstration_pair_exact_match_rate": _rate(
            totals["demo_pairs"], totals["demos"]
        ),
        "test_input_grid_exact_match_rate": _rate(
            totals["test_inputs"], totals["tests"]
        ),
        "overall_grid_exact_match_rate": _rate(
            totals["exact_grids"], totals["expected_grids"]
        ),
        "all_grids_exact_match_rate": _rate(
            totals["all_grids_exact"], totals["tests"]
        ),
        "exact_grid_count": totals["exact_grids"],
        "expected_grid_count": totals["expected_grids"],
    }
    with (destination / f"summary_{suffix}.json").open(
        "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    return summary


def run_scene_description_diagnostic(
    config: EvaluationConfig,
    *,
    output_dir: Path | None = None,
    max_tasks: int | None = None,
    shuffle_tasks: bool = False,
    seed: int | None = None,
    cameras: tuple[str, ...] = ("table_cam",),
    image_scale: int = 1,
) -> dict[str, Any]:
    """Ask for a free-form cube inventory of each final/initial scene image."""
    _validate_options("final_frame", "json", max_tasks, image_scale)
    seed = _effective_seed(config, seed)
    cameras = _normalize_cameras(cameras)
    selector = uniform_frame_indices(config.dataset.frame_stride)
    dataset = ArcTaskDataset(
        config.dataset.paths,
        expose_test_targets=False,
        frame_index_selector=selector,
        cameras=cameras,
    )
    task_indices = _task_indices(
        dataset, max_tasks, shuffle_tasks=shuffle_tasks, shuffle_seed=seed
    )
    expected_records = sum(
        len(dataset.samples[indices[0]].demo_keys) + len(indices)
        for indices in task_indices.values()
    )

    destination = output_dir or (
        config.evaluation.output_dir / "scene_description_diagnostic"
    )
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f"scale{image_scale}_{'-'.join(cameras)}"
    media_dir = destination / f"media_{suffix}"
    media_dir.mkdir(exist_ok=True)
    client = VLLMClient(
        config.server,
        model=config.model.served_name,
        generation=replace(config.generation, seed=seed),
    )
    request_overrides = _processor_overrides(config, fps=None)
    completed_count = 0
    error_count = 0
    results_path = destination / f"results_{suffix}.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        observations = _observations(
            dataset,
            task_indices,
            cameras=cameras,
            media_mode="final_frame",
            fps=config.dataset.sampled_fps,
            image_scale=image_scale,
        )
        for index, observation in enumerate(observations, start=1):
            messages = _description_messages(observation)
            media_manifest = _save_description_media(observation, media_dir)
            started_at = time.monotonic()
            try:
                completion = client.complete(
                    messages,
                    response_schema=None,
                    request_overrides=request_overrides,
                )
                result = {
                    "description": completion.text,
                    "reasoning_content": completion.reasoning_content,
                    "usage": completion.response.get("usage"),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                completed_count += 1
            except Exception as error:  # noqa: BLE001
                error_count += 1
                result = {"error": f"{type(error).__name__}: {error}"}
            record = {
                "record_id": observation["record_id"],
                "task_id": observation["task_id"],
                "episode_key": observation["episode_key"],
                "observation_type": observation["observation_type"],
                "diagnostic": "single_episode_scene_description",
                "cameras": list(cameras),
                "image_scale": image_scale,
                "request": {
                    "system_prompt": DESCRIPTION_SYSTEM_PROMPT,
                    "user_text": [
                        part["text"]
                        for part in messages[1]["content"]
                        if part["type"] == "text"
                    ],
                    "media": media_manifest,
                    "request_overrides": request_overrides,
                },
                "canonical_input_grid": observation["input_grid"],
                "canonical_output_grid": observation["output_grid"],
                **result,
            }
            _write_record(results_file, record)
            status = "ok" if "description" in result else result["error"]
            print(
                f"[{index}/{expected_records}] {observation['record_id']} {status}",
                flush=True,
            )
            if "error" in result and not config.evaluation.continue_on_error:
                raise RuntimeError(result["error"])

    summary = {
        "model": config.model.served_name,
        "diagnostic": "single_episode_scene_description",
        "cameras": list(cameras),
        "image_scale": image_scale,
        "task_count": len(task_indices),
        "shuffle_tasks": shuffle_tasks,
        "seed": seed,
        "record_count": expected_records,
        "completed_count": completed_count,
        "error_count": error_count,
        "results_path": str(results_path),
        "media_dir": str(media_dir),
    }
    with (destination / f"summary_{suffix}.json").open(
        "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    return summary


def _observations(
    dataset: ArcTaskDataset,
    task_indices: dict[str, list[int]],
    *,
    cameras: tuple[str, ...],
    media_mode: MediaMode,
    fps: float,
    image_scale: int,
) -> Iterator[dict[str, Any]]:
    for task_id, indices in task_indices.items():
        sample = dataset[indices[0]]
        primary = sample
        for demo_index, demonstration in enumerate(primary.demonstrations):
            yield {
                "record_id": f"{task_id}:demo_{demo_index}",
                "task_id": task_id,
                "episode_key": f"demo_{demo_index}",
                "observation_type": "demonstration",
                "input_grid": demonstration.input_grid,
                "output_grid": demonstration.output_grid,
                "media": _demo_media(
                    sample, cameras, demo_index, media_mode, fps, image_scale
                ),
            }
        for index in indices:
            sample = dataset[index]
            primary = sample
            yield {
                "record_id": f"{task_id}:{primary.test_key}",
                "task_id": task_id,
                "episode_key": primary.test_key,
                "observation_type": "test",
                "input_grid": primary.test_case.input_grid,
                "output_grid": None,
                "media": _test_media(sample, cameras, image_scale),
            }


def _evaluate(
    client: VLLMClient,
    observation: dict[str, Any],
    *,
    response_format: ResponseFormat,
    request_overrides: dict[str, Any],
) -> dict[str, Any]:
    is_demo = observation["observation_type"] == "demonstration"
    response_model = _response_model(is_demo, response_format)
    started_at = time.monotonic()
    completion = client.complete(
        _messages(observation, response_format=response_format),
        response_schema=(
            response_model.model_json_schema() if response_format != "json" else None
        ),
        request_overrides=request_overrides,
    )
    try:
        predicted = response_model.model_validate_json(completion.text)
        parse_error = None
    except ValidationError as error:
        predicted = None
        parse_error = str(error)
    input_grid = predicted.input_grid if predicted is not None else None
    output_grid = predicted.output_grid if is_demo and predicted is not None else None
    input_exact = input_grid == observation["input_grid"] and predicted is not None
    output_exact = (
        output_grid == observation["output_grid"] and predicted is not None
        if is_demo
        else None
    )
    metrics = {
        "valid_response": parse_error is None,
        "input_grid_exact_match": input_exact,
        "parse_error": parse_error,
    }
    if is_demo:
        metrics.update(
            output_grid_exact_match=output_exact,
            both_grids_exact_match=input_exact and output_exact,
        )
    return {
        "response_text": completion.text,
        "scene_description": (
            predicted.scene_description
            if response_format == "scene_then_structured" and predicted is not None
            else None
        ),
        "predicted_input_grid": input_grid,
        "predicted_output_grid": output_grid if is_demo else None,
        "expected_input_grid": observation["input_grid"],
        "expected_output_grid": observation["output_grid"] if is_demo else None,
        "metrics": metrics,
        "reasoning_content": completion.reasoning_content,
        "usage": completion.response.get("usage"),
        "elapsed_seconds": time.monotonic() - started_at,
    }


def _response_model(is_demo: bool, response_format: ResponseFormat) -> type[BaseModel]:
    if response_format == "scene_then_structured":
        return (
            DemonstrationSceneThenStructuredObservation
            if is_demo
            else TestSceneThenStructuredObservation
        )
    return DemonstrationObservation if is_demo else TestObservation


def _messages(
    observation: dict[str, Any], *, response_format: ResponseFormat
) -> list[dict[str, Any]]:
    is_demo = observation["observation_type"] == "demonstration"
    input_size = _grid_size(observation["input_grid"])
    if is_demo:
        output_size = _grid_size(observation["output_grid"])
        heading = (
            f"This solved demonstration has a {input_size[0]}x{input_size[1]} "
            f"input grid and a {output_size[0]}x{output_size[1]} output grid. "
            "Transcribe both grids."
        )
        if any(media_type == "video_url" for _, media_type, _ in observation["media"]):
            heading += (
                " The video frames are ordered; the completed output is visible "
                "at the end."
            )
        instruction = _response_instruction(is_demo, response_format)
    else:
        heading = (
            f"This test initial state has a {input_size[0]}x{input_size[1]} input "
            "grid. Transcribe only that input grid."
        )
        instruction = _response_instruction(is_demo, response_format)
    content: list[dict[str, Any]] = [{"type": "text", "text": heading}]
    for camera, media_type, url in observation["media"]:
        content.extend(
            [
                {"type": "text", "text": f"Camera perspective: {camera}"},
                {"type": media_type, media_type: {"url": url}},
            ]
        )
    content.append({"type": "text", "text": instruction})
    system_prompt = (
        SCENE_THEN_STRUCTURED_SYSTEM_PROMPT
        if response_format == "scene_then_structured"
        else SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _response_instruction(is_demo: bool, response_format: ResponseFormat) -> str:
    if response_format != "scene_then_structured":
        return (
            "Return only one JSON object with input_grid and output_grid. Do not "
            "describe the robot, reserve cubes, or transformation."
            if is_demo
            else "Return only one JSON object with input_grid. Do not infer or "
            "return a test output grid."
        )

    grids = "input_grid and output_grid" if is_demo else "input_grid"
    return (
        "Return one JSON object with fields in this exact order: "
        f"scene_description, then {grids}. First write one integrated description "
        "of the whole scene in scene_description; do not create separate "
        "descriptions for each grid. In that one description: identify the left, "
        "middle, and right regions; inventory the visible cubes in the grids and "
        "reserve stacks; describe each grid row by row; state occlusion, blur, or "
        "perspective ambiguity; and end with confidence by region. Then use that "
        "visual inventory to transcribe the numeric grid arrays."
    )


def _description_messages(observation: dict[str, Any]) -> list[dict[str, Any]]:
    is_demo = observation["observation_type"] == "demonstration"
    input_rows, input_columns = _grid_size(observation["input_grid"])
    if is_demo:
        output_rows, output_columns = _grid_size(observation["output_grid"])
        heading = (
            "This is the final frame of one solved demonstration. The left input "
            f"area has a {input_rows}-row by {input_columns}-column logical lattice; "
            "the completed right output area has a "
            f"{output_rows}-row by {output_columns}-column logical lattice."
        )
        areas = "the input area, output area, and reserve stacks"
    else:
        heading = (
            "This is the initial frame of one test episode. The left input area "
            f"has a {input_rows}-row by {input_columns}-column logical lattice. "
            "There is no completed test output to infer."
        )
        areas = "the input area, currently visible output area, and reserve stacks"

    instruction = f"""Describe the image in enough visual detail that another person could determine whether you can see the cubes.

1. Give a brief overall scene description and identify the left, middle, and right regions.
2. For {areas}, inventory the visible cubes using ordinary color names.
3. For each grid area, go row by row and column by column. State which logical positions visibly contain a cube and its color, which appear empty, and which are uncertain or occluded.
4. Describe any robot-arm occlusion, blur, small-object ambiguity, or perspective ambiguity that prevents reliable reading.
5. End with a candid confidence assessment for each region.

Do not state an ARC rule, do not convert colors to numbers, and do not return JSON or grid arrays. Never silently guess an occluded cube."""
    content: list[dict[str, Any]] = [{"type": "text", "text": heading}]
    for camera, media_type, url in observation["media"]:
        content.extend(
            [
                {"type": "text", "text": f"Camera perspective: {camera}"},
                {"type": media_type, media_type: {"url": url}},
            ]
        )
    content.append({"type": "text", "text": instruction})
    return [
        {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _save_description_media(
    observation: dict[str, Any], media_dir: Path
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    safe_record_id = observation["record_id"].replace(":", "_")
    for camera, media_type, url in observation["media"]:
        if media_type != "image_url" or not url.startswith("data:image/png;base64,"):
            raise ValueError("scene-description media must be an in-memory PNG image")
        payload = base64.b64decode(url.split(",", 1)[1], validate=True)
        filename = f"{safe_record_id}_{camera}.png"
        path = media_dir / filename
        path.write_bytes(payload)
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
        manifest.append(
            {
                "camera": camera,
                "content_type": "image/png",
                "path": str(path),
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return manifest


def _demo_media(
    sample: ArcTaskSample,
    cameras: tuple[str, ...],
    demo_index: int,
    media_mode: MediaMode,
    fps: float,
    image_scale: int,
) -> list[tuple[str, str, str]]:
    media: list[tuple[str, str, str]] = []
    for camera in cameras:
        frames = sample.demonstrations[demo_index].frames_for_camera(camera)
        if frames is None or len(frames) == 0:
            raise ValueError(f"demo_{demo_index} contains no {camera} RGB frame")
        frames = upscale_rgb_frames(frames, factor=image_scale)
        if media_mode == "final_frame":
            media.append((camera, "image_url", frame_to_data_url(frames[-1])))
        elif media_mode == "sampled_frames":
            media.append((camera, "video_url", frames_to_video_data_url(frames)))
        else:
            media.append((camera, "video_url", frames_to_mp4_data_url(frames, fps=fps)))
    return media


def _test_media(
    sample: ArcTaskSample, cameras: tuple[str, ...], image_scale: int
) -> list[tuple[str, str, str]]:
    media: list[tuple[str, str, str]] = []
    for camera in cameras:
        frames = sample.test_case.frames_for_camera(camera)
        if frames is None or len(frames) == 0:
            raise ValueError(f"test contains no {camera} RGB frame")
        frames = upscale_rgb_frames(frames, factor=image_scale)
        media.append((camera, "image_url", frame_to_data_url(frames[0])))
    return media


def _task_indices(
    dataset: ArcTaskDataset,
    max_tasks: int | None,
    *,
    shuffle_tasks: bool = False,
    shuffle_seed: int = 0,
) -> dict[str, list[int]]:
    """Select whole tasks, preserving the order of test pairs within each task."""
    grouped: dict[str, list[int]] = {}
    for index, task_ref in enumerate(dataset.samples):
        grouped.setdefault(task_ref.task_id, []).append(index)
    task_ids = list(grouped)
    if shuffle_tasks:
        random.Random(shuffle_seed).shuffle(task_ids)
    return {task_id: grouped[task_id] for task_id in task_ids[:max_tasks]}


def _effective_seed(config: EvaluationConfig, seed: int | None) -> int:
    """Resolve one run seed shared by model generation and task selection."""
    if seed is not None:
        return seed
    return config.generation.seed if config.generation.seed is not None else 0


def _update_totals(
    totals: dict[str, int], metrics: dict[str, Any], observation: dict[str, Any]
) -> None:
    is_demo = observation["observation_type"] == "demonstration"
    totals["valid"] += metrics["valid_response"]
    totals["exact_grids"] += metrics["input_grid_exact_match"]
    if is_demo:
        totals["demo_inputs"] += metrics["input_grid_exact_match"]
        totals["demo_outputs"] += metrics["output_grid_exact_match"]
        totals["demo_pairs"] += metrics["both_grids_exact_match"]
        totals["exact_grids"] += metrics["output_grid_exact_match"]
    else:
        totals["test_inputs"] += metrics["input_grid_exact_match"]


def _count_expected(totals: dict[str, int], observation: dict[str, Any]) -> None:
    is_demo = observation["observation_type"] == "demonstration"
    totals["records"] += 1
    totals["expected_grids"] += 2 if is_demo else 1
    totals["demos" if is_demo else "tests"] += 1


def _update_all_grids_metrics(
    totals: dict[str, int],
    demo_context_exact: dict[str, bool],
    observation: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    """Combine shared demonstration results with each test-pair result."""
    task_id = observation["task_id"]
    if observation["observation_type"] == "demonstration":
        demo_context_exact[task_id] = demo_context_exact[task_id] and bool(
            metrics and metrics["both_grids_exact_match"]
        )
        return

    all_grids_exact = demo_context_exact[task_id] and bool(
        metrics and metrics["input_grid_exact_match"]
    )
    if metrics is not None:
        metrics["all_grids_exact"] = all_grids_exact
    totals["all_grids_exact"] += all_grids_exact


def _record(
    observation: dict[str, Any],
    result: dict[str, Any],
    media_mode: MediaMode,
    response_format: ResponseFormat,
    cameras: tuple[str, ...],
    fps: float,
    image_scale: int,
) -> dict[str, Any]:
    return {
        "record_id": observation["record_id"],
        "task_id": observation["task_id"],
        "episode_key": observation["episode_key"],
        "observation_type": observation["observation_type"],
        "diagnostic": "single_episode_grid_perception",
        "media_mode": media_mode,
        "response_format": response_format,
        "cameras": list(cameras),
        "video_fps": fps,
        "image_scale": image_scale,
        **result,
    }


def _processor_overrides(
    config: EvaluationConfig, *, fps: float | None
) -> dict[str, Any]:
    configured = config.generation.request_overrides.get("mm_processor_kwargs", {})
    if not isinstance(configured, dict):
        raise TypeError("mm_processor_kwargs must be a mapping")
    kwargs = {key: value for key, value in configured.items() if key != "fps"}
    if fps is not None:
        kwargs["fps"] = fps
    # ``mm_processor_kwargs`` is a vLLM/Qwen-specific request dialect. Do not
    # send an empty object to hosted OpenAI-compatible endpoints for an
    # image-only diagnostic: strict provider routing treats it as unsupported.
    return {"mm_processor_kwargs": kwargs} if kwargs else {}


def _validate_options(
    media_mode: str,
    response_format: str,
    max_tasks: int | None,
    image_scale: int,
) -> None:
    if media_mode not in ("final_frame", "sampled_frames", "sampled_video"):
        raise ValueError(f"unsupported media mode: {media_mode}")
    if response_format not in ("structured", "json", "scene_then_structured"):
        raise ValueError(f"unsupported response format: {response_format}")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive or null")
    if not isinstance(image_scale, int) or image_scale < 1:
        raise ValueError("image_scale must be a positive integer")


def _normalize_cameras(cameras: tuple[str, ...]) -> tuple[str, ...]:
    requested = set(cameras)
    unknown = requested.difference(SUPPORTED_CAMERAS)
    if unknown:
        raise ValueError(f"unsupported cameras: {', '.join(sorted(unknown))}")
    if not requested:
        raise ValueError("at least one camera is required")
    return tuple(camera for camera in SUPPORTED_CAMERAS if camera in requested)


def _grid_size(grid: list[list[int]] | None) -> tuple[int, int]:
    if grid is None:
        raise ValueError("canonical grid metadata is required")
    columns = len(grid[0]) if grid else 0
    if any(len(row) != columns for row in grid):
        raise ValueError("canonical ARC grids must be rectangular")
    return len(grid), columns


def _write_record(results_file: Any, record: dict[str, Any]) -> None:
    results_file.write(json.dumps(record) + "\n")
    results_file.flush()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
