"""Convert raw ARC task samples into the model-facing VLM representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from src.datasets import ArcTaskSample
from src.datasets.actions import HighLevelActionEvent, compress_action_timeline

COLOR_NAMES = {
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "teal",
    9: "brown",
}


@dataclass(frozen=True)
class VLMDemonstration:
    """A solved embodied example for a VLM prompt."""

    rgb_frames: torch.Tensor
    actions: list[HighLevelActionEvent]
    input_grid: list[list[int]] | None = None
    output_grid: list[list[int]] | None = None
    input_grid_size: tuple[int, int] | None = None
    output_grid_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class VLMTaskInput:
    """The target-safe model input for one ARC task."""

    task_id: str
    test_key: str
    demonstrations: list[VLMDemonstration]
    test_first_frame: torch.Tensor
    test_input_grid: list[list[int]] | None = None
    test_input_grid_size: tuple[int, int] | None = None


def prepare_vlm_task(
    sample: ArcTaskSample, *, camera: str = "table_cam"
) -> VLMTaskInput:
    """Prepare one task without exposing test actions or later test frames."""
    test_frames = sample.test_case.frames_for_camera(camera)
    if test_frames is None or len(test_frames) == 0:
        raise ValueError("the test case must contain at least one RGB frame")

    demonstrations: list[VLMDemonstration] = []
    for demonstration in sample.demonstrations:
        frames = demonstration.frames_for_camera(camera)
        if frames is None:
            raise ValueError("each demonstration must contain RGB frames")
        if demonstration.high_level_actions is None:
            raise ValueError("each demonstration must contain high-level actions")
        demonstrations.append(
            VLMDemonstration(
                rgb_frames=frames,
                actions=compress_action_timeline(demonstration.high_level_actions),
                input_grid=demonstration.input_grid,
                output_grid=demonstration.output_grid,
                input_grid_size=_grid_size(demonstration.input_grid),
                output_grid_size=_grid_size(demonstration.output_grid),
            )
        )

    return VLMTaskInput(
        task_id=sample.task_id,
        test_key=sample.test_key,
        demonstrations=demonstrations,
        test_first_frame=test_frames[0],
        test_input_grid=sample.test_case.input_grid,
        test_input_grid_size=_grid_size(sample.test_case.input_grid),
    )


def _grid_size(grid: list[list[int]] | None) -> tuple[int, int] | None:
    if grid is None:
        return None
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    if any(len(row) != columns for row in grid):
        raise ValueError("ARC grids must be rectangular")
    return rows, columns


def describe_action(event: HighLevelActionEvent) -> str:
    """Render one compact action event for the VLM-facing representation."""
    phase, color_id, location_type, row, col = event.action
    color = COLOR_NAMES.get(color_id, f"color {color_id}")
    if phase == 1:
        if location_type == 0:
            return f"grasp a {color} cube from the reserve"
        return f"grasp a {color} cube from output row {row}, column {col}"
    if phase == 2:
        if location_type == 0:
            return f"return a {color} cube to the reserve"
        return f"place a {color} cube at output row {row}, column {col}"
    return "no action"
