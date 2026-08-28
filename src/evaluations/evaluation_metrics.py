"""Metrics for categorical high-level action predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from src.datasets import (
    HighLevelActionEvent,
    compress_action_timeline,
)
from src.vlm_actions import (
    LocationType,
    Phase,
    VLMAction,
    action_from_event,
    parse_action_response,
    parse_grid_response,
)

ColoredCell = tuple[int, int, int]
OccupiedCell = tuple[int, int]


@dataclass(frozen=True)
class ActionValidation:
    valid: bool
    errors: list[str]
    placements: list[ColoredCell]


@dataclass(frozen=True)
class SampleScore:
    exact_match: bool
    occupancy_exact_match: bool
    valid_action_sequence: bool
    colored_cell_precision: float
    colored_cell_recall: float
    colored_cell_f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    occupied_cell_precision: float
    occupied_cell_recall: float
    occupied_cell_f1: float
    occupied_true_positives: int
    occupied_false_positives: int
    occupied_false_negatives: int
    parse_error: str | None
    validation_errors: list[str]
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_action_sequence(actions: list[VLMAction]) -> ActionValidation:
    """Simulate output cells while recording soft action-sequence errors."""
    errors: list[str] = []
    output_cells: dict[tuple[int, int], int] = {}
    pending_color: int | None = None
    pending_output_coordinate: tuple[int, int] | None = None

    for index, action in enumerate(actions):
        if action.phase is Phase.GRASP:
            if not _has_valid_location_coordinates(action):
                errors.append(
                    f"action {index}: reserve actions require row/column -1; "
                    "output actions require coordinates in [0, 7]"
                )
            pending_output_coordinate = (
                (action.row, action.column)
                if _has_valid_location_coordinates(action)
                and action.location_type is LocationType.OUTPUT
                else None
            )
            pending_color = action.color_id
            continue

        if pending_color is None:
            errors.append(f"action {index}: place without a preceding grasp")
        elif pending_color != action.color_id:
            errors.append(
                f"action {index}: placed color {action.color_id} "
                f"does not match grasped color {pending_color}"
            )
        if not _has_valid_location_coordinates(action):
            errors.append(
                f"action {index}: reserve actions require row/column -1; "
                "output actions require coordinates in [0, 7]"
            )
        elif action.location_type is LocationType.OUTPUT:
            coordinate = (action.row, action.column)
            if pending_color is None:
                pass
            elif coordinate in output_cells and coordinate != pending_output_coordinate:
                errors.append(
                    f"action {index}: output coordinate {coordinate} is already occupied"
                )
            else:
                if pending_output_coordinate is not None:
                    output_cells.pop(pending_output_coordinate, None)
                output_cells[coordinate] = action.color_id
        elif pending_output_coordinate is not None:
            output_cells.pop(pending_output_coordinate, None)
        pending_color = None
        pending_output_coordinate = None

    placements = [
        (row, column, color_id) for (row, column), color_id in output_cells.items()
    ]
    return ActionValidation(valid=not errors, errors=errors, placements=placements)


def expected_actions_from_timeline(timeline: torch.Tensor) -> list[dict[str, Any]]:
    """Compress the trusted ground-truth timeline into named action objects."""
    return [
        action_from_event(event).model_dump(mode="json")
        for event in compress_action_timeline(timeline)
    ]


def placements_from_timeline(timeline: torch.Tensor) -> list[ColoredCell]:
    """Extract output placements directly from a trusted ground-truth timeline."""
    return _placements_from_events(compress_action_timeline(timeline))


def score_response(
    response_text: str,
    expected_timeline: torch.Tensor,
    *,
    task_id: str | None = None,
) -> SampleScore:
    """Score final-grid exact match, response validity, and cell overlap."""
    predicted, parse_error = parse_action_response(response_text)
    predicted_validation = (
        validate_action_sequence(predicted.actions)
        if predicted is not None
        else ActionValidation(False, [], [])
    )
    return score_cells(
        set(predicted_validation.placements),
        set(placements_from_timeline(expected_timeline)),
        parse_error=parse_error,
        valid_action_sequence=predicted_validation.valid,
        validation_errors=predicted_validation.errors,
        task_id=task_id,
    )


def score_grid_response(
    response_text: str,
    expected_grid: list[list[int]],
    *,
    task_id: str | None = None,
) -> SampleScore:
    """Score a direct output grid using the same final-cell metrics as actions."""
    predicted, parse_error = parse_grid_response(response_text)
    return score_grid(
        predicted.output_grid if predicted is not None else None,
        expected_grid,
        parse_error=parse_error,
        task_id=task_id,
    )


def score_grid(
    output_grid: list[list[int]] | None,
    expected_grid: list[list[int]],
    *,
    parse_error: str | None = None,
    task_id: str | None = None,
) -> SampleScore:
    """Score an already-parsed direct output grid."""
    return score_cells(
        _grid_cells(output_grid) if output_grid is not None else set(),
        _grid_cells(expected_grid),
        parse_error=parse_error,
        valid_action_sequence=parse_error is None,
        validation_errors=[],
        task_id=task_id,
    )


def score_cells(
    predicted_cells: set[ColoredCell],
    expected_cells: set[ColoredCell],
    *,
    parse_error: str | None,
    valid_action_sequence: bool,
    validation_errors: list[str],
    task_id: str | None = None,
) -> SampleScore:
    """Shared final-cell metric used by action and direct-grid outputs."""

    true_positives = len(predicted_cells & expected_cells)
    false_positives = len(predicted_cells - expected_cells)
    false_negatives = len(expected_cells - predicted_cells)
    precision, recall, f1 = _precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    predicted_occupied_cells = _occupied_cells(predicted_cells)
    expected_occupied_cells = _occupied_cells(expected_cells)
    occupied_true_positives = len(predicted_occupied_cells & expected_occupied_cells)
    occupied_false_positives = len(predicted_occupied_cells - expected_occupied_cells)
    occupied_false_negatives = len(expected_occupied_cells - predicted_occupied_cells)
    occupied_precision, occupied_recall, occupied_f1 = _precision_recall_f1(
        occupied_true_positives,
        occupied_false_positives,
        occupied_false_negatives,
    )
    valid = parse_error is None and valid_action_sequence
    exact_match = parse_error is None and _grid_from_cells(
        predicted_cells
    ) == _grid_from_cells(expected_cells)
    return SampleScore(
        exact_match=exact_match,
        occupancy_exact_match=(
            parse_error is None and predicted_occupied_cells == expected_occupied_cells
        ),
        valid_action_sequence=valid,
        colored_cell_precision=precision,
        colored_cell_recall=recall,
        colored_cell_f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        occupied_cell_precision=occupied_precision,
        occupied_cell_recall=occupied_recall,
        occupied_cell_f1=occupied_f1,
        occupied_true_positives=occupied_true_positives,
        occupied_false_positives=occupied_false_positives,
        occupied_false_negatives=occupied_false_negatives,
        parse_error=parse_error,
        validation_errors=validation_errors,
        task_id=task_id,
    )


def aggregate_scores(scores: list[SampleScore]) -> dict[str, Any]:
    """Aggregate sample metrics and exact-match outcomes by task."""
    if not scores:
        return {
            "sample_count": 0,
            "tasks_solved_count": 0,
            "tasks_solved_percentage": 0.0,
            "exact_match_rate": 0.0,
            "occupancy_exact_match_rate": 0.0,
            "valid_action_sequence_rate": 0.0,
            "colored_cell_precision": 0.0,
            "colored_cell_recall": 0.0,
            "colored_cell_f1": 0.0,
            "occupied_cell_precision": 0.0,
            "occupied_cell_recall": 0.0,
            "occupied_cell_f1": 0.0,
        }
    true_positives = sum(score.true_positives for score in scores)
    false_positives = sum(score.false_positives for score in scores)
    false_negatives = sum(score.false_negatives for score in scores)
    precision, recall, f1 = _precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    occupied_true_positives = sum(score.occupied_true_positives for score in scores)
    occupied_false_positives = sum(score.occupied_false_positives for score in scores)
    occupied_false_negatives = sum(score.occupied_false_negatives for score in scores)
    occupied_precision, occupied_recall, occupied_f1 = _precision_recall_f1(
        occupied_true_positives,
        occupied_false_positives,
        occupied_false_negatives,
    )
    task_outcomes: dict[str, bool] = {}
    for index, score in enumerate(scores):
        task_id = score.task_id if score.task_id is not None else f"__sample_{index}"
        task_outcomes[task_id] = task_outcomes.get(task_id, True) and score.exact_match
    tasks_solved_count = sum(task_outcomes.values())
    return {
        "sample_count": len(scores),
        "tasks_solved_count": tasks_solved_count,
        "tasks_solved_percentage": (100.0 * tasks_solved_count / len(task_outcomes)),
        "exact_match_rate": sum(score.exact_match for score in scores) / len(scores),
        "occupancy_exact_match_rate": (
            sum(score.occupancy_exact_match for score in scores) / len(scores)
        ),
        "valid_action_sequence_rate": (
            sum(score.valid_action_sequence for score in scores) / len(scores)
        ),
        "colored_cell_precision": precision,
        "colored_cell_recall": recall,
        "colored_cell_f1": f1,
        "occupied_cell_precision": occupied_precision,
        "occupied_cell_recall": occupied_recall,
        "occupied_cell_f1": occupied_f1,
    }


def _placements_from_events(events: list[HighLevelActionEvent]) -> list[ColoredCell]:
    output_cells: dict[tuple[int, int], int] = {}
    pending_output_coordinate: tuple[int, int] | None = None
    for event in events:
        phase, color_id, location_type, row, column = event.action
        if phase == 1:
            pending_output_coordinate = (row, column) if location_type == 1 else None
            continue
        if location_type == 1:
            coordinate = (row, column)
            if (
                coordinate not in output_cells
                or coordinate == pending_output_coordinate
            ):
                if pending_output_coordinate is not None:
                    output_cells.pop(pending_output_coordinate, None)
                output_cells[coordinate] = color_id
        elif pending_output_coordinate is not None:
            output_cells.pop(pending_output_coordinate, None)
        pending_output_coordinate = None
    return [(row, column, color_id) for (row, column), color_id in output_cells.items()]


def _has_valid_location_coordinates(action: VLMAction) -> bool:
    if action.location_type is LocationType.RESERVE:
        return action.row == -1 and action.column == -1
    return 0 <= action.row <= 7 and 0 <= action.column <= 7


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[float, float, float]:
    if true_positives == false_positives == false_negatives == 0:
        return 1.0, 1.0, 1.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _grid_from_cells(cells: set[ColoredCell]) -> dict[tuple[int, int], int]:
    return {(row, column): color_id for row, column, color_id in cells}


def _grid_cells(grid: list[list[int]]) -> set[ColoredCell]:
    return {
        (row, column, color_id)
        for row, values in enumerate(grid)
        for column, color_id in enumerate(values)
        if color_id != 0
    }


def _occupied_cells(cells: set[ColoredCell]) -> set[OccupiedCell]:
    """Drop cube color to score only whether each output cell is occupied."""
    return {(row, column) for row, column, _ in cells}
