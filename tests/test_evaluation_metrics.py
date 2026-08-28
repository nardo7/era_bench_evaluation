import json

import pytest
import torch
from src.evaluations.evaluation_metrics import (
    aggregate_scores,
    expected_actions_from_timeline,
    parse_action_response,
    placements_from_timeline,
    score_response,
    validate_action_sequence,
)


def _action(
    phase: str,
    color_id: int,
    location_type: str,
    row: int,
    column: int,
) -> dict:
    return {
        "phase": phase,
        "color_id": color_id,
        "location_type": location_type,
        "row": row,
        "column": column,
    }


def _grasp(color_id: int) -> dict:
    return _action("grasp", color_id, "reserve", -1, -1)


def _place(color_id: int, row: int, column: int) -> dict:
    return _action("place", color_id, "output", row, column)


def _response(actions: list[dict]) -> str:
    return json.dumps({"rule_summary": "test transformation", "actions": actions})


def _target_timeline() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [1, 2, 0, -1, -1],
                [1, 2, 0, -1, -1],
                [2, 2, 1, 0, 1],
                [1, 1, 0, -1, -1],
                [2, 1, 1, 1, 0],
            ]
        ]
    )


def test_validates_both_grasp_and_place_events() -> None:
    prediction, error = parse_action_response(_response([_grasp(2), _place(2, 0, 1)]))

    assert error is None
    assert prediction is not None
    validation = validate_action_sequence(prediction.actions)
    assert validation.valid
    assert validation.errors == []
    assert validation.placements == [(0, 1, 2)]


@pytest.mark.parametrize(
    ("actions", "expected_error"),
    [
        ([_place(2, 0, 1)], "place without a preceding grasp"),
        ([_grasp(1), _place(2, 0, 1)], "does not match grasped color"),
        (
            [_grasp(2), _action("place", 2, "output", 9, -1)],
            "output actions require coordinates in [0, 7]",
        ),
    ],
)
def test_rejects_each_invalid_action_sequence(
    actions: list[dict], expected_error: str
) -> None:
    score = score_response(
        _response(actions), torch.empty((1, 0, 5), dtype=torch.int64)
    )

    assert not score.valid_action_sequence
    assert any(expected_error in error for error in score.validation_errors)


def test_exact_match_is_independent_of_soft_sequence_validity() -> None:
    response = _response([_grasp(1), _place(2, 0, 1)])

    score = score_response(
        response,
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 1]]]),
    )

    assert score.exact_match
    assert not score.valid_action_sequence
    assert "does not match grasped color" in score.validation_errors[0]


def test_consecutive_or_trailing_grasps_are_not_hard_failures() -> None:
    response = _response(
        [
            _grasp(1),
            _grasp(2),
            _place(2, 0, 1),
            _grasp(3),
        ]
    )

    score = score_response(
        response,
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 1]]]),
    )

    assert score.valid_action_sequence
    assert score.exact_match


def test_exact_match_compares_final_grid_not_action_order() -> None:
    response = _response([_grasp(1), _place(1, 1, 0), _grasp(2), _place(2, 0, 1)])

    score = score_response(response, _target_timeline())

    assert score.exact_match
    assert score.valid_action_sequence
    assert score.colored_cell_f1 == 1.0


def test_invalid_plan_still_reports_colored_cell_overlap() -> None:
    response = _response([_grasp(3), _place(2, 0, 1), _grasp(3), _place(3, 1, 1)])

    score = score_response(response, _target_timeline())

    assert not score.exact_match
    assert not score.valid_action_sequence
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.colored_cell_f1 == pytest.approx(0.5)


def test_occupancy_metrics_ignore_cube_color_but_not_cell_position() -> None:
    score = score_response(
        _response([_grasp(3), _place(3, 0, 1)]),
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 1]]]),
    )

    assert not score.exact_match
    assert score.colored_cell_f1 == 0.0
    assert score.occupancy_exact_match
    assert score.occupied_true_positives == 1
    assert score.occupied_false_positives == 0
    assert score.occupied_false_negatives == 0
    assert score.occupied_cell_precision == 1.0
    assert score.occupied_cell_recall == 1.0
    assert score.occupied_cell_f1 == 1.0


def test_occupancy_metrics_count_misplaced_cells_as_extra_and_missing() -> None:
    score = score_response(
        _response([_grasp(2), _place(2, 0, 0)]),
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 1]]]),
    )

    assert not score.occupancy_exact_match
    assert score.occupied_true_positives == 0
    assert score.occupied_false_positives == 1
    assert score.occupied_false_negatives == 1
    assert score.occupied_cell_precision == 0.0
    assert score.occupied_cell_recall == 0.0
    assert score.occupied_cell_f1 == 0.0


def test_malformed_response_cannot_occupancy_match_an_empty_target() -> None:
    score = score_response("not JSON", torch.empty((1, 0, 5), dtype=torch.int64))

    assert score.parse_error is not None
    assert not score.occupancy_exact_match


def test_ground_truth_compression_retains_both_event_types() -> None:
    expected = expected_actions_from_timeline(_target_timeline())

    assert [action["phase"] for action in expected] == [
        "grasp",
        "place",
        "grasp",
        "place",
    ]
    assert placements_from_timeline(_target_timeline()) == [(0, 1, 2), (1, 0, 1)]


def test_ground_truth_placement_extraction_applies_output_removals() -> None:
    timeline = torch.tensor(
        [
            [
                [1, 2, 0, -1, -1],
                [2, 2, 1, 0, 0],
                [1, 2, 1, 0, 0],
                [2, 2, 1, 0, 1],
            ]
        ]
    )

    assert placements_from_timeline(timeline) == [(0, 1, 2)]


def test_correction_removes_then_replaces_an_output_cube() -> None:
    response = _response(
        [
            _grasp(2),
            _place(2, 0, 0),
            _action("grasp", 2, "output", 0, 0),
            _action("place", 2, "reserve", -1, -1),
            _grasp(3),
            _place(3, 0, 0),
        ]
    )

    score = score_response(
        response,
        torch.tensor([[[1, 3, 0, -1, -1], [2, 3, 1, 0, 0]]]),
    )

    assert score.valid_action_sequence
    assert score.exact_match


def test_an_output_grasp_does_not_remove_a_cube_until_its_place_completes() -> None:
    response = _response(
        [_grasp(2), _place(2, 0, 0), _action("grasp", 2, "output", 0, 0)]
    )

    score = score_response(
        response,
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 0]]]),
    )

    assert score.valid_action_sequence
    assert score.exact_match


def test_occupied_output_place_does_not_replace_the_existing_cube() -> None:
    response = _response([_grasp(2), _place(2, 0, 0), _grasp(3), _place(3, 0, 0)])

    score = score_response(
        response,
        torch.tensor([[[1, 2, 0, -1, -1], [2, 2, 1, 0, 0]]]),
    )

    assert score.exact_match
    assert not score.valid_action_sequence
    assert "already occupied" in score.validation_errors[0]


def test_empty_prediction_exactly_matches_empty_target() -> None:
    score = score_response(_response([]), torch.empty((1, 0, 5), dtype=torch.int64))

    assert score.exact_match
    assert score.valid_action_sequence
    assert score.colored_cell_f1 == 1.0


def test_pydantic_rejects_malformed_or_numeric_action_names() -> None:
    missing_fields, missing_error = parse_action_response(
        '{"actions": [{"phase": "grasp"}]}'
    )
    numeric_phase, numeric_error = parse_action_response(
        _response([_action(1, 2, "reserve", -1, -1)])
    )

    assert missing_fields is None
    assert missing_error is not None and "color_id" in missing_error
    assert numeric_phase is None
    assert numeric_error is not None and "phase" in numeric_error


def test_pydantic_allows_an_omitted_summary_but_rejects_an_empty_one() -> None:
    missing_summary, missing_error = parse_action_response('{"actions": []}')
    empty_summary, empty_error = parse_action_response(
        '{"rule_summary": "", "actions": []}'
    )

    assert missing_summary is not None
    assert missing_error is None
    assert empty_summary is None
    assert empty_error is not None and "rule_summary" in empty_error


def test_aggregate_scores_uses_micro_colored_cell_metrics() -> None:
    scores = [
        score_response(_response([]), torch.empty((1, 0, 5), dtype=torch.int64)),
        score_response(_response([]), _target_timeline()),
    ]

    summary = aggregate_scores(scores)

    assert summary["sample_count"] == 2
    assert summary["tasks_solved_count"] == 1
    assert summary["tasks_solved_percentage"] == 50.0
    assert summary["exact_match_rate"] == 0.5
    assert summary["valid_action_sequence_rate"] == 1.0
    assert summary["colored_cell_precision"] == 0.0
    assert summary["colored_cell_recall"] == 0.0
    assert summary["colored_cell_f1"] == 0.0
    assert summary["occupancy_exact_match_rate"] == 0.5
    assert summary["occupied_cell_precision"] == 0.0
    assert summary["occupied_cell_recall"] == 0.0
    assert summary["occupied_cell_f1"] == 0.0


def test_aggregate_scores_counts_multi_test_tasks_once() -> None:
    solved = score_response(
        _response([]), torch.empty((1, 0, 5), dtype=torch.int64), task_id="task_a"
    )
    unsolved = score_response(_response([]), _target_timeline(), task_id="task_a")
    second_solved = score_response(
        _response([]), torch.empty((1, 0, 5), dtype=torch.int64), task_id="task_b"
    )

    summary = aggregate_scores([solved, unsolved, second_solved])

    assert summary["sample_count"] == 3
    assert summary["tasks_solved_count"] == 1
    assert summary["tasks_solved_percentage"] == 50.0
