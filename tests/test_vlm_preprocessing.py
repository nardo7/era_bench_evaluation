import torch
from src.datasets.dataset import (
    ArcDemonstration,
    ArcTaskSample,
    ArcTestCase,
)
from src.vlm_preprocessing import describe_action, prepare_vlm_task


def test_prepares_target_safe_vlm_input() -> None:
    demonstration = ArcDemonstration(
        input_grid=[[1, 2], [3, 4], [5, 6]],
        output_grid=[[1, 0, 2]],
        table_cam_rgb_frames=torch.zeros((3, 2, 2, 3), dtype=torch.uint8),
        high_level_actions=torch.tensor(
            [
                [
                    [1, 2, 0, -1, -1],
                    [1, 2, 0, -1, -1],
                    [2, 2, 1, 0, 1],
                ]
            ]
        ),
    )
    test_frames = torch.stack(
        [
            torch.full((2, 2, 3), 7, dtype=torch.uint8),
            torch.full((2, 2, 3), 9, dtype=torch.uint8),
        ]
    )
    sample = ArcTaskSample(
        task_id="task_001",
        demonstrations=[demonstration],
        test_case=ArcTestCase(
            input_grid=[[1], [2], [3], [4]],
            table_cam_rgb_frames=test_frames,
            high_level_actions=torch.tensor([[[2, 9, 1, 7, 7]]]),
        ),
    )

    prepared = prepare_vlm_task(sample)

    assert prepared.test_first_frame.tolist() == test_frames[0].tolist()
    assert prepared.test_key == "test"
    assert prepared.demonstrations[0].input_grid == [[1, 2], [3, 4], [5, 6]]
    assert prepared.demonstrations[0].output_grid == [[1, 0, 2]]
    assert prepared.test_input_grid == [[1], [2], [3], [4]]
    assert prepared.demonstrations[0].input_grid_size == (3, 2)
    assert prepared.demonstrations[0].output_grid_size == (1, 3)
    assert prepared.test_input_grid_size == (4, 1)
    assert [event.action for event in prepared.demonstrations[0].actions] == [
        (1, 2, 0, -1, -1),
        (2, 2, 1, 0, 1),
    ]
    assert describe_action(prepared.demonstrations[0].actions[0]) == (
        "grasp a red cube from the reserve"
    )
    assert describe_action(prepared.demonstrations[0].actions[1]) == (
        "place a red cube at output row 0, column 1"
    )
