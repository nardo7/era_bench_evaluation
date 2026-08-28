import torch
from src.datasets import compress_action_timeline


def test_compresses_action_runs_and_omits_no_action() -> None:
    timeline = torch.tensor(
        [
            [
                [0, -1, -1, -1, -1],
                [1, 2, 0, -1, -1],
                [1, 2, 0, -1, -1],
                [2, 2, 1, 0, 1],
                [2, 2, 1, 0, 1],
                [0, -1, -1, -1, -1],
            ]
        ]
    )

    events = compress_action_timeline(timeline)

    assert [event.action for event in events] == [
        (1, 2, 0, -1, -1),
        (2, 2, 1, 0, 1),
    ]
    assert [(event.start_timestep, event.end_timestep) for event in events] == [
        (1, 2),
        (3, 4),
    ]


def test_can_retain_no_action_runs() -> None:
    timeline = torch.tensor([[0, -1, -1, -1, -1], [0, -1, -1, -1, -1]])

    events = compress_action_timeline(timeline, include_no_action=True)

    assert len(events) == 1
    assert events[0].action == (0, -1, -1, -1, -1)
    assert (events[0].start_timestep, events[0].end_timestep) == (0, 1)
