"""Utilities for working with recorded ARC high-level actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ActionVector = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class HighLevelActionEvent:
    """One contiguous run of a recorded high-level action."""

    action: ActionVector
    start_timestep: int
    end_timestep: int


def compress_action_timeline(
    timeline: torch.Tensor,
    *,
    include_no_action: bool = False,
) -> list[HighLevelActionEvent]:
    """Run-length encode a ``[T, 5]`` or ``[1, T, 5]`` action timeline.

    By default, ``phase == 0`` runs are omitted because they do not contribute
    to the output construction plan.
    """
    timeline = _normalise_timeline(timeline)
    if len(timeline) == 0:
        return []

    event_starts = torch.ones(len(timeline), dtype=torch.bool, device=timeline.device)
    event_starts[1:] = torch.any(timeline[1:] != timeline[:-1], dim=1)
    event_indices = torch.nonzero(event_starts, as_tuple=False).squeeze(1)
    event_actions = timeline[event_indices]
    event_ends = torch.cat(
        (event_indices[1:] - 1, event_indices.new_tensor([len(timeline) - 1]))
    )

    if not include_no_action:
        non_no_action = event_actions[:, 0] != 0
        event_indices = event_indices[non_no_action]
        event_actions = event_actions[non_no_action]
        event_ends = event_ends[non_no_action]

    return [
        HighLevelActionEvent(
            action=tuple(int(value) for value in action),
            start_timestep=int(start_timestep),
            end_timestep=int(end_timestep),
        )
        for action, start_timestep, end_timestep in zip(
            event_actions.tolist(),
            event_indices.tolist(),
            event_ends.tolist(),
            strict=True,
        )
    ]


def _normalise_timeline(timeline: torch.Tensor) -> torch.Tensor:
    if timeline.ndim == 3:
        if timeline.shape[0] != 1:
            raise ValueError(
                "a three-dimensional action timeline must have batch size 1"
            )
        timeline = timeline[0]
    if timeline.ndim != 2 or timeline.shape[1] != 5:
        raise ValueError("action timeline must have shape [T, 5] or [1, T, 5]")
    return timeline
