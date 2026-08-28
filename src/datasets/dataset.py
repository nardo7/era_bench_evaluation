from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

TEST_KEY_PATTERN = re.compile(r"test_(\d+)")
FrameIndexSelector = Callable[[int], Sequence[int]]
SUPPORTED_CAMERAS = ("table_cam", "origin_top_cam")


@dataclass(frozen=True)
class TaskRef:
    file_path: Path
    demo_keys: list[str]
    test_key: str
    task_id: str


@dataclass
class ArcDemonstration:
    """One solved example belonging to an ARC task."""

    input_grid: list[list[int]] | None = None
    output_grid: list[list[int]] | None = None
    table_cam_rgb_frames: torch.Tensor | None = None
    origin_top_cam_rgb_frames: torch.Tensor | None = None

    def __len__(self):
        frames = self.table_cam_rgb_frames
        if frames is None:
            frames = self.origin_top_cam_rgb_frames
        return frames.shape[0] if frames is not None else 0

    @property
    def num_steps(self) -> int:
        """Return the number of high-level steps in the demonstration."""
        return self.__len__()

    # Embodied representation of the solved example.
    high_level_actions: torch.Tensor | None = None

    def frames_for_camera(self, camera: str) -> torch.Tensor | None:
        return getattr(self, f"{camera}_rgb_frames")


@dataclass
class ArcTestCase:
    """One unsolved query for the same ARC task."""

    input_grid: list[list[int]] | None = None

    expected_output_grid: list[list[int]] | None = None
    table_cam_rgb_frames: torch.Tensor | None = None
    origin_top_cam_rgb_frames: torch.Tensor | None = None
    # Embodied representation of the solved example.
    high_level_actions: torch.Tensor | None = None

    def __len__(self):
        frames = self.table_cam_rgb_frames
        if frames is None:
            frames = self.origin_top_cam_rgb_frames
        return frames.shape[0] if frames is not None else 0

    @property
    def num_steps(self) -> int:
        """Return the number of high-level steps in the demonstration."""
        return self.__len__()

    def frames_for_camera(self, camera: str) -> torch.Tensor | None:
        return getattr(self, f"{camera}_rgb_frames")


@dataclass
class ArcTaskSample:
    """
    Complete few-shot context for solving one test case.

    This is the atomic item returned by the dataset.
    """

    task_id: str
    demonstrations: Sequence[ArcDemonstration]
    test_case: ArcTestCase
    test_key: str = "test"
    ground_truth_rule: str | None = None


class ArcTaskDataset(Dataset[ArcTaskSample]):
    def __init__(
        self,
        file_paths: list[Path] | Path,
        *,
        expose_test_targets: bool = False,
        frame_index_selector: FrameIndexSelector | None = None,
        cameras: Sequence[str] = ("table_cam",),
    ) -> None:
        self.cameras = tuple(cameras)
        unknown = set(self.cameras).difference(SUPPORTED_CAMERAS)
        if not self.cameras or unknown:
            raise ValueError(
                "cameras must contain one or more supported cameras: "
                + ", ".join(SUPPORTED_CAMERAS)
            )
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must not contain duplicates")
        self.source = file_paths
        self.expose_test_targets = expose_test_targets
        self.frame_index_selector = frame_index_selector
        # One entry per (task, test case), not per physical episode.
        self.samples: list[TaskRef] = self._discover_task_samples(file_paths)

    def __len__(self) -> int:
        l = len(self.samples)
        return l

    def __getitem__(self, index: int) -> ArcTaskSample:
        task_ref = self.samples[index]

        return self._load_task(task_ref)

    def _discover_task_samples(self, file_paths: list[Path] | Path) -> list[TaskRef]:
        """
        Discover complete ARC task units.

        Each returned reference identifies:
        - one task ID;
        - all training/demo pairs belonging to that task;
        - one test pair to solve.

        If a task has multiple test pairs, create one dataset item per test pair,
        with the same demonstration context.
        """
        file_paths = [file_paths] if isinstance(file_paths, Path) else file_paths
        task_refs: list[TaskRef] = []
        for file_path in file_paths:
            with h5py.File(file_path, "r") as dataset:
                for task_id in dataset["data"]:
                    if task_id.startswith("demo_"):
                        # Ignore source episode groups created by Isaac Lab.
                        continue

                    task_group = dataset["data"][task_id]
                    if not isinstance(task_group, h5py.Group):
                        continue

                    demo_keys = sorted(
                        (key for key in task_group if key.startswith("demo_")),
                        key=_episode_key_sort_key,
                    )
                    test_keys = sorted(
                        (key for key in task_group if _is_test_key(key)),
                        key=_episode_key_sort_key,
                    )
                    if not test_keys:
                        raise ValueError(
                            f"Task view /data/{task_id} contains no test or test_<n> group"
                        )

                    for test_key in test_keys:
                        task_refs.append(
                            TaskRef(
                                task_id=task_id,
                                demo_keys=demo_keys,
                                test_key=test_key,
                                file_path=file_path,
                            )
                        )
        print(
            f"Discovered {len(task_refs)} task views across {len(file_paths)} dataset files"
        )
        return task_refs

    def _load_task(self, task_ref: TaskRef) -> ArcTaskSample:
        with h5py.File(task_ref.file_path, "r") as dataset:
            task_group = dataset["data"][task_ref.task_id]
            grid_metadata = self._load_task_metadata(task_group)
            ground_truth_rule = self._load_ground_truth_rule(task_group)
            demos: list[ArcDemonstration] = []
            for demo_key in task_ref.demo_keys:
                demo_group = task_group[demo_key]
                input_grid, output_grid = self._load_grid_pair(grid_metadata, demo_key)

                demos.append(
                    ArcDemonstration(
                        input_grid=input_grid,
                        output_grid=output_grid,
                        **self._load_camera_frames(demo_group),
                        high_level_actions=torch.from_numpy(
                            demo_group["high_level_actions/action_timeline"][()]
                        ),
                    )
                )
            test_group = dataset["data"][task_ref.task_id][task_ref.test_key]
            test_input_grid, test_output_grid = self._load_grid_pair(
                grid_metadata, task_ref.test_key
            )

            test = ArcTestCase(
                input_grid=test_input_grid,
                expected_output_grid=(
                    test_output_grid if self.expose_test_targets else None
                ),
                **self._load_camera_frames(test_group),
                high_level_actions=torch.from_numpy(
                    test_group["high_level_actions/action_timeline"][()]
                ),
            )

            return ArcTaskSample(
                task_id=task_ref.task_id,
                demonstrations=demos,
                test_case=test,
                test_key=task_ref.test_key,
                ground_truth_rule=ground_truth_rule,
            )

    @staticmethod
    def _load_ground_truth_rule(task_group: h5py.Group) -> str | None:
        """Load the task's inferred rule when rule metadata is available."""
        try:
            rule_group = task_group["rule"]
            rule_data = rule_group["data"]
        except KeyError:
            return None
        if not isinstance(rule_data, h5py.Dataset):
            raise TypeError(f"Expected {rule_data.name} to be a dataset")

        payload = rule_data[()]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise TypeError(f"Expected {rule_data.name} to contain a UTF-8 JSON string")
        rule = json.loads(payload).get("inferred_rule")
        if not isinstance(rule, str):
            raise TypeError(
                f"Expected {rule_data.name} to contain a string inferred_rule"
            )
        return rule

    def _load_camera_frames(self, episode_group: h5py.Group) -> dict[str, torch.Tensor]:
        return {
            f"{camera}_rgb_frames": self._load_rgb_frames(episode_group, camera)
            for camera in self.cameras
        }

    def _load_rgb_frames(self, episode_group: h5py.Group, camera: str) -> torch.Tensor:
        frames_key = f"obs/{camera}"
        frames = episode_group[frames_key]
        if not isinstance(frames, h5py.Dataset):
            raise TypeError(
                f"Expected {episode_group.name}/{frames_key} to be a dataset"
            )
        if self.frame_index_selector is None:
            return torch.from_numpy(frames[()])

        indices = _validate_frame_indices(
            self.frame_index_selector(len(frames)), frame_count=len(frames)
        )
        return torch.from_numpy(frames[indices])

    @staticmethod
    def _load_task_metadata(task_group: h5py.Group) -> h5py.Group:
        try:
            metadata = task_group["arc_grids"]
        except KeyError as error:
            raise ValueError(
                f"ARC grid metadata is missing from {task_group.name}; "
                "regenerate task views with scripts/create_task_views.py --overwrite"
            ) from error
        if not isinstance(metadata, h5py.Group):
            raise TypeError(f"Expected {metadata.name} to be a group")
        return metadata

    @staticmethod
    def _load_grid_pair(
        grid_metadata: h5py.Group, episode_key: str
    ) -> tuple[list[list[int]], list[list[int]]]:
        try:
            pair = grid_metadata[episode_key]
            if not isinstance(pair, h5py.Group):
                raise TypeError(f"Expected {pair.name} to be a group")
            input_grid = pair["input_grid"]
            output_grid = pair["output_grid"]
        except KeyError as error:
            raise ValueError(
                f"Missing ARC grids for {grid_metadata.name}/{episode_key}"
            ) from error
        if not isinstance(input_grid, h5py.Dataset) or input_grid.ndim != 2:
            raise TypeError(f"Expected {input_grid.name} to be a 2D dataset")
        if not isinstance(output_grid, h5py.Dataset) or output_grid.ndim != 2:
            raise TypeError(f"Expected {output_grid.name} to be a 2D dataset")
        return (
            input_grid[()].astype(int).tolist(),
            output_grid[()].astype(int).tolist(),
        )


def _is_test_key(key: str) -> bool:
    return key == "test" or TEST_KEY_PATTERN.fullmatch(key) is not None


def _episode_key_sort_key(key: str) -> tuple[int, int | str]:
    """Sort numbered demo/test groups numerically and retain legacy ``test``."""
    if key == "test":
        return (0, 0)
    suffix = key.rsplit("_", maxsplit=1)[-1]
    return (1, int(suffix))


def uniform_frame_indices(stride: int) -> FrameIndexSelector:
    """Select every ``stride``-th frame and always retain the final frame."""
    if stride < 1:
        raise ValueError("stride must be at least 1")

    def selector(frame_count: int) -> list[int]:
        if frame_count == 0:
            return []
        indices = list(range(0, frame_count, stride))
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
        return indices

    return selector


def _validate_frame_indices(indices: Sequence[int], *, frame_count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.intp)
    if values.ndim != 1:
        raise ValueError("frame indices must be one-dimensional")
    if values.size and (
        values[0] < 0 or values[-1] >= frame_count or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("frame indices must be sorted, unique, and in bounds")
    return values
