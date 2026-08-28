import h5py
import numpy as np
import pytest
import torch

from src.datasets import (
    ArcDemonstration,
    ArcTaskDataset,
    ArcTaskSample,
    ArcTestCase,
    uniform_frame_indices,
)


def _add_grid_metadata(task_group, grids: dict[str, tuple[list, list]]) -> None:
    metadata = task_group.create_group("arc_grids")
    for episode_key, (input_grid, output_grid) in grids.items():
        pair = metadata.create_group(episode_key)
        pair.create_dataset("input_grid", data=input_grid)
        pair.create_dataset("output_grid", data=output_grid)


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "era_bench.hdf5"
    with h5py.File(path, "w") as dataset:
        data_group = dataset.create_group("data")
        data_group.create_group("demo_999")
        task_group = data_group.create_group("task_001")
        _add_grid_metadata(
            task_group,
            {
                "demo_0": ([[1, 0]], [[1]]),
                "demo_1": ([[2]], [[2, 0]]),
                "test": ([[3]], [[3]]),
            },
        )

        for episode_name, frame_count in (("demo_0", 2), ("demo_1", 3), ("test", 4)):
            episode_group = task_group.create_group(episode_name)
            obs_group = episode_group.create_group("obs")
            obs_group.create_dataset(
                "table_cam",
                data=np.zeros((frame_count, 2, 3, 3), dtype=np.uint8),
            )
            obs_group.create_dataset(
                "origin_top_cam",
                data=np.full((frame_count, 2, 3, 3), 9, dtype=np.uint8),
            )
            action_group = episode_group.create_group("high_level_actions")
            action_group.create_dataset(
                "action_timeline",
                data=np.zeros((1, frame_count - 1, 5), dtype=np.int64),
            )

    return path


def test_discovers_task_view_and_skips_source_demos(dataset_path) -> None:
    dataset = ArcTaskDataset(dataset_path)

    assert len(dataset) == 1
    assert dataset.samples[0].task_id == "task_001"
    assert dataset.samples[0].demo_keys == ["demo_0", "demo_1"]
    assert dataset.samples[0].test_key == "test"
    assert dataset.samples[0].file_path == dataset_path


def test_loads_task_with_real_output_types(dataset_path) -> None:
    sample = ArcTaskDataset(dataset_path, expose_test_targets=True)[0]

    assert isinstance(sample, ArcTaskSample)
    assert sample.task_id == "task_001"
    assert sample.test_key == "test"
    assert len(sample.demonstrations) == 2
    assert all(
        isinstance(demonstration, ArcDemonstration)
        for demonstration in sample.demonstrations
    )
    assert isinstance(sample.test_case, ArcTestCase)
    assert sample.demonstrations[0].table_cam_rgb_frames.shape == (2, 2, 3, 3)
    assert sample.test_case.table_cam_rgb_frames.shape == (4, 2, 3, 3)
    assert sample.demonstrations[0].input_grid == [[1, 0]]
    assert sample.demonstrations[1].output_grid == [[2, 0]]
    assert sample.test_case.input_grid == [[3]]
    assert sample.test_case.expected_output_grid == [[3]]
    for episode in (*sample.demonstrations, sample.test_case):
        assert isinstance(episode.table_cam_rgb_frames, torch.Tensor)
        assert episode.table_cam_rgb_frames.dtype == torch.uint8
        assert isinstance(episode.high_level_actions, torch.Tensor)
        assert episode.high_level_actions.dtype == torch.int64


def test_loads_requested_camera_fields_into_one_sample(dataset_path) -> None:
    sample = ArcTaskDataset(dataset_path, cameras=("table_cam", "origin_top_cam"))[0]

    demonstration = sample.demonstrations[0]
    assert demonstration.table_cam_rgb_frames.shape == (2, 2, 3, 3)
    assert demonstration.origin_top_cam_rgb_frames.shape == (2, 2, 3, 3)
    assert demonstration.origin_top_cam_rgb_frames[0, 0, 0, 0].item() == 9
    assert (
        demonstration.frames_for_camera("table_cam")
        is demonstration.table_cam_rgb_frames
    )
    assert not hasattr(demonstration, "rgb_frames")


def test_leaves_unrequested_camera_fields_unset(dataset_path) -> None:
    sample = ArcTaskDataset(dataset_path)[0]

    assert sample.demonstrations[0].origin_top_cam_rgb_frames is None


def test_creates_one_sample_per_numbered_test_with_shared_demos(tmp_path) -> None:
    path = tmp_path / "era_bench.hdf5"
    with h5py.File(path, "w") as dataset:
        data_group = dataset.create_group("data")
        task_group = data_group.create_group("task_001")
        _add_grid_metadata(
            task_group,
            {
                "demo_0": ([[1]], [[1]]),
                "demo_1": ([[2]], [[2]]),
                "test_0": ([[3]], [[3]]),
                "test_1": ([[4]], [[4]]),
            },
        )
        for episode_name, frame_count in (
            ("demo_0", 2),
            ("demo_1", 3),
            ("test_0", 4),
            ("test_1", 5),
        ):
            episode_group = task_group.create_group(episode_name)
            obs_group = episode_group.create_group("obs")
            obs_group.create_dataset(
                "table_cam",
                data=np.zeros((frame_count, 2, 3, 3), dtype=np.uint8),
            )
            action_group = episode_group.create_group("high_level_actions")
            action_group.create_dataset(
                "action_timeline",
                data=np.zeros((1, frame_count - 1, 5), dtype=np.int64),
            )

    dataset = ArcTaskDataset(path)

    assert len(dataset) == 2
    assert [sample.test_key for sample in dataset.samples] == ["test_0", "test_1"]
    assert all(sample.demo_keys == ["demo_0", "demo_1"] for sample in dataset.samples)
    assert dataset[0].test_case.table_cam_rgb_frames.shape[0] == 4
    assert dataset[1].test_case.table_cam_rgb_frames.shape[0] == 5
    assert dataset[0].test_key == "test_0"
    assert dataset[1].test_key == "test_1"


def test_selects_frames_from_hdf5_before_loading_them(dataset_path) -> None:
    with h5py.File(dataset_path, "r+") as dataset:
        frames = dataset["data/task_001/demo_1/obs/table_cam"]
        frames[...] = np.arange(len(frames), dtype=np.uint8)[:, None, None, None]

    dataset = ArcTaskDataset(
        dataset_path,
        frame_index_selector=uniform_frame_indices(stride=2),
    )

    selected = dataset[0].demonstrations[1].table_cam_rgb_frames

    assert selected.shape[0] == 2
    assert selected[:, 0, 0, 0].tolist() == [0, 2]
