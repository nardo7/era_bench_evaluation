import h5py
import numpy as np
import pytest
from arc_utils.models import Split, Task

from scripts.create_task_views import create_task_views


def _add_source_episode(data_group, name: str, task_id: str, task_pair: int) -> None:
    episode_group = data_group.create_group(name)
    task_info = episode_group.create_group("task_info")
    task_info.create_dataset(
        "task_id", data=np.array([[ord(char) for char in task_id]], dtype=np.uint8)
    )
    task_info.create_dataset("task_pair", data=np.array([[task_pair]], dtype=np.int64))


def test_uses_final_arc_test_count_to_create_numbered_test_views(tmp_path) -> None:
    dataset_path = tmp_path / "era_bench.hdf5"
    task = Task(
        task_id="task_001",
        train=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[0]], "output": [[0]]},
        ],
        test=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[0]], "output": [[0]]},
        ],
        split=Split.TRAIN,
    )

    with h5py.File(dataset_path, "w") as dataset:
        data_group = dataset.create_group("data")
        for pair_index in range(4):
            _add_source_episode(
                data_group, f"demo_{pair_index}", "task_001", pair_index
            )
        rules_group = dataset.create_group("rules")
        rules_group.create_group("task_001")

    assert create_task_views(dataset_path, arc_tasks=[task]) == 1

    with h5py.File(dataset_path, "r") as dataset:
        task_group = dataset["data/task_001"]
        assert sorted(task_group) == [
            "arc_grids",
            "demo_0",
            "demo_1",
            "rule",
            "test_0",
            "test_1",
        ]
        assert int(task_group["test_0/task_info/task_pair"][0, 0]) == 2
        assert int(task_group["test_1/task_info/task_pair"][0, 0]) == 3
        assert task_group["arc_grids/demo_0/input_grid"][()].tolist() == [[0]]
        assert task_group["arc_grids/test_1/output_grid"][()].tolist() == [[0]]


def test_keeps_earliest_source_episode_for_duplicate_task_pair(tmp_path) -> None:
    dataset_path = tmp_path / "era_bench.hdf5"
    task = Task(
        task_id="task_001",
        train=[
            {"input": [[0]], "output": [[0]]},
            {"input": [[0]], "output": [[0]]},
            {"input": [[0]], "output": [[0]]},
        ],
        test=[{"input": [[0]], "output": [[0]]}],
        split=Split.TRAIN,
    )

    with h5py.File(dataset_path, "w") as dataset:
        data_group = dataset.create_group("data")
        for source_index, pair_index in enumerate([0, 1, 2, 3, 3, 3, 3]):
            _add_source_episode(
                data_group,
                f"demo_{source_index}",
                "task_001",
                pair_index,
            )
        rules_group = dataset.create_group("rules")
        rules_group.create_group("task_001")

    with pytest.warns(
        UserWarning,
        match=r"Task task_001 has 3 duplicate source episode\(s\).*keeping the earliest",
    ):
        assert create_task_views(dataset_path, arc_tasks=[task]) == 1

    with h5py.File(dataset_path, "r") as dataset:
        assert dataset["data/task_001/test_0"].id == dataset["data/demo_3"].id
