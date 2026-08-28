"""Create task-centric hard-link views in an ERA Bench HDF5 dataset."""

import argparse
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
from arc_utils import Task, load_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "era_bench.hdf5"
SOURCE_DEMO_PATTERN = re.compile(r"demo_\d+")
VIEW_MARKER = "era_bench_task_view_v1"


@dataclass(frozen=True)
class TaskPair:
    task_pair: int
    source_demo: str


def decode_task_id(dataset: h5py.Dataset) -> str:
    """Decode task IDs stored as a one-row uint8 ASCII array."""
    values = dataset[()].reshape(-1)
    try:
        return bytes(values).decode("ascii")
    except (TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid task ID encoding at {dataset.name}") from error


def decode_task_pair(dataset: h5py.Dataset) -> int:
    """Decode task-pair indices stored as one-element integer arrays."""
    values = dataset[()].reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"Expected one task-pair value at {dataset.name}, found {values.size}"
        )
    return int(values[0])


def collect_task_pairs(data_group: h5py.Group) -> dict[str, list[TaskPair]]:
    task_pairs: dict[str, list[TaskPair]] = defaultdict(list)

    for source_demo in sorted(data_group):
        if not SOURCE_DEMO_PATTERN.fullmatch(source_demo):
            continue

        demo_group = data_group[source_demo]
        if not isinstance(demo_group, h5py.Group):
            raise TypeError(f"Expected /data/{source_demo} to be a group")

        try:
            task_info = demo_group["task_info"]
            task_id_dataset = task_info["task_id"]
            task_pair_dataset = task_info["task_pair"]
        except KeyError as error:
            raise ValueError(
                f"Missing task metadata in /data/{source_demo}: {error}"
            ) from error

        if not isinstance(task_info, h5py.Group):
            raise TypeError(f"Expected /data/{source_demo}/task_info to be a group")
        if not isinstance(task_id_dataset, h5py.Dataset):
            raise TypeError(f"Expected {task_id_dataset.name} to be a dataset")
        if not isinstance(task_pair_dataset, h5py.Dataset):
            raise TypeError(f"Expected {task_pair_dataset.name} to be a dataset")

        task_pairs[decode_task_id(task_id_dataset)].append(
            TaskPair(
                task_pair=decode_task_pair(task_pair_dataset),
                source_demo=source_demo,
            )
        )

    if not task_pairs:
        raise ValueError(
            "No source demo groups matching /data/demo_<integer> were found"
        )

    return dict(task_pairs)


def validate_task_pairs(
    task_pairs: dict[str, list[TaskPair]], rules_group: h5py.Group
) -> dict[str, list[TaskPair]]:
    missing_rules = sorted(set(task_pairs) - set(rules_group))
    if missing_rules:
        raise ValueError(f"Missing rules for task IDs: {', '.join(missing_rules)}")

    ordered_task_pairs: dict[str, list[TaskPair]] = {}
    for task_id, pairs in task_pairs.items():
        ordered_pairs = sorted(
            pairs,
            key=lambda pair: (
                pair.task_pair,
                int(pair.source_demo.removeprefix("demo_")),
            ),
        )
        unique_pairs: dict[int, TaskPair] = {}
        duplicate_sources: dict[int, list[str]] = defaultdict(list)
        for pair in ordered_pairs:
            if pair.task_pair in unique_pairs:
                duplicate_sources[pair.task_pair].append(pair.source_demo)
                continue
            unique_pairs[pair.task_pair] = pair

        ordered_pairs = list(unique_pairs.values())
        pair_indices = [pair.task_pair for pair in ordered_pairs]
        expected_indices = list(range(len(ordered_pairs)))
        if pair_indices != expected_indices:
            raise ValueError(
                f"Task {task_id} has task-pair indices {pair_indices}; "
                f"expected {expected_indices}"
            )
        if duplicate_sources:
            duplicate_count = sum(
                len(sources) for sources in duplicate_sources.values()
            )
            duplicate_indices = sorted(duplicate_sources)
            warnings.warn(
                f"Task {task_id} has {duplicate_count} duplicate source episode(s) for "
                f"task-pair indices {duplicate_indices}; keeping the earliest source episode for each pair",
                UserWarning,
                stacklevel=2,
            )
        ordered_task_pairs[task_id] = ordered_pairs

    return ordered_task_pairs


def get_test_pair_counts(
    task_ids: list[str], arc_tasks: Iterable[Task]
) -> dict[str, int]:
    """Return the number of original ARC test pairs for each recorded task.

    The recorded source episodes are ordered by ``task_pair``.  A generated
    task view reserves the final N recorded episodes as tests, where N comes
    from the typed ARC task loaded through ``arc_utils``.
    """
    tasks_by_id = {task.task_id: task for task in arc_tasks}
    missing_task_ids = sorted(set(task_ids) - set(tasks_by_id))
    if missing_task_ids:
        raise ValueError(
            f"Missing ARC tasks for recorded task IDs: {', '.join(missing_task_ids)}"
        )

    test_pair_counts: dict[str, int] = {}
    for task_id in task_ids:
        test_pair_count = len(tasks_by_id[task_id].test)
        if not test_pair_count:
            raise ValueError(f"ARC task {task_id} must contain at least one test pair")
        test_pair_counts[task_id] = test_pair_count

    return test_pair_counts


def validate_destinations(
    data_group: h5py.Group,
    task_ids: list[str],
    *,
    overwrite: bool,
) -> None:
    for task_id in task_ids:
        if task_id not in data_group:
            continue
        if not overwrite:
            raise FileExistsError(
                f"/data/{task_id} already exists; use --overwrite to replace generated views"
            )
        existing_group = data_group[task_id]
        if (
            not isinstance(existing_group, h5py.Group)
            or existing_group.attrs.get("view_type") != VIEW_MARKER
        ):
            raise ValueError(
                f"Refusing to replace /data/{task_id}: it is not a generated task view"
            )


def create_task_views(
    dataset_path: Path,
    *,
    arc_tasks: Iterable[Task] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    mode = "r" if dry_run else "r+"
    with h5py.File(dataset_path, mode) as dataset:
        try:
            data_group = dataset["data"]
            rules_group = dataset["rules"]
        except ValueError as error:
            raise ValueError(
                f"Dataset is missing required root group: {error}"
            ) from error

        if not isinstance(data_group, h5py.Group) or not isinstance(
            rules_group, h5py.Group
        ):
            raise TypeError("Both /data and /rules must be HDF5 groups")

        task_pairs = validate_task_pairs(collect_task_pairs(data_group), rules_group)
        task_ids = sorted(task_pairs)
        if arc_tasks is None:
            arc_tasks = [*load_tasks("training"), *load_tasks("evaluation")]
        arc_tasks = list(arc_tasks)
        tasks_by_id = {task.task_id: task for task in arc_tasks}
        test_pair_counts = get_test_pair_counts(task_ids, arc_tasks)
        for task_id in task_ids:
            recorded_pair_count = len(task_pairs[task_id])
            test_pair_count = test_pair_counts[task_id]
            if recorded_pair_count < test_pair_count:
                raise ValueError(
                    f"Task {task_id} has {recorded_pair_count} recorded pairs but "
                    f"requires {test_pair_count} final test pairs"
                )
            if dry_run:
                pairs = task_pairs[task_id]
                demo_count = len(pairs) - test_pair_count
                test_indices = [pair.task_pair for pair in pairs[-test_pair_count:]]
                print(
                    f"Would create /data/{task_id} with {demo_count} demos, "
                    f"{test_pair_count} tests (pairs {test_indices}), and rule link"
                )
        validate_destinations(data_group, task_ids, overwrite=overwrite)

        if dry_run:
            return len(task_ids)

        for task_id in task_ids:
            if task_id in data_group:
                del data_group[task_id]

            view_group = data_group.create_group(task_id)
            view_group.attrs["view_type"] = VIEW_MARKER
            grid_metadata = view_group.create_group("arc_grids")
            pairs = task_pairs[task_id]
            test_pair_count = test_pair_counts[task_id]
            for pair in pairs[:-test_pair_count]:
                episode_key = f"demo_{pair.task_pair}"
                view_group[episode_key] = data_group[pair.source_demo]
                _write_grid_metadata(
                    grid_metadata.create_group(episode_key),
                    tasks_by_id[task_id],
                    pair.task_pair,
                )
            for test_index, pair in enumerate(pairs[-test_pair_count:]):
                episode_key = f"test_{test_index}"
                view_group[episode_key] = data_group[pair.source_demo]
                _write_grid_metadata(
                    grid_metadata.create_group(episode_key),
                    tasks_by_id[task_id],
                    pair.task_pair,
                )
            view_group["rule"] = rules_group[task_id]

    return len(task_ids)


def _write_grid_metadata(group: h5py.Group, task: Task, task_pair: int) -> None:
    pairs = [*task.train, *task.test]
    if not 0 <= task_pair < len(pairs):
        raise ValueError(
            f"Task {task.task_id} has no ARC grid pair at index {task_pair}"
        )
    pair = pairs[task_pair]
    group.create_dataset("input_grid", data=pair.input)
    group.create_dataset("output_grid", data=pair.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing task views created by this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print the views without modifying it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_count = create_task_views(
        args.dataset,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Created"
    print(f"{action} {task_count} task-centric views in {args.dataset}")


if __name__ == "__main__":
    main()
