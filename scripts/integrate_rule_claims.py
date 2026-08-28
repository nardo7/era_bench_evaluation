import argparse
import json
import sys
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_models import Task

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "era_bench.hdf5"
DEFAULT_RULES_PATH = (
    PROJECT_ROOT / "data" / "hard_successful_selected_tasks_rule_inference.json"
)


def load_tasks(rules_path: Path) -> list[Task]:
    with rules_path.open(encoding="utf-8") as rules_file:
        source = json.load(rules_file)

    tasks = [Task.model_validate(task) for task in source["tasks"]]
    task_ids = [task.task_id for task in tasks]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("The rules JSON contains duplicate task IDs")
    return tasks


def integrate_tasks(
    dataset_path: Path,
    tasks: list[Task],
    *,
    overwrite: bool = False,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(dataset_path, "r+") as dataset:
        if "rules" in dataset:
            if not overwrite:
                raise FileExistsError(
                    f"{dataset_path} already contains /rules; use --overwrite"
                )
            del dataset["rules"]

        rules_group = dataset.create_group("rules")
        for task in tasks:
            task_group = rules_group.create_group(task.task_id)
            task_group.create_dataset(
                "data",
                data=task.model_dump_json(),
                dtype=string_dtype,
            )

        if "rules" in dataset["data"]:
            del dataset["data/rules"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate inferred ARC rules into the ERA Bench HDF5 dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing /data/rules group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and load the JSON without modifying the HDF5 dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.rules)
    if args.dry_run:
        print(f"Validated {len(tasks)} tasks from {args.rules}")
        return

    integrate_tasks(args.dataset, tasks, overwrite=args.overwrite)
    print(f"Integrated {len(tasks)} tasks into {args.dataset} at /rules")


if __name__ == "__main__":
    main()
