"""Ask a VLM to describe visible cubes in one whole-scene image per request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluations.evaluation_config import load_evaluation_config
from src.evaluations.grid_perception_diagnostic import (
    SUPPORTED_CAMERAS,
    run_scene_description_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <configured output_dir>/scene_description_diagnostic.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Optional task limit; omitted means the complete dataset.",
    )
    parser.add_argument(
        "--shuffle-tasks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle task order before applying --max-tasks (default: disabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Seed for model generation and task shuffling. Defaults to the "
            "configured generation seed, or 0 when it is null."
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=SUPPORTED_CAMERAS,
        default=["table_cam"],
        help="Use table_cam, origin_top_cam, or both perspectives.",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=1,
        help="Nearest-neighbor whole-scene scale factor; 1 preserves native size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_evaluation_config(args.config)
    summary = run_scene_description_diagnostic(
        config,
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
        shuffle_tasks=args.shuffle_tasks,
        seed=args.seed,
        cameras=tuple(args.cameras),
        image_scale=args.image_scale,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
