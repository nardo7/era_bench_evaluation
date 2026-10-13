"""Diagnose one whole-scene episode per VLM request."""

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
    run_grid_perception_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--media-mode",
        choices=("final_frame", "sampled_frames", "sampled_video", "all"),
        default="all",
        help="Use one final image, either video representation, or all three.",
    )
    parser.add_argument(
        "--response-format",
        choices=("structured", "json", "scene_then_structured", "both"),
        default="both",
        help=(
            "Use direct constrained grids, ordinary JSON, or a scene-description "
            "scaffold followed by constrained grids."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <configured output_dir>/grid_perception_diagnostic.",
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
        "--video-fps",
        type=float,
        help=(
            "FPS used for MP4 encoding and requested from Qwen/vLLM. Defaults "
            "to dataset.sampled_fps (1 FPS in the provided configs)."
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
    media_modes = (
        ("final_frame", "sampled_frames", "sampled_video")
        if args.media_mode == "all"
        else (args.media_mode,)
    )
    response_formats = (
        ("structured", "json")
        if args.response_format == "both"
        else (args.response_format,)
    )
    summaries = {
        f"{media_mode}_{response_format}": run_grid_perception_diagnostic(
            config,
            media_mode=media_mode,
            response_format=response_format,
            output_dir=args.output_dir,
            max_tasks=args.max_tasks,
            shuffle_tasks=args.shuffle_tasks,
            seed=args.seed,
            video_fps=args.video_fps,
            cameras=tuple(args.cameras),
            image_scale=args.image_scale,
        )
        for media_mode in media_modes
        for response_format in response_formats
    }
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
