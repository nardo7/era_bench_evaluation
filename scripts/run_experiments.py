from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluations.evaluation_config import (
    EvaluationConfig,
    load_evaluation_config,
)
from src.evaluations.evaluator import run_evaluation

GRID_FOCUSED_EXPERIMENTS = (
    {
        "induce_rule_summary": False,
        "include_frames": False,
        "include_actions": False,
        "include_final_frames": False,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_actions": False,
        "include_final_frames": False,
    },
    {
        "induce_rule_summary": False,
        "include_frames": True,
        "include_actions": False,
        "include_final_frames": False,
    },
    {
        "induce_rule_summary": True,
        "include_frames": True,
        "include_actions": False,
        "include_final_frames": False,
    },
    {
        "induce_rule_summary": False,
        "include_frames": False,
        "include_actions": False,
        "include_final_frames": True,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_actions": False,
        "include_final_frames": True,
    },
    {
        "induce_rule_summary": True,
        "include_frames": True,
        "include_actions": True,
        "include_final_frames": False,
    },
)

# Embodied prompts always include actions, and need either frames or textual grids.
# Keep this independent from the grid-focused mask so its sweep can evolve separately.
EMBODIED_EXPERIMENTS = (
    {
        "induce_rule_summary": False,
        "include_frames": False,
        "include_textual_grids": False,
        "include_final_frames": True,
        "induce_scene_descriptions": False,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_textual_grids": False,
        "include_final_frames": True,
        "induce_scene_descriptions": True,
    },
    {
        "induce_rule_summary": False,
        "include_frames": False,
        "include_textual_grids": True,
        "include_final_frames": False,
        "induce_scene_descriptions": False,
    },
    {
        "induce_rule_summary": False,
        "include_frames": False,
        "include_textual_grids": True,
        "include_final_frames": True,
        "induce_scene_descriptions": False,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_textual_grids": True,
        "include_final_frames": False,
        "induce_scene_descriptions": False,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_textual_grids": False,
        "include_final_frames": True,
        "induce_scene_descriptions": True,
    },
    {
        "induce_rule_summary": True,
        "include_frames": False,
        "include_textual_grids": True,
        "include_final_frames": True,
        "induce_scene_descriptions": True,
    },
)

EXPERIMENTS_BY_PROMPT_MODE = {
    "grid_focused": GRID_FOCUSED_EXPERIMENTS,
    "embodied": EMBODIED_EXPERIMENTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=EXPERIMENTS_BY_PROMPT_MODE,
        default="embodied",
        help="Prompt type to sweep (default: embodied).",
    )
    return parser.parse_args()


def experiment_configs(
    config: EvaluationConfig, prompt_mode: str = "embodied"
) -> Iterator[EvaluationConfig]:
    """Yield the requested experiment configurations for one prompt type."""
    for experiment in EXPERIMENTS_BY_PROMPT_MODE[prompt_mode]:
        evaluation = replace(
            config.evaluation,
            prompt_mode=prompt_mode,
            **experiment,
        )
        yield replace(config, evaluation=evaluation)


def main() -> None:
    args = parse_args()
    config = load_evaluation_config(args.config)
    experiments = EXPERIMENTS_BY_PROMPT_MODE[args.prompt_mode]
    for index, experiment_config in enumerate(
        experiment_configs(config, args.prompt_mode), start=1
    ):
        evaluation = experiment_config.evaluation
        parameters = {
            "prompt_mode": evaluation.prompt_mode,
            "induce_rule_summary": evaluation.induce_rule_summary,
            "induce_scene_descriptions": evaluation.induce_scene_descriptions,
            "include_frames": evaluation.include_frames,
            "include_textual_grids": evaluation.include_textual_grids,
            "include_actions": evaluation.include_actions,
            "include_final_frames": evaluation.include_final_frames,
            "cameras": list(evaluation.cameras),
        }
        print(f"Running experiment {index}/{len(experiments)}: {parameters}")
        summary = run_evaluation(experiment_config)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
