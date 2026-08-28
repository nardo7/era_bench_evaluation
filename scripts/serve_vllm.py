"""Launch a vLLM OpenAI-compatible server from an evaluation YAML file."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluations import load_evaluation_config
from src.vllm import build_vllm_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without running it."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_vllm_command(load_evaluation_config(args.config))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
