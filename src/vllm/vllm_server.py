"""Build the vLLM server command from the shared evaluation config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluations.evaluation_config import EvaluationConfig


def build_vllm_command(config: EvaluationConfig) -> list[str]:
    """Return a configurable ``vllm serve`` command."""
    arguments = dict(config.model.vllm_args)
    arguments.setdefault("served-model-name", config.model.served_name)

    command = [
        config.model.vllm_executable,
        "serve",
        config.model.model_id,
    ]
    for name, value in arguments.items():
        _append_argument(command, name, value)
    return command


def _append_argument(command: list[str], name: str, value: Any) -> None:
    option = name if name.startswith("--") else f"--{name}"
    if isinstance(value, bool):
        if value:
            command.append(option)
        return
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            command.extend((option, _argument_value(item)))
        return
    command.extend((option, _argument_value(value)))


def _argument_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, Path):
        return str(value)
    return str(value)
