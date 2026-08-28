"""Typed configuration for VLM evaluation and vLLM serving."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    paths: list[Path]
    frame_stride: int = 20
    sampled_fps: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    served_name: str
    vllm_executable: str = "vllm"
    vllm_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServerConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 4096
    max_tokens_field: str = "max_completion_tokens"
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = 0
    structured_output: bool = True
    request_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRunConfig:
    output_dir: Path
    max_samples: int | None = None
    continue_on_error: bool = False
    prompt_mode: str = "embodied"
    include_frames: bool = True
    include_textual_grids: bool = False
    include_actions: bool = False
    induce_rule_summary: bool = True
    induce_scene_descriptions: bool = False
    include_final_frames: bool = False
    cameras: tuple[str, ...] = ("table_cam",)

    def __post_init__(self) -> None:
        """Normalize the output and required context for each prompt focus."""
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if not self.cameras or any(not camera for camera in self.cameras):
            raise ValueError("evaluation.cameras must contain at least one camera")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("evaluation.cameras must not contain duplicates")
        if self.prompt_mode == "embodied":
            object.__setattr__(self, "include_actions", True)
            if (
                not self.include_frames
                and not self.include_final_frames
                and not self.include_textual_grids
            ):
                raise ValueError(
                    "embodied prompts must include videos, final frames or textual grids"
                )
        elif self.prompt_mode == "grid_focused":
            object.__setattr__(self, "include_textual_grids", True)
        else:
            raise ValueError("evaluation.prompt_mode must be embodied or grid_focused")


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: DatasetConfig
    model: ModelConfig
    server: ServerConfig
    generation: GenerationConfig
    evaluation: EvaluationRunConfig


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load an evaluation YAML file, resolving file paths beside the YAML."""
    path = path.resolve()
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise TypeError("the evaluation config must be a YAML mapping")
    _load_environment_file(path, raw.get("env_file"))

    base_dir = path.parent
    dataset_raw = _mapping(raw, "dataset")
    model_raw = _mapping(raw, "model")
    server_raw = _mapping(raw, "server", required=False)
    generation_raw = _mapping(raw, "generation", required=False)
    evaluation_raw = _mapping(raw, "evaluation")

    source_paths = dataset_raw.get("paths")
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("dataset.paths must contain at least one path")

    model_id = _required_string(model_raw, "id")
    served_name = str(model_raw.get("served_name", model_id))
    max_samples = evaluation_raw.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError("evaluation.max_samples must be positive or null")
    prompt_mode = str(evaluation_raw.get("prompt_mode", "embodied"))
    if prompt_mode not in {"embodied", "grid_focused"}:
        raise ValueError("evaluation.prompt_mode must be embodied or grid_focused")
    include_frames = bool(
        evaluation_raw.get("include_frames", prompt_mode == "embodied")
    )
    include_textual_grids = bool(evaluation_raw.get("include_textual_grids", False))
    include_actions = bool(evaluation_raw.get("include_actions", False))
    cameras_raw = evaluation_raw.get("cameras", ["table_cam"])
    if isinstance(cameras_raw, str):
        cameras_raw = [cameras_raw]
    if not isinstance(cameras_raw, list) or not all(
        isinstance(camera, str) and camera for camera in cameras_raw
    ):
        raise ValueError("evaluation.cameras must be a non-empty list of camera names")

    frame_stride = int(dataset_raw.get("frame_stride", 20))
    if frame_stride < 1:
        raise ValueError("dataset.frame_stride must be at least 1")
    sampled_fps = float(dataset_raw.get("sampled_fps", 1.0))
    if sampled_fps <= 0:
        raise ValueError("dataset.sampled_fps must be positive")
    max_tokens_field = str(
        generation_raw.get("max_tokens_field", "max_completion_tokens")
    )
    if not max_tokens_field:
        raise ValueError("generation.max_tokens_field must be a non-empty string")
    seed_raw = generation_raw.get("seed", 0)
    temperature_raw = generation_raw.get("temperature", 0.0)
    top_p_raw = generation_raw.get("top_p", 1.0)
    return EvaluationConfig(
        dataset=DatasetConfig(
            paths=[_resolve_path(base_dir, value) for value in source_paths],
            frame_stride=frame_stride,
            sampled_fps=sampled_fps,
        ),
        model=ModelConfig(
            model_id=model_id,
            served_name=served_name,
            vllm_executable=str(model_raw.get("vllm_executable", "vllm")),
            vllm_args=dict(model_raw.get("vllm_args", {})),
        ),
        server=ServerConfig(
            base_url=str(server_raw.get("base_url", "http://127.0.0.1:8000/v1")),
            api_key=_server_api_key(server_raw),
            timeout_seconds=float(server_raw.get("timeout_seconds", 300.0)),
        ),
        generation=GenerationConfig(
            max_tokens=int(generation_raw.get("max_tokens", 4096)),
            max_tokens_field=max_tokens_field,
            temperature=(
                float(temperature_raw) if temperature_raw is not None else None
            ),
            top_p=float(top_p_raw) if top_p_raw is not None else None,
            seed=int(seed_raw) if seed_raw is not None else None,
            structured_output=bool(generation_raw.get("structured_output", True)),
            request_overrides=dict(generation_raw.get("request_overrides", {})),
        ),
        evaluation=EvaluationRunConfig(
            output_dir=_resolve_path(
                base_dir, _required_string(evaluation_raw, "output_dir")
            ),
            max_samples=max_samples,
            continue_on_error=bool(evaluation_raw.get("continue_on_error", False)),
            prompt_mode=prompt_mode,
            include_frames=include_frames,
            include_textual_grids=include_textual_grids,
            include_actions=include_actions,
            induce_rule_summary=bool(evaluation_raw.get("induce_rule_summary", True)),
            induce_scene_descriptions=bool(
                evaluation_raw.get("induce_scene_descriptions", False)
            ),
            include_final_frames=bool(
                evaluation_raw.get("include_final_frames", False)
            ),
            cameras=tuple(cameras_raw),
        ),
    )


def _mapping(
    source: dict[str, Any], key: str, *, required: bool = True
) -> dict[str, Any]:
    value = source.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a YAML mapping")
    return value


def _required_string(source: dict[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _server_api_key(server: dict[str, Any]) -> str | None:
    """Return a literal API key or load one from the configured environment."""
    literal_key = server.get("api_key")
    environment_name = server.get("api_key_env_name")
    if literal_key is not None and environment_name is not None:
        raise ValueError(
            "server.api_key and server.api_key_env_name are mutually exclusive"
        )
    if environment_name is None:
        return str(literal_key) if literal_key is not None else None
    if not isinstance(environment_name, str) or not environment_name:
        raise ValueError("server.api_key_env_name must be a non-empty string")
    environment_value = os.environ.get(environment_name)
    if not environment_value:
        raise ValueError(
            f"environment variable {environment_name} configured by "
            "server.api_key_env_name is not set"
        )
    return environment_value


def _load_environment_file(config_path: Path, configured_path: Any) -> None:
    """Load an optional dotenv file without replacing shell-provided secrets."""
    if configured_path is None:
        environment_path = _find_default_environment_file(config_path)
        if environment_path is None:
            return
    else:
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError("env_file must be a non-empty string")
        environment_path = _resolve_path(config_path.parent, configured_path)
        if not environment_path.is_file():
            raise ValueError(f"env_file does not exist: {environment_path}")

    for line_number, source_line in enumerate(
        environment_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name.isidentifier():
            raise ValueError(
                f"invalid env_file entry at {environment_path}:{line_number}"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _find_default_environment_file(config_path: Path) -> Path | None:
    """Find a project .env file without searching beyond the repository root."""
    for directory in (config_path.parent, *config_path.parent.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break
    return None


def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()
