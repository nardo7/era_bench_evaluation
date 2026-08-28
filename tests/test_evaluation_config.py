import os
from pathlib import Path

import pytest
from src.evaluations import load_evaluation_config
from src.vllm.vllm_server import build_vllm_command


def test_loads_yaml_and_builds_extensible_vllm_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
  frame_stride: 20
  sampled_fps: 1.0
model:
  id: Qwen/Qwen2.5-VL-3B-Instruct
  served_name: test-model
  vllm_args:
    dtype: auto
    enforce-eager: true
    limit-mm-per-prompt:
      image: 1
      video: 3
server: {}
generation:
  request_overrides:
    repetition_penalty: 1.05
evaluation:
  output_dir: output
  max_samples: 1
  include_textual_grids: true
  include_final_frames: true
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)
    command = build_vllm_command(config)

    assert config.dataset.paths == [(tmp_path / "dataset.hdf5").resolve()]
    assert config.generation.request_overrides == {"repetition_penalty": 1.05}
    assert config.evaluation.prompt_mode == "embodied"
    assert config.evaluation.induce_rule_summary
    assert config.evaluation.include_frames
    assert config.evaluation.include_actions
    assert config.evaluation.include_textual_grids
    assert config.evaluation.include_final_frames
    assert command[:3] == ["vllm", "serve", "Qwen/Qwen2.5-VL-3B-Instruct"]
    assert "--enforce-eager" in command
    assert (
        command[command.index("--limit-mm-per-prompt") + 1] == '{"image":1,"video":3}'
    )
    assert "--allowed-local-media-path" not in command


def test_loads_vllm_mlx_request_dialect(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
  served_name: test-model
  vllm_executable: vllm-mlx
  vllm_args:
    cache-memory-percent: 0.2
server: {}
generation:
  max_tokens_field: max_tokens
  seed: null
  request_overrides:
    video_fps: 1.0
    video_max_frames: 64
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.generation.max_tokens_field == "max_tokens"
    assert config.generation.seed is None
    assert config.evaluation.prompt_mode == "embodied"
    assert config.evaluation.induce_rule_summary
    assert config.evaluation.include_frames
    assert config.evaluation.include_actions
    assert not config.evaluation.include_textual_grids
    assert config.generation.request_overrides == {
        "video_fps": 1.0,
        "video_max_frames": 64,
    }
    assert build_vllm_command(config)[:3] == [
        "vllm-mlx",
        "serve",
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    ]


def test_allows_omitting_unsupported_generation_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: hosted-model
generation:
  temperature: null
  top_p: null
  seed: null
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.generation.temperature is None
    assert config.generation.top_p is None
    assert config.generation.seed is None


def test_grid_focused_prompt_normalizes_grid_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
evaluation:
  output_dir: output
  prompt_mode: grid_focused
  include_frames: false
  include_actions: true
  induce_rule_summary: false
  include_final_frames: false
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.evaluation.prompt_mode == "grid_focused"
    assert config.evaluation.include_textual_grids
    assert not config.evaluation.include_frames
    assert config.evaluation.include_actions
    assert not config.evaluation.induce_rule_summary
    assert not config.evaluation.include_final_frames


def test_loads_scene_description_induction_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
evaluation:
  output_dir: output
  prompt_mode: grid_focused
  induce_scene_descriptions: true
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.evaluation.induce_scene_descriptions


def test_final_frames_do_not_require_video_context(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
evaluation:
  output_dir: output
  prompt_mode: grid_focused
  include_frames: false
  include_final_frames: true
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.evaluation.include_final_frames


def test_rejects_embodied_prompt_without_frames_or_grids(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
evaluation:
  output_dir: output
  prompt_mode: embodied
  include_frames: false
  include_textual_grids: false
""".strip(),
        encoding="utf-8",
    )

    try:
        load_evaluation_config(config_path)
    except ValueError as error:
        assert (
            str(error)
            == "embodied prompts must include videos, final frames or textual grids"
        )
    else:
        raise AssertionError("embodied prompts require task-input context")


def test_loads_server_api_key_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-from-environment")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: qwen/qwen3-vl-8b-instruct
server:
  base_url: https://openrouter.ai/api/v1
  api_key_env_name: TEST_OPENROUTER_API_KEY
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.server.base_url == "https://openrouter.ai/api/v1"
    assert config.server.api_key == "secret-from-environment"
    assert config.model.served_name == "qwen/qwen3-vl-8b-instruct"


def test_loads_environment_file_before_resolving_server_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "secrets.env").write_text(
        """
# The user's shell environment keeps precedence over these values.
OPENROUTER_UNUSED_KEY=unused
TEST_OPENROUTER_API_KEY=secret-from-file
HF_TOKEN=hf_token_from_file
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env_file: secrets.env
dataset:
  paths: [dataset.hdf5]
model:
  id: hosted-model
server:
  api_key_env_name: TEST_OPENROUTER_API_KEY
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    config = load_evaluation_config(config_path)

    assert config.server.api_key == "secret-from-file"
    assert os.environ["HF_TOKEN"] == "hf_token_from_file"


def test_environment_file_does_not_replace_exported_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-from-shell")
    (tmp_path / "secrets.env").write_text(
        "TEST_OPENROUTER_API_KEY=secret-from-file\n", encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env_file: secrets.env
dataset:
  paths: [dataset.hdf5]
model:
  id: hosted-model
server:
  api_key_env_name: TEST_OPENROUTER_API_KEY
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    assert load_evaluation_config(config_path).server.api_key == "secret-from-shell"


def test_rejects_missing_server_api_key_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISSING_OPENROUTER_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
server:
  api_key_env_name: MISSING_OPENROUTER_API_KEY
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "environment variable MISSING_OPENROUTER_API_KEY configured by "
            "server.api_key_env_name is not set"
        ),
    ):
        load_evaluation_config(config_path)


def test_rejects_literal_and_environment_server_api_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  paths: [dataset.hdf5]
model:
  id: test-model
server:
  api_key: literal-secret
  api_key_env_name: OPENROUTER_API_KEY
evaluation:
  output_dir: output
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="server.api_key and server.api_key_env_name are mutually exclusive",
    ):
        load_evaluation_config(config_path)
