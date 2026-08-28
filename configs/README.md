# Evaluation configuration reference

Each YAML file in this directory defines one evaluation run and, for local
backends, the command used to start its model server. Copy an existing file,
adjust it for your dataset and endpoint, then pass it to `serve_vllm.py` and/or
`evaluate_vlm.py` from the repository root.

```bash
cp configs/qwen2_5_vl_3b.yaml configs/my_model.yaml
# Edit configs/my_model.yaml.
python scripts/serve_vllm.py --config configs/my_model.yaml --dry-run
python scripts/evaluate_vlm.py --config configs/my_model.yaml
```

All relative paths are resolved relative to the YAML file, not the current
working directory. For prompt, media, scoring, and transport semantics, see the
[architecture documentation](../docs/architecture.md).

## Structure

```yaml
dataset:
  paths: [../data/era_bench.hdf5]
  frame_stride: 20
  sampled_fps: 1.0

# Optional. See Credentials below.
env_file: ../.env

model:
  id: Qwen/Qwen2.5-VL-3B-Instruct
  served_name: qwen2.5-vl-3b
  vllm_executable: vllm
  vllm_args: {}

server:
  base_url: http://127.0.0.1:8000/v1
  api_key: null
  timeout_seconds: 300

generation:
  max_tokens: 4096
  max_tokens_field: max_completion_tokens
  temperature: 0.0
  top_p: 1.0
  seed: 0
  structured_output: true
  request_overrides: {}

evaluation:
  output_dir: ../outputs/my_model
  max_samples: 10
  continue_on_error: false
  prompt_mode: embodied
  include_frames: true
  include_final_frames: false
  include_textual_grids: false
  include_actions: false
  induce_rule_summary: true
  induce_scene_descriptions: false
  cameras: [table_cam]
```

Only `dataset`, `model`, and `evaluation` are required top-level mappings.
Every field omitted from `server` or `generation` receives the default shown in
the example above, except where noted below.

## Dataset

`dataset.paths` is one HDF5 path or a non-empty list of paths. The evaluator
discovers task views in every listed file. The files must already contain the
task-centric views described in
[HDF5 preparation and layout](../docs/architecture.md#hdf5-preparation-and-layout).

`frame_stride` selects every Nth camera frame and always retains the final frame;
it must be at least 1. `sampled_fps` is positive and describes the sampled
sequence. It is not sent to the model processor, so keep any
backend-specific video-FPS request override consistent with it.

## Model and local server

`model.id` is required and identifies the model passed to the local server.
`model.served_name` is the name sent in Chat Completions requests; it defaults to
`id`. `vllm_executable` defaults to `vllm` and can be set to `vllm-mlx` for the
Apple Silicon configuration.

`vllm_args` is passed directly to the local server command. Each key is prefixed
with `--` exactly as written, so use the option spelling expected by the chosen
executable (for example, `served-model-name`). `true` emits a flag,
`false` and `null` omit it, lists repeat the option, and dictionaries are passed
as compact JSON. Use `--dry-run` with `scripts/serve_vllm.py` to inspect the
exact command.

Hosted configurations still need `model.id` and `served_name`, but omit
local-server settings and do not use `serve_vllm.py`.

## Server and credentials

`server.base_url` defaults to `http://127.0.0.1:8000/v1`.
`timeout_seconds` defaults to 300. Set either a literal `api_key` or
`api_key_env_name`; the two options are mutually exclusive. Prefer the latter:

```yaml
env_file: ../.env
server:
  base_url: https://openrouter.ai/api/v1
  api_key_env_name: OPENROUTER_API_KEY
```

Copy `../.env.example` to `../.env` and keep `.env` untracked. If `env_file` is
omitted, the loader looks for `.env` between the configuration directory and the
repository root. Existing shell environment variables take precedence over
values in that file.

## Generation

`max_tokens` and `max_tokens_field` control the completion-token limit. Use the
field name required by the endpoint, such as `max_completion_tokens` or
`max_tokens`. `temperature`, `top_p`, and `seed` may be `null` to omit them from
the request.

With `structured_output: true`, the evaluator sends a strict JSON Schema for
the selected output type. Set it to `false` only for endpoints that do not
support structured output. `request_overrides` is copied into every Chat
Completions request after the standard generation fields; use it for
backend-specific options such as processor video sampling.

## Evaluation

`output_dir` is required. `max_samples` must be a positive integer or `null` for
all discovered samples. `continue_on_error` controls whether an error stops the
run after its result record is written.

`prompt_mode` selects the prediction type and enforces its required context:

| Mode | Prediction | Normalization |
| --- | --- | --- |
| `embodied` | Semantic actions | Forces `include_actions: true`; requires videos, final frames, or textual grids. |
| `grid_focused` | Direct output grid | Forces `include_textual_grids: true`. |

The remaining context fields are independent ablation controls:

- `include_frames`: add sampled table-camera demonstration videos and the initial test image.
- `include_final_frames`: add final demonstration images and initial test images from each selected camera.
- `include_textual_grids`: add solved demonstration grids and the test input grid as text.
- `include_actions`: add compressed semantic demonstration actions; it is always enabled for `embodied`.
- `induce_rule_summary`: require a `rule_summary` response field and ask for it in the prompt.
- `induce_scene_descriptions`: require a structured qualitative scene-description field.

`cameras` defaults to `[table_cam]`. The current dataset API supports
`table_cam` and `origin_top_cam`; camera names must be non-empty and unique.
Videos always use `table_cam`, while final-frame and test-image attachments use
the selected cameras. See
[Multimodal prompt construction](../docs/architecture.md#4-multimodal-prompt-construction)
for the precise media behavior.

## Included examples

- `qwen2_5_*.yaml` and `qwen3_*.yaml`: local vLLM examples.
- `qwen2_5_vl_3b_mlx.yaml`: Apple Silicon vllm-mlx example.
- `openrouter_*.yaml`: hosted OpenRouter examples using an environment-backed
  API key.
- `internvl3_5_38b.yaml`: local InternVL example.

Treat these files as starting points: confirm the dataset path, available model,
server options, endpoint capabilities, and output directory before a full run.
