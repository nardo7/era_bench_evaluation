# ERA Bench VLM evaluation

This project evaluates vision-language models on embodied ARC tasks recorded in
Isaac Lab. Each evaluation sample combines the solved demonstrations for one ARC
task with a held-out test pair. Models can predict either semantic robot actions
or a direct ARC output grid through an OpenAI-compatible endpoint.

The repository supports local vLLM, Apple Silicon through vllm-mlx, and hosted
OpenAI-compatible providers. Evaluation runs are configured with YAML and write
per-sample JSONL records plus aggregate JSON summaries.

For the HDF5 format, task-view construction, prompt and target-isolation
contracts, model transport, scoring semantics, diagnostics, and module ownership,
see [the architecture documentation](docs/architecture.md).

## Setup

Create and activate a virtual environment, then install the evaluation
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a local vLLM inference host, also install:

```bash
python -m pip install -r requirements-vllm.txt
```

On Apple Silicon, install the pinned vllm-mlx stack instead:

```bash
python -m pip install -r requirements-vllm-metal.txt
```

The evaluator requires an ERA Bench HDF5 file with task-centric views. Set
`dataset.paths` in the selected file under `configs/` to that dataset. Dataset
preparation and the complete raw/task-view layouts are documented in
[HDF5 preparation and layout](docs/architecture.md#hdf5-preparation-and-layout).
For every YAML option and the included configuration examples, see the
[configuration reference](configs/README.md).

## Quick start

Choose a configuration and inspect the generated local server command:

```bash
python scripts/serve_vllm.py \
  --config configs/qwen2_5_vl_3b.yaml \
  --dry-run
```

Start the model server in the first terminal:

```bash
python scripts/serve_vllm.py --config configs/qwen2_5_vl_3b.yaml
```

Run the evaluation in a second terminal:

```bash
python scripts/evaluate_vlm.py --config configs/qwen2_5_vl_3b.yaml
```

Results are written beneath the configured `evaluation.output_dir`. Start with a
small `evaluation.max_samples` value, then set it to `null` for a full run.
Configuration behavior and artifact naming are described in
[Evaluation data flow](docs/architecture.md#evaluation-data-flow).

### Apple Silicon

Use the MLX configuration for both processes:

```bash
python scripts/serve_vllm.py --config configs/qwen2_5_vl_3b_mlx.yaml
python scripts/evaluate_vlm.py --config configs/qwen2_5_vl_3b_mlx.yaml
```

### Hosted endpoint

Hosted configurations do not need a local server process. Copy the environment
template, add the required credential, and run the evaluator directly:

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY in .env.
python scripts/evaluate_vlm.py \
  --config configs/openrouter_qwen3_vl_8b.yaml
```

Keep credentials out of YAML and source control. Transport differences and
supported visual representations are covered in
[vLLM transport](docs/architecture.md#5-vllm-transport).

## Additional commands

Run the predefined prompt-context experiment sweep:

```bash
python scripts/run_experiments.py \
  --config configs/qwen2_5_vl_3b.yaml
```

This runs the `embodied` sweep by default; add `--prompt-mode grid_focused` to
run the direct-grid sweep instead.

To run that sweep with a local vLLM server inside one Slurm allocation, see the
[Slurm experiment launcher](scripts/run_vllm_experiments.md).

The repository also includes grid-perception and scene-description diagnostics.
Use the command help for available options:

```bash
python scripts/diagnose_grid_perception.py --help
python scripts/diagnose_scene_description.py --help
```

Their purpose, inputs, and scoring are documented in
[Visual grid-perception diagnostic](docs/architecture.md#7-visual-grid-perception-diagnostic).
