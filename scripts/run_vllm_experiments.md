# Slurm experiment launcher

`run_vllm_experiments.slurm` runs one local vLLM server and one predefined ERA
Bench prompt-context sweep in the same Slurm allocation. It waits for the server
to accept OpenAI-compatible requests before starting the sweep, and terminates
the server when the job exits or receives the scheduler's termination signal.

## Before submitting

1. Place the YAML configuration inside this checkout, normally under `configs/`.
2. Create `<checkout>/.venv` and install the evaluation and vLLM dependencies.
3. Edit the `#SBATCH` resource directives and `module load` commands near the
   top of the script for your cluster. The checked-in values target UBELIX with
   one H100 GPU, 8 CPUs, 32 GB RAM, and a five-hour limit.
4. Confirm the server command without allocating a GPU:

   ```bash
   .venv/bin/python scripts/serve_vllm.py \
     --config configs/qwen3_vl_8b.yaml \
     --dry-run
   ```

For YAML fields, credentials, and endpoint settings, see the
[configuration reference](../configs/README.md).

## Submit a sweep

```bash
sbatch scripts/run_vllm_experiments.slurm \
  configs/qwen3_vl_8b.yaml
```

The optional `--prompt-mode` is passed to `scripts/run_experiments.py` and
defaults to `embodied`; pass `--prompt-mode grid_focused` for the grid-output
sweep. The launcher runs every
predefined ablation for that mode, not a single call to `evaluate_vlm.py`.

The configuration path is resolved relative to the submission directory, then
the launcher finds the checkout by walking upward from that configuration. This
allows `sbatch` to be submitted from outside the repository as long as the
configuration remains inside it.

## Runtime behavior and outputs

The job starts `scripts/serve_vllm.py` in the background and derives the health
and model-list URLs, along with any configured API key, from the YAML. It polls
`/health` and `/v1/models` until the server is ready, then starts the sweep.
Set `SERVER_START_TIMEOUT_SECONDS` before submission to override the default
ten-minute startup timeout:

```bash
SERVER_START_TIMEOUT_SECONDS=900 \
  sbatch scripts/run_vllm_experiments.slurm configs/qwen3_vl_8b.yaml
```

Scheduler stdout and stderr default to `~/era-bench-vllm.out` and
`~/era-bench-vllm.err`. The background server log is written to the checkout as
`<slurm-job-id>-vllm.log`; evaluation artifacts go to `evaluation.output_dir` in
the supplied YAML. On timeout or an early server exit, the launcher prints the
last 200 server-log lines to stderr.

The script requests `SIGTERM` 120 seconds before its wall-time limit and waits
briefly for vLLM to stop before force-killing it if necessary. Check the
scheduler log and the server log after a cancelled or failed job.
