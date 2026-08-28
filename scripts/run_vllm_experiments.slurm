#!/bin/bash
# -----------------------------------------------------------------------------
# USER CODE TUNNEL SETTINGS
# -----------------------------------------------------------------------------
#SBATCH --output=~/era-bench-vllm.out
#SBATCH --error=~/era-bench-vllm.err
#SBATCH --time=05:00:00

# Account / partition
#SBATCH --account=gratis
#SBATCH --partition=gpu

# Request 1x H100
#SBATCH --gres=gpu:h100:1

# CPU / RAM for the VS Code server + your workload
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Send SIGTERM 120s before walltime so cleanup can run
#SBATCH --signal=B:TERM@120

#SBATCH --job-name=era-bench-vllm


# Initialize Lmod (SLURM batch shells do not source /etc/profile.d, so the
# module shell function is not defined unless we source it explicitly).
# UBELIX uses Lmod at /software.9/lmod/lmod/init/bash.
# source /software.9/lmod/lmod/init/bash

# Ensure CUDA toolkit is available on each new SLURM job start.
module purge
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

set -euo pipefail

usage() {
    echo "Usage: sbatch $0 CONFIG_PATH [--prompt-mode grid_focused|embodied]" >&2
    echo "Example: sbatch $0 configs/qwen3_vl_8b.yaml" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage

# Resolve the manually supplied config before changing to the repository root.
CONFIG_PATH="$1"
shift
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG_PATH"
fi
[[ -f "$CONFIG_PATH" ]] || {
    echo "Configuration file not found: $CONFIG_PATH" >&2
    exit 2
}

find_project_root() {
    # Slurm copies the batch script to its spool directory, so BASH_SOURCE[0]
    # is not a stable way to locate this checkout.  Configs normally live in
    # <checkout>/configs; walk upward from the supplied config to find it.
    local candidate
    candidate="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
    while [[ "$candidate" != "/" ]]; do
        if [[ -f "$candidate/scripts/run_experiments.py" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        candidate="$(dirname "$candidate")"
    done
    return 1
}

PROJECT_ROOT="$(find_project_root)" || {
    echo "Could not locate the evaluation checkout above config: $CONFIG_PATH" >&2
    echo "Place the config under the checkout (for example, configs/model.yaml)." >&2
    exit 2
}
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
[[ -x "$VENV_PYTHON" ]] || {
    echo "Expected virtual-environment Python was not found: $VENV_PYTHON" >&2
    exit 1
}
# Activation puts the venv's `vllm` executable first on PATH.  Use the
# interpreter explicitly below so every Python process uses this environment.
source "$VENV_DIR/bin/activate"

# The evaluation config is the single source of truth for the server URL and
# optional API key.  /health checks the vLLM engine; /v1/models confirms the
# OpenAI-compatible request endpoint is accepting connections.
read -r HEALTH_URL MODELS_URL VLLM_API_KEY < <(
    "$VENV_PYTHON" - "$CONFIG_PATH" <<'PY'
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import sys

from src.evaluations.evaluation_config import load_evaluation_config

config = load_evaluation_config(Path(sys.argv[1]))
base_url = config.server.base_url.rstrip("/")
parts = urlsplit(base_url)
path = parts.path.rstrip("/")
if path.endswith("/v1"):
    path = path[:-3]
root_url = urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
print(root_url + "/health", base_url + "/models", config.server.api_key or "")
PY
)

VLLM_LOG="${SLURM_JOB_ID:-local}-vllm.log"
VLLM_PID=""

cleanup() {
    local exit_status=$?
    trap - EXIT TERM INT

    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$VLLM_PID" 2>/dev/null && kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "Starting vLLM with config: $CONFIG_PATH"
echo "vLLM server log: $VLLM_LOG"
"$VENV_PYTHON" scripts/serve_vllm.py --config "$CONFIG_PATH" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

CURL_AUTH_ARGS=()
if [[ -n "$VLLM_API_KEY" ]]; then
    CURL_AUTH_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-600}"
echo "Waiting up to ${SERVER_START_TIMEOUT_SECONDS}s for vLLM to become ready..."
start_seconds=$SECONDS
while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before it became ready. Its log follows:" >&2
        tail -n 200 "$VLLM_LOG" >&2 || true
        exit 1
    fi

    if curl --fail --silent --show-error --max-time 5 "$HEALTH_URL" >/dev/null 2>&1 \
        && curl --fail --silent --show-error --max-time 5 "${CURL_AUTH_ARGS[@]}" "$MODELS_URL" >/dev/null 2>&1; then
        break
    fi

    if (( SECONDS - start_seconds >= SERVER_START_TIMEOUT_SECONDS )); then
        echo "Timed out waiting for vLLM. Its log follows:" >&2
        tail -n 200 "$VLLM_LOG" >&2 || true
        exit 1
    fi
    sleep 5
done

echo "vLLM is ready; starting the experiment sweep."
"$VENV_PYTHON" scripts/run_experiments.py --config "$CONFIG_PATH" "$@"
