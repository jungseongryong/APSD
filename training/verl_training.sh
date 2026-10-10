#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash training/verl_training.sh <run_name|auto> <apsd|sdpo> <data_dir> [Hydra overrides...]" >&2
    exit 2
fi

export EXPERIMENT="$1"
config_name="$2"
data_dir="$3"
shift 3

if [[ "$config_name" != "apsd" && "$config_name" != "sdpo" ]]; then
    echo "Expected config name apsd or sdpo." >&2
    exit 2
fi
if [[ "$EXPERIMENT" == "auto" ]]; then
    export EXPERIMENT="apsd_$(date -u +%Y%m%dT%H%M%S%N)_$$"
fi
if [[ -z "$EXPERIMENT" || "$EXPERIMENT" == "." || "$EXPERIMENT" == ".." || ! "$EXPERIMENT" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Run names must contain only letters, digits, dots, underscores, or hyphens." >&2
    exit 2
fi
if [[ ! -f "$data_dir/train.parquet" || ! -f "$data_dir/test.parquet" ]]; then
    echo "Data directory must contain train.parquet and test.parquet: $data_dir" >&2
    exit 2
fi

export TASK="$(cd "$data_dir" && pwd)"
export APSD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export APSD_LOG_DIR="${APSD_LOG_DIR:-$APSD_ROOT/logs}"
export APSD_CHECKPOINT_DIR="${APSD_CHECKPOINT_DIR:-$APSD_ROOT/checkpoints}"
export PYTHONPATH="$APSD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
ulimit -c 0

cd "$APSD_ROOT"
echo "Experiment: $EXPERIMENT"
echo "Config: $config_name"
echo "Data: $TASK"
exec "${PYTHON_BIN:-python}" -m verl.trainer.main_ppo --config-name "$config_name" "$@"
