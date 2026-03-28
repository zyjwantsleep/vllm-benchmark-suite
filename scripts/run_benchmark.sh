#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH="$ROOT_DIR/configs/qwen25_1p5b_local.json"
LOG_DIR="$ROOT_DIR/results/logs"
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/benchmark_${RUN_ID}.log"
GPU_LOG="$LOG_DIR/gpu_metrics_${RUN_ID}.csv"
if ! command -v python >/dev/null 2>&1; then
  echo "[error] python not found in current shell." >&2
  echo "[error] please activate your vllm312 virtual environment first." >&2
  exit 1
fi

python "$ROOT_DIR/src/monitor_gpu.py" --output "$GPU_LOG" --interval 1.0 >/dev/null 2>&1 &
GPU_MONITOR_PID=$!
cleanup() {
  kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python "$ROOT_DIR/src/benchmark_client.py" --config "$CONFIG_PATH" --gpu-metrics-path "$GPU_LOG" "$@" 2>&1 | tee "$LOG_FILE"
echo "[benchmark] gpu metrics: $GPU_LOG"
echo "[benchmark] log file: $LOG_FILE"