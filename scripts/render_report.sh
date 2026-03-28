#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="$ROOT_DIR/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/render_$(date +%Y%m%d_%H%M%S).log"
if ! command -v python >/dev/null 2>&1; then
  echo "[error] python not found in current shell." >&2
  echo "[error] please activate your vllm312 virtual environment first." >&2
  exit 1
fi

python "$ROOT_DIR/src/visualize_results.py" --results-dir "$ROOT_DIR/results" 2>&1 | tee "$LOG_FILE"
echo "[render] log file: $LOG_FILE"