#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH="$ROOT_DIR/configs/qwen25_1p5b_local.json"
LOG_DIR="$ROOT_DIR/results/logs"
PROMPT_DIR="$ROOT_DIR/prompts/generated"
mkdir -p "$LOG_DIR" "$PROMPT_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/matrix_${RUN_ID}.log"
if ! command -v python >/dev/null 2>&1; then
  echo "[error] python not found in current shell." >&2
  echo "[error] please activate your vllm312 virtual environment first." >&2
  exit 1
fi

mapfile -t MODEL_AND_MATRIX < <(python - <<'PY' "$CONFIG_PATH"
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
print(cfg['model']['path'])
for p in cfg['experiment_matrix']['prompt_tokens']:
    for c in cfg['experiment_matrix']['concurrency']:
        for t in cfg['experiment_matrix']['max_tokens']:
            print(f"{p} {c} {t}")
PY
)
MODEL_PATH=${MODEL_AND_MATRIX[0]}
MATRIX=("${MODEL_AND_MATRIX[@]:1}")

should_skip() {
  local prompt_tokens=$1
  local concurrency=$2
  local max_tokens=$3

  if (( prompt_tokens >= 2048 && concurrency >= 8 )); then
    return 0
  fi
  if (( prompt_tokens >= 2048 && max_tokens > 128 )); then
    return 0
  fi
  if (( prompt_tokens >= 1024 && concurrency >= 12 && max_tokens >= 256 )); then
    return 0
  fi
  if (( prompt_tokens >= 1024 && concurrency >= 8 && max_tokens > 256 )); then
    return 0
  fi
  if (( prompt_tokens >= 256 && concurrency >= 12 && max_tokens > 256 )); then
    return 0
  fi
  return 1
}

: > "$LOG_FILE"
python "$ROOT_DIR/src/generate_long_prompts.py" --model-path "$MODEL_PATH" --output-dir "$PROMPT_DIR" --targets 256 1024 --samples-per-target 8 2>&1 | tee -a "$LOG_FILE"

for item in "${MATRIX[@]}"; do
  prompt_tokens=${item%% *}
  rest=${item#* }
  concurrency=${rest%% *}
  max_tokens=${rest##* }

  if should_skip "$prompt_tokens" "$concurrency" "$max_tokens"; then
    label=$([[ "$prompt_tokens" == "0" ]] && echo base || echo "$prompt_tokens"); echo "[matrix] skip prompt=$label concurrency=$concurrency max_tokens=$max_tokens" | tee -a "$LOG_FILE"
    continue
  fi

  dataset_path="$ROOT_DIR/prompts/sample_prompts.jsonl"
  if [[ "$prompt_tokens" != "0" ]]; then
    dataset_path="$PROMPT_DIR/long_prompts_${prompt_tokens}.jsonl"
  fi
  gpu_log="$LOG_DIR/gpu_metrics_${RUN_ID}_p${prompt_tokens}_c${concurrency}_t${max_tokens}.csv"
  label=$([[ "$prompt_tokens" == "0" ]] && echo base || echo "$prompt_tokens"); echo "[matrix] prompt=$label concurrency=$concurrency max_tokens=$max_tokens" | tee -a "$LOG_FILE"
  python "$ROOT_DIR/src/monitor_gpu.py" --output "$gpu_log" --interval 1.0 >/dev/null 2>&1 &
  gpu_pid=$!
  set +e
  python "$ROOT_DIR/src/benchmark_client.py" --config "$CONFIG_PATH" --dataset-path "$dataset_path" --prompt-token-target "$prompt_tokens" --concurrency "$concurrency" --max-tokens "$max_tokens" --gpu-metrics-path "$gpu_log" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  kill "$gpu_pid" >/dev/null 2>&1 || true
  wait "$gpu_pid" 2>/dev/null || true
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "[matrix] failed: prompt_tokens=$prompt_tokens concurrency=$concurrency max_tokens=$max_tokens" | tee -a "$LOG_FILE"
  fi
done

echo "[matrix] log file: $LOG_FILE"