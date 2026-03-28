#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH="$ROOT_DIR/configs/qwen25_1p5b_local.json"
LOG_DIR="$ROOT_DIR/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
if ! command -v python >/dev/null 2>&1; then
  echo "[error] python not found in current shell." >&2
  echo "[error] please activate your vllm312 virtual environment first." >&2
  exit 1
fi


readarray -t CFG < <(python - <<'PY' "$CONFIG_PATH"
import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
print(cfg['model']['path'])
print(cfg['model']['served_model_name'])
print(str(cfg['model']['trust_remote_code']).lower())
print(cfg['server']['host'])
print(cfg['server']['port'])
print(cfg['server']['dtype'])
print(cfg['server']['max_model_len'])
print(cfg['server']['gpu_memory_utilization'])
print(cfg['server']['tensor_parallel_size'])
print(cfg['server']['max_num_seqs'])
PY
)

MODEL_PATH=${CFG[0]}
SERVED_NAME=${CFG[1]}
TRUST_REMOTE_CODE=${CFG[2]}
HOST=${CFG[3]}
PORT=${CFG[4]}
DTYPE=${CFG[5]}
MAX_MODEL_LEN=${CFG[6]}
GPU_MEM=${CFG[7]}
TP_SIZE=${CFG[8]}
MAX_NUM_SEQS=${CFG[9]}

CMD=(python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --tensor-parallel-size "$TP_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS")

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

echo "[server] log file: $LOG_FILE"
echo "[server] endpoint: http://$HOST:$PORT/v1/completions"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"