#!/usr/bin/env bash
# Nemotron 3 Nano Omni を ROCm llama.cpp の llama-server で常駐起動する。
# OpenAI 互換 API を :8080 に出す。nemotron_client.py がこれを叩く。
#
# パスは環境変数で上書き可能（既定は本マシンの実在パス）。
set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${NEMOTRON_MODEL:-$HOME/nemotron-3/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf}"
MMPROJ="${NEMOTRON_MMPROJ:-$HOME/nemotron-3/mmproj-F16.gguf}"
HOST="${NEMOTRON_HOST:-0.0.0.0}"
PORT="${NEMOTRON_PORT:-8080}"
CTX="${NEMOTRON_CTX:-8192}"

for f in "$LLAMA_SERVER" "$MODEL" "$MMPROJ"; do
  if [ ! -e "$f" ]; then
    echo "ERROR: not found: $f" >&2
    echo "       set LLAMA_SERVER / NEMOTRON_MODEL / NEMOTRON_MMPROJ to override." >&2
    exit 1
  fi
done

echo "serving Nemotron on $HOST:$PORT (ctx=$CTX, all layers on GPU)"
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -ngl 99 -c "$CTX" --host "$HOST" --port "$PORT"
