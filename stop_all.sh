#!/usr/bin/env bash
# 合成双子デモを一括停止する（start_all.sh の対）。
#   - tmux セッション（GUI / 自動プレイ / start_all が起動した llama-server）を停止
#   - 取り残しの twin_stream / play_twins プロセスを掃除
#
# 既に起動していた（tmux 外の）llama-server は既定では止めない。
# それも止めたい場合: ./stop_all.sh --server
set -uo pipefail

SESSION="${TWIN_DEMO_SESSION:-twin-demo}"
MODE_FILE="${TWIN_DEMO_MODE_FILE:-/tmp/twin_demo_mode}"
KILL_SERVER=0
[ "${1:-}" = "--server" ] && KILL_SERVER=1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION" && echo "[stop] tmux session '$SESSION' を停止"
else
  echo "[stop] tmux session '$SESSION' は無し"
fi

# 取り残し掃除（[t] のブラケットで自己マッチを防ぐ）
pkill -f '[t]win_stream.py'  2>/dev/null && echo "[stop] twin_stream 停止" || true
pkill -f '[p]lay_twins.py'   2>/dev/null && echo "[stop] play_twins 停止"  || true
rm -f "$MODE_FILE"

if [ "$KILL_SERVER" = 1 ]; then
  pkill -f '[l]lama-server' 2>/dev/null && echo "[stop] llama-server 停止" || echo "[stop] llama-server プロセス無し"
else
  echo "[stop] llama-server は据え置き（止めるなら ./stop_all.sh --server）"
fi
echo "[stop] 完了"
