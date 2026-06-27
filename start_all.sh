#!/usr/bin/env bash
# 合成双子デモを tmux で一括起動する。
#   - GUI（合成 val 顔 + A/B ボタン、--loop で連続再生）
#   - 必要なら llama-server（Nemotron VLM）を起動
#   - 自動プレイ: CNN(40枚) -> VLM few-shot explain(6枚)
#
# 取得/クリックは Wayland portal（play_twins）。実行中に GNOME の
# 「画面共有＋操作」許可ダイアログが出るので承認すること（CNN/VLM で各1回）。
#
# 停止は ./stop_all.sh。確認は: tmux attach -t twin-demo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${TWIN_DEMO_SESSION:-twin-demo}"
PY="$ROOT/.venv/bin/python"
PORT="${NEMOTRON_PORT:-8080}"
GEOMETRY="${TWIN_DEMO_GEOMETRY:-1100x900+200+120}"
MODE_FILE="${TWIN_DEMO_MODE_FILE:-/tmp/twin_demo_mode}"
export DISPLAY="${DISPLAY:-:0}"
export TWIN_DEMO_MODE_FILE="$MODE_FILE"
rm -f "$MODE_FILE"  # 前回の残りを消す（GUI 初期表示は「待機中」に）

command -v tmux >/dev/null || { echo "ERROR: tmux が無い"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: venv python が無い: $PY"; exit 1; }

# 既存セッションは作り直す
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "[start] tmux session '$SESSION' を作成"
# --- window 0: GUI（合成双子ストリーム、連続再生）---
tmux new-session -d -s "$SESSION" -n stim
tmux send-keys -t "$SESSION:stim" \
  "cd '$ROOT' && DISPLAY='$DISPLAY' '$PY' scripts/twin_stream.py --loop \
   --geometry '$GEOMETRY' --a-dir data/synthetic/val/A --b-dir data/synthetic/val/B" Enter

# --- VLM サーバ: 未起動なら起動 ---
if curl -sf --max-time 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "[start] llama-server は起動済み（:$PORT）"
else
  echo "[start] llama-server を起動（model ロードに時間がかかる）"
  tmux new-window -t "$SESSION" -n server \
    "cd '$ROOT' && bash scripts/serve_nemotron.sh"
fi

# --- window: 自動プレイ CNN(40) -> VLM(6) ---
# 空のシェルを作って send-keys で投入（ネスト引用符を避ける）。$ROOT/$PORT は
# 今展開、\$(seq ...) は実行時評価。コマンド終了後もペインはシェルとして残る。
tmux new-window -t "$SESSION" -n play
tmux send-keys -t "$SESSION:play" "cd '$ROOT'" Enter
tmux send-keys -t "$SESSION:play" \
  "sleep 3 && \
echo '=== CNN 自動プレイ (40枚): 画面共有ダイアログを承認 ===' && \
'$PY' scripts/play_twins.py --backend cnn --frames 40 --interval 0.7 ; \
echo '=== llama-server を待機 ===' ; \
for i in \$(seq 1 240); do curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done ; \
echo '=== VLM few-shot explain 自動プレイ (6枚): 画面共有ダイアログを承認 ===' && \
'$PY' scripts/play_twins.py --backend nemotron --explain --frames 6 \
   --refs-dir data/synthetic/train --refs-per-class 2 --interval 0.5 ; \
echo '=== デモ完了（停止は ./stop_all.sh）==='" Enter

cat <<EOF

[start] 起動しました。
  - GUI: 合成双子ストリーム（--loop 連続再生）
  - 自動プレイ: CNN(40) -> VLM few-shot explain(6)
  - 実行中に GNOME の「画面共有＋操作」許可が CNN/VLM で各1回出ます。承認後はマウスに触れないでください。

  ライブ確認:  tmux attach -t $SESSION   （切替: Ctrl-b 0/1/2、デタッチ: Ctrl-b d）
  停止:        ./stop_all.sh
EOF
