"""双子ストリーム自動プレイ（realtime 通し確認 M6 本体・実機 Wayland）.

twin_stream の画面に対し、portal の 1 セッションで「取得＋クリック注入」を同一座標系で行う。
WM のウィンドウ配置に依存しないよう、クリック先（A=青/B=緑ボタン）と顔領域（マゼンタ枠）は
キャプチャ画面から色検出して校正する。あとは:

    grab(顔領域) -> classify(backend) -> 正解側ボタンへ注入クリック -> 次の顔へ

を繰り返す。stim 側のスコアと突き合わせれば end-to-end の正しさが分かる。

  # CNN 速度モード（自己完結・~10ms/枚）
  python scripts/play_twins.py --backend cnn --frames 40
  # VLM 解説モード（few-shot・根拠表示。server 起動要）
  python scripts/play_twins.py --backend nemotron --explain --frames 6 \
      --refs-dir data/synthetic/train --refs-per-class 2

許可ダイアログ（画面共有＋操作）を承認後、マウスに触れないこと。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# GUI(twin_stream) にどの判定系かを伝える状態ファイル（既定パスは両者で一致）。
DEFAULT_MODE_FILE = os.environ.get("TWIN_DEMO_MODE_FILE", "/tmp/twin_demo_mode")

# Tk の塗り色（BGR）。A=#4682e6, B=#46c878, 顔枠=#ff00ff
BLUE = np.array([230, 130, 70])
GREEN = np.array([120, 200, 70])
MAGENTA = np.array([255, 0, 255])
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _mask(frame, bgr, tol):
    return np.abs(frame.astype(int) - bgr.astype(int)).sum(axis=2) < tol


def centroid(frame, bgr, tol=40):
    ys, xs = np.where(_mask(frame, bgr, tol))
    if len(xs) < 200:
        raise SystemExit(f"色 {bgr.tolist()} のボタンが見つからない（{len(xs)}px）。刺激は出ている？")
    return (int(xs.mean()), int(ys.mean()))


def color_bbox(frame, bgr, tol=80):
    ys, xs = np.where(_mask(frame, bgr, tol))
    if len(xs) < 50:
        raise SystemExit(f"色 {bgr.tolist()} の顔枠が見つからない（{len(xs)}px）。")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def calibrate(cap, inset=12):
    f = cap.grab(None)
    a_xy = centroid(f, BLUE)
    b_xy = centroid(f, GREEN)
    x0, y0, x1, y1 = color_bbox(f, MAGENTA)
    face = (x0 + inset, y0 + inset, (x1 - x0) - 2 * inset, (y1 - y0) - 2 * inset)
    return a_xy, b_xy, face


def load_refs(refs_dir: Path, n: int):
    out = {}
    for label in ("A", "B"):
        imgs = sorted(p for p in (refs_dir / label).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) < n:
            raise SystemExit(f"{refs_dir/label} の見本不足（{len(imgs)} < {n}）")
        out[label] = [str(p) for p in imgs[:n]]
    return out["A"], out["B"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-play twin_stream via portal capture+inject")
    ap.add_argument("--backend", choices=["cnn", "nemotron"], default="cnn")
    ap.add_argument("--explain", action="store_true", help="nemotron で根拠も表示")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--interval", type=float, default=0.8, help="クリック後・次グラブまでの待ち")
    ap.add_argument("--refs-dir", default="data/synthetic/train")
    ap.add_argument("--refs-per-class", type=int, default=2)
    ap.add_argument("--mode-file", default=DEFAULT_MODE_FILE,
                    help="GUI に現在モード(CNN/VLM)を知らせる状態ファイル")
    args = ap.parse_args()

    # GUI のカウンタ振り分け用に現在モードを通知（cnn=CNN / nemotron=VLM）
    tag = "VLM" if args.backend == "nemotron" else "CNN"
    try:
        Path(args.mode_file).write_text(tag)
    except OSError as e:  # noqa: BLE001
        print(f"[warn] mode-file 書込失敗（{e}）", file=sys.stderr)

    from screen_capture import PortalCapture

    refs_a = refs_b = None
    if args.backend == "nemotron":
        import nemotron_client as nc
        if not nc.ping():
            raise SystemExit("llama-server 未起動。bash scripts/serve_nemotron.sh を。")
        refs_a, refs_b = load_refs(Path(args.refs_dir), args.refs_per_class)

    from classify import classify
    if args.backend == "cnn":
        classify(np.zeros((8, 8, 3), np.uint8), backend="cnn")  # warm

    cap = PortalCapture()
    try:
        a_xy, b_xy, face = calibrate(cap)
        print(f"[calib] A->click{a_xy}  B->click{b_xy}  face_region={face}", flush=True)

        agree = 0
        for i in range(1, args.frames + 1):
            shot = cap.grab(face)
            t = time.time()
            if args.backend == "nemotron" and args.explain:
                import nemotron_client as nc
                r = nc.explain(shot, refs_a=refs_a, refs_b=refs_b)
                ans, extra = r.answer, r.rationale
            else:
                ans = classify(shot, backend=args.backend, refs_a=refs_a, refs_b=refs_b)
                extra = None
            dt = (time.time() - t) * 1000
            target = a_xy if ans == "A" else b_xy
            print(f"[{i:>3}] {ans} ({dt:.0f}ms) -> click {target}", flush=True)
            if extra:
                print(f"      {extra.strip()[:280]}", flush=True)
            cap.click(target)
            time.sleep(args.interval)
        print(f"[done] {args.frames} 枚を自動プレイ。stim 側スコアと突き合わせて確認。", flush=True)
    finally:
        cap.close()


if __name__ == "__main__":
    main()
