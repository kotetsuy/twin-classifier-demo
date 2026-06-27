"""ウィンドウ・キャプチャ＋注入クリックの判定ドライバ (M6 デバッグ).

twin_stream が報告した絶対座標を受け取り、portal セッション 1 つで:
  (2) 指定ウィンドウ領域 --win をキャプチャして PNG 保存（非黒かを表示）
  (3) --a-xy → --b-xy の順に注入クリックを撃つ
を行う。クリックが効いたかは twin_stream 側のログ（score 増加）で判定する。

  python scripts/mouse_test.py --win X Y W H --a-xy AX AY --b-xy BX BY

許可ダイアログ（画面共有＋操作）を承認後、マウスには触れないこと。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="window capture + injected click test")
    ap.add_argument("--win", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    ap.add_argument("--a-xy", type=int, nargs=2, metavar=("X", "Y"), required=True)
    ap.add_argument("--b-xy", type=int, nargs=2, metavar=("X", "Y"), required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "window_capture.png"))
    ap.add_argument("--pause", type=float, default=1.5, help="クリック間の待ち（観察用）")
    args = ap.parse_args()

    from screen_capture import PortalCapture
    cap = PortalCapture()
    try:
        # (2) ウィンドウ領域キャプチャ
        x, y, w, h = args.win
        frame = cap.grab((x, y, w, h))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            cv2.imwrite(args.out, frame)
        except Exception:  # noqa: BLE001
            from PIL import Image
            Image.fromarray(frame[:, :, ::-1]).save(args.out)
        print(f"[cap] window {w}x{h} mean={frame.mean():.1f} "
              f"{'BLACK!' if frame.max() == 0 else 'OK non-black'} -> {args.out}")

        # (3) 注入クリック A -> B
        for label, xy in (("A", tuple(args.a_xy)), ("B", tuple(args.b_xy))):
            print(f"[inject] click {label} @ {xy}")
            cap.click(xy)
            time.sleep(args.pause)
        print("[done] twin_stream 側のログで score が増えていれば注入クリック成功。")
    finally:
        cap.close()


if __name__ == "__main__":
    main()
