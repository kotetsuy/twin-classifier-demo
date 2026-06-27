"""リアルタイム判定ループ (M6).

mss で画面の指定領域をキャプチャ → classify で A/B 判定 → 回答位置をクリック、を
繰り返す。HANDOFF の「キャプチャ→判定→操作」自動化の最終像。2 モード:

  - 解説デモモード (--mode explain, 既定): 1 枚ごとに Nemotron VLM の few-shot
        判定＋根拠（必要なら思考トレース）を表示してからクリック。ペースは自分で
        握る（~2-3s/枚）。ネタとしての本命。見本は --refs-dir から読む（形態①）。
  - 速度勝負モード (--mode speed): 学習済み CNN でサブミリ秒判定して連打。
        VLM は使わない（事後解説に回す想定）。

判定結果のクリック先は引数で与える: A→--a-xy, B→--b-xy（画面絶対座標）。
未指定ならキャプチャ領域の左 1/3・右 2/3 の中心を使う（"左右どっち" 回答を想定）。

安全のため既定は --dry-run（クリックせずログのみ）。実クリックは --no-dry-run。
画面取得は --capture で選ぶ: X11 は mss、Wayland(GNOME 等) は xdg-desktop-portal
ScreenCast + PipeWire（auto が自動判別）。pynput でマウス制御、なければ自動 dry-run。

CLI:
    # 解説デモ（見本つき VLM）。領域とクリック先を指定、まずは dry-run で確認:
    python src/realtime.py --mode explain --refs-dir data/synthetic/train \\
        --region 100 100 400 400 --a-xy 300 800 --b-xy 900 800
    # 速度モード（CNN, 実クリック, 30枚で停止）:
    python src/realtime.py --mode speed --no-dry-run --interval 0.1 --max-frames 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# --- 入出力ヘルパ ---------------------------------------------------------

def load_refs(refs_dir: Path, n: int) -> tuple[list, list]:
    """refs_dir/{A,B} の先頭 n 枚ずつを few-shot 見本として返す。"""
    out = {}
    for label in ("A", "B"):
        d = refs_dir / label
        imgs = sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) < n:
            raise SystemExit(f"{d} の見本が足りない（{len(imgs)} < {n}）")
        out[label] = [str(p) for p in imgs[:n]]
    return out["A"], out["B"]


class Clicker:
    """クリック注入。優先順位:
      1. dry_run なら常にログのみ。
      2. injector（capture が持つコンポジタ経由注入。Wayland 用）があればそれ。
      3. なければ pynput(XTEST, X11 用)。どちらも不可なら dry-run にフォールバック。
    """

    def __init__(self, dry_run: bool, injector=None):
        self.injector = injector if (not dry_run and injector is not None) else None
        self.controller = None
        self.button = None
        if not dry_run and self.injector is None:
            try:
                from pynput.mouse import Button, Controller

                self.controller = Controller()
                self.button = Button.left
            except Exception as e:  # noqa: BLE001
                print(f"[warn] pynput 不可（{e}）→ dry-run にフォールバック")
        self.dry_run = self.injector is None and self.controller is None

    def click(self, xy: tuple[int, int]) -> None:
        if self.dry_run:
            print(f"    [dry-run] would click {xy}")
            return
        if self.injector is not None:
            self.injector.click(xy)
            return
        self.controller.position = xy
        self.controller.click(self.button, 1)


def _default_targets(region: tuple[int, int, int, int]) -> tuple[tuple, tuple]:
    """領域の左1/3・右2/3 中心を A/B のクリック先にする。"""
    x, y, w, h = region
    cy = y + h // 2
    return (x + w // 3, cy), (x + 2 * w // 3, cy)


# --- バックエンド（predict 関数を作る）-----------------------------------

def make_predict(args, refs_dir: Path | None):
    """frame(BGR ndarray) -> ("A"|"B", rationale|None) を返す関数を作る。"""
    if args.mode == "speed":
        from classify import classify

        # 初回ロードを温める
        _ = classify(np.zeros((8, 8, 3), np.uint8), backend="cnn")

        def predict(frame):
            return classify(frame, backend="cnn"), None

        return predict

    # explain モード: few-shot VLM（形態①）
    import nemotron_client as nc

    if not nc.ping():
        raise SystemExit("llama-server 未起動。`bash scripts/serve_nemotron.sh` を起動してください。")
    if refs_dir is None:
        raise SystemExit("explain モードは --refs-dir が必須（形態①の見本）。")
    refs_a, refs_b = load_refs(refs_dir, args.refs_per_class)

    def predict(frame):
        if args.explain:
            r = nc.explain(frame, refs_a=refs_a, refs_b=refs_b, think=args.show_thinking)
            extra = r.thinking if args.show_thinking and r.thinking else r.rationale
            return r.answer, extra
        return nc.judge(frame, refs_a=refs_a, refs_b=refs_b), None

    return predict


# --- メインループ ---------------------------------------------------------

def run(args) -> None:
    from screen_capture import open_capture

    region = tuple(args.region) if args.region else None
    a_xy = tuple(args.a_xy) if args.a_xy else None
    b_xy = tuple(args.b_xy) if args.b_xy else None
    refs_dir = Path(args.refs_dir) if args.refs_dir else None

    predict = make_predict(args, refs_dir)
    cap = open_capture(args.capture)
    injector = cap if getattr(cap, "can_inject", False) else None
    clicker = Clicker(args.dry_run, injector=injector)

    try:
        if region is None:  # 既定はキャプチャ対象（モニタ）全体
            region = cap.geometry()
        if a_xy is None or b_xy is None:
            a_xy, b_xy = _default_targets(region)
        print(f"capture={type(cap).__name__}  region={region}  A->click{a_xy}  "
              f"B->click{b_xy}  mode={args.mode}  "
              f"{'DRY-RUN' if clicker.dry_run else 'LIVE-CLICK'}")

        frame_no = 0
        try:
            while args.max_frames == 0 or frame_no < args.max_frames:
                frame_no += 1
                shot = cap.grab(region)
                t = time.time()
                answer, extra = predict(shot)
                dt = (time.time() - t) * 1000
                target = a_xy if answer == "A" else b_xy
                print(f"[{frame_no}] {answer} ({dt:.0f}ms) -> click {target}")
                if extra:
                    print(f"    {extra.strip()[:300]}")
                clicker.click(target)
                if args.interval > 0:
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n停止しました。")
    finally:
        cap.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Realtime capture->classify->click loop (M6)")
    ap.add_argument("--mode", choices=["explain", "speed"], default="explain")
    ap.add_argument("--capture", choices=["auto", "mss", "portal", "pipewire"], default="auto",
                    help="画面取得/操作方式（auto: Wayland は portal / それ以外 mss）")
    ap.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="キャプチャ領域（既定: プライマリモニタ全体）")
    ap.add_argument("--a-xy", type=int, nargs=2, metavar=("X", "Y"), help="A 回答のクリック先")
    ap.add_argument("--b-xy", type=int, nargs=2, metavar=("X", "Y"), help="B 回答のクリック先")
    ap.add_argument("--refs-dir", default=None, help="few-shot 見本ディレクトリ（A/ B/ を含む）")
    ap.add_argument("--refs-per-class", type=int, default=2)
    ap.add_argument("--explain", dest="explain", action="store_true", default=True,
                    help="根拠も表示（explain モード既定）")
    ap.add_argument("--no-explain", dest="explain", action="store_false",
                    help="判定のみ（explain モードでも judge を使い高速化）")
    ap.add_argument("--show-thinking", action="store_true", help="思考トレースを表示（収束しない場合あり）")
    ap.add_argument("--interval", type=float, default=2.0, help="判定間隔 秒")
    ap.add_argument("--max-frames", type=int, default=0, help="最大判定回数（0=Ctrl-Cまで）")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                    help="クリックせずログのみ（既定・安全側）")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="実際にクリックする")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
