"""双子ストリーム提示プログラム（realtime 通し確認 M6 用の刺激ステージ）.

realtime ループに「クリックして当てさせる相手」を与えるための画面。Tk のウィンドウに
1 枚ずつ顔を出し、左下に [A]・右下に [B] の回答ボタン領域を描く。クリック（人手でも
注入でも）された座標を <Button-1> ハンドラが受け取り、提示中の正解ラベルと突き合わせて
スコアを更新し、次の顔へ進む。

既定はウィンドウ表示（--geometry WxH+X+Y）。フルスクリーンは入力を奪うので既定では使わない
（--fullscreen で従来動作）。WM がウィンドウ位置を最終決定するため、起動後に実測した
キャンバスの絶対原点から A/B ボタン・顔領域の「画面絶対座標」を算出し stderr に出す。
その座標をそのまま realtime / mouse_test の --region/--a-xy/--b-xy に渡せる。

  python scripts/twin_stream.py --geometry 1100x900+200+120

q または Esc で終了。--auto-advance 秒 でクリックなしでも自動送り。
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def parse_geometry(geom: str):
    m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)$", geom)
    if not m:
        raise SystemExit(f"--geometry は WxH+X+Y 形式: {geom}")
    return tuple(int(g) for g in m.groups())  # w, h, x, y


def layout(sw, sh):
    fw = fh = min(sw, sh) * 4 // 10              # 顔は短辺の 40%
    face = (sw // 2 - fw // 2, sh // 6, fw, fh)  # 上寄り中央
    bw, bh = sw * 3 // 10, sh * 7 // 40          # ボタン寸法
    by = sh * 70 // 100
    a_rect = (sw // 4 - bw // 2, by, bw, bh)
    b_rect = (sw * 3 // 4 - bw // 2, by, bw, bh)
    return face, a_rect, b_rect


def center(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


class Stream:
    def __init__(self, args):
        a_imgs = list_images(Path(args.a_dir))
        b_imgs = list_images(Path(args.b_dir))
        if not a_imgs or not b_imgs:
            raise SystemExit(f"画像が見つからない: A={len(a_imgs)} B={len(b_imgs)}")
        self.pool = [(p, "A") for p in a_imgs] + [(p, "B") for p in b_imgs]
        random.seed(args.seed)
        random.shuffle(self.pool)
        self.auto = args.auto_advance

        self.root = tk.Tk()
        self.root.title("twin_stream")
        if args.fullscreen:
            self.root.attributes("-fullscreen", True)
            self.root.update_idletasks()
            self.sw = self.root.winfo_screenwidth()
            self.sh = self.root.winfo_screenheight()
        else:
            w, h, x, y = parse_geometry(args.geometry)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.sw, self.sh = w, h

        self.face_rect, self.a_rect, self.b_rect = layout(self.sw, self.sh)
        self.canvas = tk.Canvas(self.root, width=self.sw, height=self.sh,
                                bg="#141414", highlightthickness=0)
        self.canvas.place(x=0, y=0)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("q", lambda e: self.quit())

        self.idx = 0
        self.score = 0
        self.total = 0
        self.last = "-"
        self._imgref = None

    def cur(self):
        return self.pool[self.idx % len(self.pool)]

    def report(self):
        """マップ後の実測絶対座標を出す（realtime / mouse_test 用）。"""
        self.root.update_idletasks()
        ox, oy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        fx, fy, fw, fh = self.face_rect
        face_abs = (ox + fx, oy + fy, fw, fh)
        a_xy = (ox + center(self.a_rect)[0], oy + center(self.a_rect)[1])
        b_xy = (ox + center(self.b_rect)[0], oy + center(self.b_rect)[1])
        win_abs = (ox, oy, self.sw, self.sh)
        print("=== 実測 絶対座標（このまま realtime / mouse_test に渡せる）===", file=sys.stderr)
        print(f"  WIN_ABS   {win_abs[0]} {win_abs[1]} {win_abs[2]} {win_abs[3]}", file=sys.stderr)
        print(f"  FACE_ABS  {face_abs[0]} {face_abs[1]} {face_abs[2]} {face_abs[3]}", file=sys.stderr)
        print(f"  A_XY      {a_xy[0]} {a_xy[1]}", file=sys.stderr)
        print(f"  B_XY      {b_xy[0]} {b_xy[1]}", file=sys.stderr)
        print(f"  realtime: python src/realtime.py --mode speed --capture portal --no-dry-run "
              f"--interval 1.0 --max-frames {len(self.pool)} "
              f"--region {face_abs[0]} {face_abs[1]} {face_abs[2]} {face_abs[3]} "
              f"--a-xy {a_xy[0]} {a_xy[1]} --b-xy {b_xy[0]} {b_xy[1]}", file=sys.stderr)
        print(f"  mouse_test: python scripts/mouse_test.py "
              f"--win {win_abs[0]} {win_abs[1]} {win_abs[2]} {win_abs[3]} "
              f"--a-xy {a_xy[0]} {a_xy[1]} --b-xy {b_xy[0]} {b_xy[1]}", file=sys.stderr, flush=True)

    def render(self):
        c = self.canvas
        c.delete("all")
        path, truth = self.cur()
        fx, fy, fw, fh = self.face_rect
        img = Image.open(path).convert("RGB").resize((fw, fh))
        self._imgref = ImageTk.PhotoImage(img)
        c.create_image(fx, fy, anchor="nw", image=self._imgref)
        c.create_rectangle(fx - 3, fy - 3, fx + fw + 3, fy + fh + 3, outline="#5a5a5a", width=3)
        # 検出用マゼンタ枠（play_twins が顔領域を色検出で切り出すための目印）
        c.create_rectangle(fx - 8, fy - 8, fx + fw + 8, fy + fh + 8, outline="#ff00ff", width=4)
        c.create_text(fx, fy - 22, anchor="nw", text=f"truth={truth}",
                      fill="#b4b4b4", font=("TkDefaultFont", 18))
        for rect, label, col in ((self.a_rect, "A", "#4682e6"), (self.b_rect, "B", "#46c878")):
            x, y, w, h = rect
            c.create_rectangle(x, y, x + w, y + h, fill=col, outline="")
            c.create_text(x + w // 2, y + h // 2, text=label, fill="white",
                          font=("TkDefaultFont", 40, "bold"))
        acc = f"{100*self.score/self.total:.0f}%" if self.total else "-"
        c.create_text(20, 28, anchor="nw",
                      text=f"score {self.score}/{self.total} ({acc})  last: {self.last}",
                      fill="#e6e6e6", font=("TkDefaultFont", 18))

    def advance(self):
        self.idx += 1
        if self.idx >= len(self.pool):
            self.idx = len(self.pool)
            print(f"[stim] 全 {len(self.pool)} 枚を出し切りました。", flush=True)
            self.render_end()
            return
        self.render()

    def render_end(self):
        c = self.canvas
        c.delete("all")
        acc = f"{100*self.score/self.total:.0f}%" if self.total else "-"
        c.create_text(self.sw // 2, self.sh // 2, text=f"DONE  {self.score}/{self.total} ({acc})",
                      fill="#e6e6e6", font=("TkDefaultFont", 36, "bold"))

    def on_click(self, ev):
        x, y = ev.x, ev.y
        hit = None
        for rect, label in ((self.a_rect, "A"), (self.b_rect, "B")):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                hit = label
        if hit is None or self.idx >= len(self.pool):
            print(f"[stim] click@({x},{y}) ボタン外", flush=True)
            return
        truth = self.cur()[1]
        ok = hit == truth
        self.score += int(ok)
        self.total += 1
        self.last = f"click {hit} vs truth {truth} -> {'OK' if ok else 'NG'}"
        print(f"[stim] {self.total:>3}  {self.last}  running {self.score}/{self.total}",
              flush=True)
        self.advance()

    def _auto_tick(self):
        if self.idx < len(self.pool):
            self.advance()
            self.root.after(int(self.auto * 1000), self._auto_tick)

    def quit(self):
        print(f"[stim] 終了 score {self.score}/{self.total}", flush=True)
        self.root.destroy()

    def run(self):
        self.render()
        self.root.after(400, self.report)
        if self.auto > 0:
            self.root.after(int(self.auto * 1000) + 400, self._auto_tick)
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Twin stream stimulus for realtime through-test")
    ap.add_argument("--a-dir", default="data/synthetic/val/A")
    ap.add_argument("--b-dir", default="data/synthetic/val/B")
    ap.add_argument("--geometry", default="1100x900+200+120", help="WxH+X+Y（ウィンドウ表示）")
    ap.add_argument("--fullscreen", action="store_true", help="フルスクリーン（入力を奪うので非推奨）")
    ap.add_argument("--auto-advance", type=float, default=0.0,
                    help="秒。>0 ならクリックが無くても自動で次へ")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Stream(args).run()


if __name__ == "__main__":
    main()
