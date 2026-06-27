"""ザ・たっち顔クロップの A/B 手動ラベリング GUI.

extract_faces.py が出した faces/manifest.csv の各クロップを 1 件ずつ表示し、
A=たくや / B=かずや / S=不明(skip) を付ける。識別の手掛かりとして、整列クロップに加えて
元写真（該当顔を赤枠でハイライト）も並べて出す（キャプションや左右関係が手掛かりになる）。

ヒント: たくや=鼻の横にほくろ・丸顔・父似 / かずや=面長・母似。

キー:  A=たくや  B=かずや  S/Space=skip  U=undo  Q/Esc=保存して終了
結果は faces/labels.csv に逐次保存（再実行で続きから）。

  python scripts/label_faces.py
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "data" / "raw" / "the_touch"
FACES = SRC_ROOT / "faces"
LABELS = FACES / "labels.csv"
CROP_PX, SRC_PX = 340, 460


def load_manifest():
    with (FACES / "manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_labels():
    if not LABELS.exists():
        return {}
    with LABELS.open(newline="") as f:
        return {r["crop"]: r["label"] for r in csv.DictReader(f)}


class Labeler:
    def __init__(self):
        self.items = load_manifest()
        self.labels = load_labels()
        # 未ラベルから開始
        self.order = list(range(len(self.items)))
        self.pos = next((i for i, idx in enumerate(self.order)
                         if self.items[idx]["crop"] not in self.labels), len(self.order))
        self.history: list[int] = []

        self.root = tk.Tk()
        self.root.title("ザ・たっち A/B ラベリング")
        self.root.configure(bg="#1e1e1e")
        self.info = tk.Label(self.root, bg="#1e1e1e", fg="#e6e6e6",
                             font=("TkDefaultFont", 14), justify="left")
        self.info.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=6)
        self.src_lbl = tk.Label(self.root, bg="#1e1e1e")
        self.src_lbl.grid(row=1, column=0, padx=10, pady=6)
        self.crop_lbl = tk.Label(self.root, bg="#1e1e1e")
        self.crop_lbl.grid(row=1, column=1, padx=10, pady=6)
        hint = ("A=たくや(鼻横ほくろ/丸顔)   B=かずや(面長)   "
                "S/Space=skip   U=undo   Q/Esc=保存終了")
        tk.Label(self.root, text=hint, bg="#1e1e1e", fg="#9ad29a",
                 font=("TkDefaultFont", 13)).grid(row=2, column=0, columnspan=2, pady=8)

        for k in ("a", "A"):
            self.root.bind(k, lambda e: self.label("A"))
        for k in ("b", "B"):
            self.root.bind(k, lambda e: self.label("B"))
        for k in ("s", "S", "<space>"):
            self.root.bind(k, lambda e: self.label("skip"))
        for k in ("u", "U"):
            self.root.bind(k, lambda e: self.undo())
        for k in ("q", "Q", "<Escape>"):
            self.root.bind(k, lambda e: self.quit())
        self._refs = []

    def counts(self):
        a = sum(1 for v in self.labels.values() if v == "A")
        b = sum(1 for v in self.labels.values() if v == "B")
        s = sum(1 for v in self.labels.values() if v == "skip")
        return a, b, s

    def show(self):
        self._refs.clear()
        if self.pos >= len(self.order):
            self.info.config(text="全件ラベル済み。Q で終了。")
            self.src_lbl.config(image="")
            self.crop_lbl.config(image="")
            return
        item = self.items[self.order[self.pos]]
        a, b, s = self.counts()
        self.info.config(text=(f"[{self.pos+1}/{len(self.order)}]  {item['source']}  "
                               f"(weak={item['weak_label']}, eye={item['eye_dist']})    "
                               f"A:{a}  B:{b}  skip:{s}"))
        # crop
        crop = Image.open(FACES / item["crop"]).convert("RGB").resize((CROP_PX, CROP_PX))
        cimg = ImageTk.PhotoImage(crop)
        self.crop_lbl.config(image=cimg)
        self._refs.append(cimg)
        # source with bbox
        sp = SRC_ROOT / item["source"]
        if sp.exists():
            src = Image.open(sp).convert("RGB")
            sw, sh = src.size
            scale = SRC_PX / max(sw, sh)
            src2 = src.resize((int(sw * scale), int(sh * scale)))
            d = ImageDraw.Draw(src2)
            bx, by, bw, bh = (float(item[k]) for k in ("bx", "by", "bw", "bh"))
            d.rectangle([bx*scale, by*scale, (bx+bw)*scale, (by+bh)*scale],
                        outline="#ff3030", width=4)
            simg = ImageTk.PhotoImage(src2)
            self.src_lbl.config(image=simg)
            self._refs.append(simg)
        else:
            self.src_lbl.config(image="")

    def label(self, val):
        if self.pos >= len(self.order):
            return
        crop = self.items[self.order[self.pos]]["crop"]
        self.labels[crop] = val
        self.history.append(self.pos)
        self.save()
        self.pos += 1
        self.show()

    def undo(self):
        if not self.history:
            return
        self.pos = self.history.pop()
        crop = self.items[self.order[self.pos]]["crop"]
        self.labels.pop(crop, None)
        self.save()
        self.show()

    def save(self):
        with LABELS.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["crop", "label"])
            for c, v in self.labels.items():
                w.writerow([c, v])

    def quit(self):
        self.save()
        a, b, s = self.counts()
        print(f"保存: A(たくや)={a} B(かずや)={b} skip={s} -> {LABELS}")
        self.root.destroy()

    def run(self):
        self.show()
        self.root.mainloop()


def main() -> None:
    if not (FACES / "manifest.csv").exists():
        sys.exit("先に extract_faces.py を実行してください。")
    Labeler().run()


if __name__ == "__main__":
    main()
