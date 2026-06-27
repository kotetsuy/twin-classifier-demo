"""labels.csv（手動ラベル）から data/the_touch/{train,val}/{A,B} を作る.

A=たくや / B=かずや のラベル済みクロップ（整列済み 224x224）を train/val に分割コピーする。
skip は除外。出力先は実写データセットのルート data/the_touch（合成 data/synthetic と並列）。
TWIN_DATASET=the_touch スイッチでこのデータが学習/評価/分類に使われる（src/data_config.py）。
既存の出力は --clean で消してから作る。

  python scripts/build_ab_split.py --val-ratio 0.2 --seed 0 --clean
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "data" / "raw" / "the_touch" / "faces"
DST_ROOT = ROOT / "data" / "the_touch"  # 実写データセット（data/synthetic と並列）


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/val A/B from labels.csv")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="既存の data/the_touch/{train,val}/{A,B} を消去してから作る")
    args = ap.parse_args()

    labels_path = FACES / "labels.csv"
    if not labels_path.exists():
        raise SystemExit("labels.csv が無い。先に scripts/label_faces.py でラベル付けを。")
    with labels_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ("A", "B")]

    by = {"A": [], "B": []}
    for r in rows:
        by[r["label"]].append(r["crop"])

    rng = random.Random(args.seed)
    made = {}
    for split in ("train", "val"):
        for cls in ("A", "B"):
            d = DST_ROOT / split / cls
            if args.clean and d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    for cls in ("A", "B"):
        crops = by[cls][:]
        rng.shuffle(crops)
        n_val = max(1, int(len(crops) * args.val_ratio)) if crops else 0
        val, train = crops[:n_val], crops[n_val:]
        for split, names in (("train", train), ("val", val)):
            for name in names:
                shutil.copy(FACES / name, DST_ROOT / split / cls / name)
            made[(split, cls)] = len(names)

    print("=== A/B データセット作成 ===  (A=たくや, B=かずや)")
    for split in ("train", "val"):
        print(f"  {split}: A={made[(split,'A')]}  B={made[(split,'B')]}")
    total = sum(made.values())
    print(f"  合計 {total} 枚 -> data/the_touch/{{train,val}}/{{A,B}}"
          "（TWIN_DATASET=the_touch で使用）")
    if total < 40:
        print("  ※ 枚数が少ないと CNN は過学習しやすい（PROGRESSの実測: 40枚/クラスで val 50%）。"
              "VLM few-shot 照合の方が少数で機能する。")


if __name__ == "__main__":
    main()
