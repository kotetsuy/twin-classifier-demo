"""合成「双子」データセット生成（route A）。

権利・プライバシー完全クリアな学習/評価データを手元で生成する。実在人物を一切
使わずに、酷似した2アイデンティティ A / B を作り、ImageFolder 形式で出力する:

    data/synthetic/{train,val}/{A,B}/*.png

A と B は同一の「ゲノム」（顔の基本パラメータ）を共有し、HANDOFF が挙げる
見分け手がかり（ほくろ・眉の角度・生え際・輪郭の微妙な左右非対称）だけが
安定して異なる。各画像には撮影ゆらぎ（微小な回転・並進・スケール・明度・
背景色・ノイズ）を加え、同一人物内のばらつきを作る。これにより:

- CNN (M4) は「安定した識別特徴」を学習できる（顔位置の丸暗記では解けない）
- VLM (M3) には難問になりやすい（記事の比較軸になる）
- 難易度は --diff で調整（小さいほど双子らしく難しい）

写実性は route C（Wikimedia の実在双子ギャラリー）が担うので、本生成器は
「ラベル付き・制御可能な識別タスク」を作ることに徹する（漫画的な顔でよい）。

CLI:
    python scripts/make_synthetic_twins.py                 # 既定 train=60 val=20/クラス
    python scripts/make_synthetic_twins.py --n-train 100 --n-val 30 --diff 0.6
    python scripts/make_synthetic_twins.py --out data/synthetic --size 224 --seed 0
"""

from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Genome:
    """1 アイデンティティの顔パラメータ。A/B は数項目だけ違える。"""

    skin_hue: float          # 肌色 HSV 色相
    face_w: float            # 顔幅（正規化 0..1）
    face_h: float            # 顔高
    eye_y: float             # 目の高さ
    eye_dx: float            # 目の間隔（中心からの片側距離）
    eye_r: float             # 目の半径
    brow_tilt_l: float       # 左眉の傾き（+ で外側上がり）— 識別特徴になりうる
    brow_tilt_r: float       # 右眉の傾き
    nose_len: float          # 鼻の長さ
    mouth_w: float           # 口幅
    hairline_y: float        # 生え際の高さ（小さいほど広い額）— 識別特徴
    hair_hue: float          # 髪色
    mole: tuple[float, float] | None  # ほくろ位置（正規化）。None=なし — 識別特徴


def _rng_genome(rng: np.random.Generator) -> Genome:
    """ベースとなる人物ゲノムをランダム生成する。"""
    return Genome(
        skin_hue=rng.uniform(0.03, 0.09),
        face_w=rng.uniform(0.52, 0.60),
        face_h=rng.uniform(0.66, 0.74),
        eye_y=rng.uniform(0.42, 0.46),
        eye_dx=rng.uniform(0.12, 0.15),
        eye_r=rng.uniform(0.045, 0.055),
        brow_tilt_l=rng.uniform(-0.02, 0.02),
        brow_tilt_r=rng.uniform(-0.02, 0.02),
        nose_len=rng.uniform(0.12, 0.16),
        mouth_w=rng.uniform(0.16, 0.22),
        hairline_y=rng.uniform(0.20, 0.26),
        hair_hue=rng.uniform(0.05, 0.11),
        mole=None,
    )


def _make_twin_pair(rng: np.random.Generator, diff: float) -> tuple[Genome, Genome]:
    """ベースから双子 A/B を作る。diff∈(0,1] が大きいほど差が明瞭=易しい。"""
    base = _rng_genome(rng)
    a = replace(base)
    # B は数項目だけ安定して変える（左右非対称の手がかり）。
    b = replace(
        base,
        brow_tilt_l=base.brow_tilt_l + 0.05 * diff,      # B は左眉が上がり気味
        hairline_y=base.hairline_y - 0.03 * diff,        # B は額がやや広い
        mole=(0.62, 0.60),                                # B は右頬にほくろ
    )
    # A にも別の弱い癖を1つ（対称的に難しくする）。
    a = replace(a, brow_tilt_r=base.brow_tilt_r - 0.03 * diff)
    return a, b


def _hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _draw_face(g: Genome, size: int, rng: np.random.Generator) -> Image.Image:
    """ゲノムから1枚の顔画像を描く（撮影ゆらぎ込み）。"""
    S = size
    # --- 撮影ゆらぎ（同一人物内のばらつき）---
    jitter_x = rng.uniform(-0.03, 0.03)
    jitter_y = rng.uniform(-0.03, 0.03)
    scale = rng.uniform(0.94, 1.06)
    angle = rng.uniform(-7, 7)
    bright = rng.uniform(0.88, 1.10)
    bg_v = rng.uniform(0.55, 0.85)
    bg = _hsv(rng.uniform(0, 1), 0.15, bg_v)

    img = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(img)

    def px(nx, ny):  # 正規化座標 -> ピクセル（ジッタ/スケール込み、中心基準）
        cx, cy = 0.5 + jitter_x, 0.52 + jitter_y
        return ((nx - 0.5) * scale + cx) * S, ((ny - 0.5) * scale + cy) * S

    skin = _hsv(g.skin_hue, 0.45, 0.92 * bright)
    skin_edge = _hsv(g.skin_hue, 0.55, 0.62 * bright)
    hair = _hsv(g.hair_hue, 0.6, 0.35 * bright)

    # 髪（顔より一回り大きい楕円）
    hw, hh = g.face_w * 1.18, g.face_h * 1.16
    d.ellipse([*px(0.5 - hw / 2, g.hairline_y - 0.06), *px(0.5 + hw / 2, g.hairline_y + hh)], fill=hair)
    # 顔
    d.ellipse([*px(0.5 - g.face_w / 2, g.hairline_y + 0.02),
               *px(0.5 + g.face_w / 2, g.hairline_y + 0.02 + g.face_h)],
              fill=skin, outline=skin_edge, width=2)

    # 目
    for sgn in (-1, 1):
        ex = 0.5 + sgn * g.eye_dx
        d.ellipse([*px(ex - g.eye_r, g.eye_y - g.eye_r * 0.7),
                   *px(ex + g.eye_r, g.eye_y + g.eye_r * 0.7)], fill=(250, 250, 250), outline=(60, 50, 45))
        d.ellipse([*px(ex - g.eye_r * 0.4, g.eye_y - g.eye_r * 0.4),
                   *px(ex + g.eye_r * 0.4, g.eye_y + g.eye_r * 0.4)], fill=(55, 38, 28))

    # 眉（傾きが左右で違う=識別特徴）
    for sgn, tilt in ((-1, g.brow_tilt_l), (1, g.brow_tilt_r)):
        ex = 0.5 + sgn * g.eye_dx
        by = g.eye_y - g.eye_r * 1.7
        d.line([*px(ex - g.eye_r, by + tilt * sgn), *px(ex + g.eye_r, by - tilt * sgn)],
               fill=_hsv(g.hair_hue, 0.6, 0.30 * bright), width=max(2, S // 70))

    # 鼻
    nx, ny0 = 0.5, g.eye_y + 0.02
    d.line([*px(nx, ny0), *px(nx - 0.015, ny0 + g.nose_len)], fill=skin_edge, width=max(2, S // 80))
    d.line([*px(nx - 0.015, ny0 + g.nose_len), *px(nx + 0.02, ny0 + g.nose_len)], fill=skin_edge, width=max(2, S // 80))

    # 口
    my = ny0 + g.nose_len + 0.06
    d.arc([*px(0.5 - g.mouth_w / 2, my - 0.03), *px(0.5 + g.mouth_w / 2, my + 0.05)],
          15, 165, fill=_hsv(0.0, 0.45, 0.6 * bright), width=max(2, S // 70))

    # ほくろ（あれば識別特徴）
    if g.mole is not None:
        mx, mmy = g.mole
        r = S * 0.012
        cx, cy = px(mx, mmy)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_hsv(g.skin_hue, 0.7, 0.35 * bright))

    # 回転＋わずかなノイズ＋ぼかしで生っぽさを足す
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg)
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def _emit(genome: Genome, out_dir: Path, n: int, rng: np.random.Generator, size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _draw_face(genome, size, rng).save(out_dir / f"{i:04d}.png")


def generate(out: Path, n_train: int, n_val: int, size: int, diff: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = _make_twin_pair(rng, diff)
    for split, n in (("train", n_train), ("val", n_val)):
        for label, g in (("A", a), ("B", b)):
            _emit(g, out / split / label, n, rng, size)
    total = (n_train + n_val) * 2
    print(f"generated {total} images -> {out}/{{train,val}}/{{A,B}}  (size={size}, diff={diff})")
    print("識別手がかり: B=左眉上がり+額広め+右頬ほくろ / A=右眉下がり気味・ほくろなし")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic twin A/B dataset (route A)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--n-train", type=int, default=60, help="クラスあたり train 枚数")
    ap.add_argument("--n-val", type=int, default=20, help="クラスあたり val 枚数")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--diff", type=float, default=0.7, help="A/B の差（0<diff<=1、小さいほど難しい）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, args.n_train, args.n_val, args.size, args.diff, args.seed)


if __name__ == "__main__":
    main()
