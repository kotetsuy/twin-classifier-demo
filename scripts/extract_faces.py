"""収集写真から顔を検出・整列してクロップを書き出す（A/B ラベリングの前処理）.

ザ・たっちはコンビ写真が大半なので 1 枚から複数顔を取り出す（face_align は num_faces=1
なので本スクリプトは num_faces を増やした landmarker を使う）。各顔を face_align と同じ
相似変換で 224x224 に整列（左右反転なし＝非対称を保存）し、`faces/` に保存。元写真上の
顔 bbox も manifest に残す（ラベリング時に元画像のどの顔かを示すため）。

  python scripts/extract_faces.py            # data/raw/the_touch/{both,takuya,kazuya} を処理
  python scripts/extract_faces.py --min-eye 22  # 目間距離(px)の下限で低解像を除外
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import face_align as fa  # noqa: E402

SRC_ROOT = ROOT / "data" / "raw" / "the_touch"
GROUPS = ("both", "takuya", "kazuya")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_FIELDS = ["crop", "source", "weak_label", "face_index",
                   "eye_dist", "bx", "by", "bw", "bh"]


def make_landmarker(num_faces: int):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    base = mp_python.BaseOptions(model_asset_path=str(fa._ensure_model()))
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=num_faces)
    return vision.FaceLandmarker.create_from_options(opts)


def faces_in(landmarker, img_bgr):
    """各顔の (left_eye, right_eye, bbox) を返す。"""
    import mediapipe as mp
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_img)
    out = []
    for lms in res.face_landmarks:
        a = np.array([lms[fa._IRIS_A].x * w, lms[fa._IRIS_A].y * h])
        b = np.array([lms[fa._IRIS_B].x * w, lms[fa._IRIS_B].y * h])
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        xs = [p.x * w for p in lms]
        ys = [p.y * h for p in lms]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        out.append((left, right, bbox))
    # 左→右の順で安定化
    out.sort(key=lambda t: t[0][0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract aligned face crops for labeling")
    ap.add_argument("--num-faces", type=int, default=4)
    ap.add_argument("--min-eye", type=float, default=18.0, help="目間距離(px)の下限")
    args = ap.parse_args()

    out_dir = SRC_ROOT / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarker = make_landmarker(args.num_faces)
    rows = []
    n_img = n_face = 0
    for label in GROUPS:
        d = SRC_ROOT / label
        if not d.is_dir():
            continue
        for src in sorted(p for p in d.glob("*") if p.suffix.lower() in IMAGE_EXTS):
            img = cv2.imread(str(src))
            if img is None:
                continue
            n_img += 1
            faces = faces_in(landmarker, img)
            for i, (le, re, bbox) in enumerate(faces):
                eye_dist = float(np.hypot(*(re - le)))
                if eye_dist < args.min_eye:
                    continue
                crop = fa.align_from_eyes(img, le, re)
                cname = f"{label}__{src.stem}__f{i}.png"
                cv2.imwrite(str(out_dir / cname), crop)
                rows.append({
                    "crop": cname, "source": f"{label}/{src.name}", "weak_label": label,
                    "face_index": i, "eye_dist": round(eye_dist, 1),
                    "bx": bbox[0], "by": bbox[1], "bw": bbox[2], "bh": bbox[3],
                })
                n_face += 1
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{n_img} 枚から {n_face} 顔を抽出 -> {out_dir}")
    print(f"manifest: {out_dir/'manifest.csv'}")


if __name__ == "__main__":
    main()
