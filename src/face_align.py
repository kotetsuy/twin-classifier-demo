"""顔アライメント (M2).

両目（虹彩中心）を基準に相似変換（回転＋等倍スケール＋並進）で顔を整列し、
224x224 に正規化する。

双子の識別は微細な左右非対称が手がかりになるため、左右反転は一切行わない。
相似変換は det(R) = s^2 > 0 で鏡像を生まないので、非対称が保存される。

検出は mediapipe FaceMesh（refine_landmarks=True の虹彩ランドマーク）を使う。
幾何変換は検出から切り離した純粋関数 `similarity_transform` に分離してあり、
顔画像なしでも数値的に検証できる。

CLI:
    python src/face_align.py --selftest
    python src/face_align.py --input data/train --output data/train_aligned
    python src/face_align.py --input face.jpg --output aligned.png
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# mediapipe Tasks の FaceLandmarker モデル。リポジトリには含めず初回に取得する。
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(
    os.environ.get("FACE_LANDMARKER_MODEL", Path(__file__).resolve().parent.parent / ".models" / "face_landmarker.task")
)

OUTPUT_SIZE = (224, 224)  # (W, H)
# 整列後に左目（画像左側の目）を置く正規化座標 (x, y) ∈ [0, 1]。
# 右目は水平対称の (1 - x, y) に置く。inter-ocular ≈ 0.30 * W。
DESIRED_LEFT_EYE = (0.35, 0.40)

# mediapipe FaceMesh の虹彩中心ランドマーク（refine_landmarks=True で有効）。
# 468 = 一方の虹彩中心, 473 = もう一方。左右は x 座標で判定するので割り当ては任意。
_IRIS_A = 468
_IRIS_B = 473

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_landmarker = None  # 遅延初期化（mediapipe のロードは重い）


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading FaceLandmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(_ensure_model()))
        opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def detect_eye_centers(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """画像から両目の虹彩中心をピクセル座標で返す。

    戻り値は (left_eye_px, right_eye_px)。left は画像左側（x が小さい方）の目。
    顔が検出できなければ None。
    """
    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    a = np.array([lms[_IRIS_A].x * w, lms[_IRIS_A].y * h], dtype=np.float64)
    b = np.array([lms[_IRIS_B].x * w, lms[_IRIS_B].y * h], dtype=np.float64)
    # 画像左側（x 小）を left に。鏡像化しないため順序のみ入れ替える。
    return (a, b) if a[0] <= b[0] else (b, a)


def similarity_transform(
    src_left: np.ndarray,
    src_right: np.ndarray,
    dst_left: np.ndarray,
    dst_right: np.ndarray,
) -> np.ndarray:
    """2 組の対応点から相似変換行列 (2x3) を解析的に求める。

    回転・等倍スケール・並進のみ（剪断・鏡像なし）。det(R) = s^2 > 0。
    """
    p1 = np.asarray(src_left, dtype=np.float64)
    p2 = np.asarray(src_right, dtype=np.float64)
    q1 = np.asarray(dst_left, dtype=np.float64)
    q2 = np.asarray(dst_right, dtype=np.float64)

    v = p2 - p1
    w = q2 - q1
    nv = float(np.hypot(*v))
    nw = float(np.hypot(*w))
    if nv < 1e-9:
        raise ValueError("src eyes coincide; cannot solve transform")

    scale = nw / nv
    # v を w に重ねる回転角。2D 外積で sin、内積で cos。
    cos = float(np.dot(v, w)) / (nv * nw)
    sin = float(v[0] * w[1] - v[1] * w[0]) / (nv * nw)
    r = scale * np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    t = q1 - r @ p1
    return np.hstack([r, t.reshape(2, 1)])


def align_from_eyes(
    image_bgr: np.ndarray,
    left_eye_px: np.ndarray,
    right_eye_px: np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    desired_left_eye: tuple[float, float] = DESIRED_LEFT_EYE,
) -> np.ndarray:
    """既知の両目座標を使って整列済み画像を返す。"""
    w_out, h_out = output_size
    dlx, dly = desired_left_eye
    dst_left = np.array([dlx * w_out, dly * h_out])
    dst_right = np.array([(1.0 - dlx) * w_out, dly * h_out])
    m = similarity_transform(left_eye_px, right_eye_px, dst_left, dst_right)
    return cv2.warpAffine(
        image_bgr, m, (w_out, h_out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def align_face(
    image_bgr: np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
    desired_left_eye: tuple[float, float] = DESIRED_LEFT_EYE,
) -> np.ndarray | None:
    """1 枚の画像から顔を検出・整列して正規化済み画像を返す。顔がなければ None。"""
    eyes = detect_eye_centers(image_bgr)
    if eyes is None:
        return None
    return align_from_eyes(image_bgr, eyes[0], eyes[1], output_size, desired_left_eye)


def _process_path(input_path: Path, output_path: Path) -> tuple[int, int]:
    """ファイル or ディレクトリ(ImageFolder)を整列処理。(成功, 顔なし) を返す。"""
    ok = miss = 0
    if input_path.is_file():
        img = cv2.imread(str(input_path))
        if img is None:
            raise SystemExit(f"cannot read image: {input_path}")
        aligned = align_face(img)
        if aligned is None:
            print(f"[no face] {input_path}")
            return 0, 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), aligned)
        print(f"[ok] {input_path} -> {output_path}")
        return 1, 0

    for src in sorted(input_path.rglob("*")):
        if src.suffix.lower() not in IMAGE_EXTS or not src.is_file():
            continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"[skip unreadable] {src}")
            continue
        aligned = align_face(img)
        rel = src.relative_to(input_path)
        dst = output_path / rel.with_suffix(".png")
        if aligned is None:
            print(f"[no face] {src}")
            miss += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), aligned)
        ok += 1
    print(f"done: {ok} aligned, {miss} without a detected face")
    return ok, miss


def _selftest() -> None:
    """顔画像なしで幾何変換とパイプライン疎通を検証する。"""
    rng = np.random.default_rng(0)
    w_out, h_out = OUTPUT_SIZE
    dlx, dly = DESIRED_LEFT_EYE
    exp_left = np.array([dlx * w_out, dly * h_out])
    exp_right = np.array([(1.0 - dlx) * w_out, dly * h_out])

    # 1) 幾何: 任意の目位置 → 正準位置に正確に写ること。
    for _ in range(200):
        pl = rng.uniform(0, 1000, size=2)
        pr = rng.uniform(0, 1000, size=2)
        if np.hypot(*(pr - pl)) < 1e-3:
            continue
        m = similarity_transform(pl, pr, exp_left, exp_right)
        got_left = m[:, :2] @ pl + m[:, 2]
        got_right = m[:, :2] @ pr + m[:, 2]
        assert np.allclose(got_left, exp_left, atol=1e-6), (got_left, exp_left)
        assert np.allclose(got_right, exp_right, atol=1e-6)
        # 2) 鏡像でないこと: det(R) > 0、かつ相似（R^T R = s^2 I）。
        r = m[:, :2]
        det = float(np.linalg.det(r))
        assert det > 0, f"transform is a reflection (det={det})"
        s2 = r.T @ r
        assert np.allclose(s2, s2[0, 0] * np.eye(2), atol=1e-6), "not a similarity"

    # 3) warp が正しいサイズを返す。
    dummy = (rng.uniform(0, 255, size=(480, 640, 3))).astype(np.uint8)
    out = align_from_eyes(dummy, np.array([250.0, 220.0]), np.array([390.0, 220.0]))
    assert out.shape == (h_out, w_out, 3), out.shape

    # 4) 検出パイプライン疎通: 顔のないノイズ画像 → None（クラッシュしない）。
    noise = (rng.uniform(0, 255, size=(256, 256, 3))).astype(np.uint8)
    assert align_face(noise) is None, "expected no face in random noise"

    print("face_align self-test PASSED (geometry exact, no reflection, detection OK)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Face alignment (M2)")
    ap.add_argument("--selftest", action="store_true", help="run numeric self-test")
    ap.add_argument("--input", type=Path, help="image file or ImageFolder dir")
    ap.add_argument("--output", type=Path, help="output file or dir")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --selftest)")
    _process_path(args.input, args.output)


if __name__ == "__main__":
    main()
