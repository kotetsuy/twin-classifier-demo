"""顔アライメント (M2).

目・鼻のランドマークを基準に顔をクロップし、224x224 に正規化する。
双子の微細な非対称が識別の手がかりになるため、左右反転 augmentation は使わない。

TODO(M2): mediapipe FaceMesh で実装。
"""

from __future__ import annotations

OUTPUT_SIZE = (224, 224)


def align_face(image):
    """1 枚の画像から顔を検出・整列し、正規化済み画像を返す。"""
    raise NotImplementedError("M2: face alignment not implemented yet")
