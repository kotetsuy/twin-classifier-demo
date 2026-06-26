"""ゼロショット VLM 判定 (M3).

moondream2 を ROCm でロードし、1 枚の顔画像から "A"/"B" を返す。
プロンプトには「双子であること」と「観察可能な見分けの手がかり」を言語で与える。
将来 Qwen-VL 系へ差し替えられるよう、バックエンドは抽象化する。

TODO(M3): moondream2 ロード + プロンプト設計。
"""

from __future__ import annotations


def classify_zeroshot(image, hints: str | None = None) -> str:
    """顔画像を "A" または "B" に分類する。"""
    raise NotImplementedError("M3: zero-shot VLM not implemented yet")
