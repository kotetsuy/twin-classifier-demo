"""統一分類インターフェース (M5).

backend に応じて VLM / CNN のいずれかで顔画像を判定する。

    classify(image, backend="vlm" | "cnn") -> "A" | "B"

TODO(M5): zeroshot_vlm / train_cnn の成果物を束ねる。
"""

from __future__ import annotations


def classify(image, backend: str = "vlm") -> str:
    """顔画像を "A" または "B" に分類する。"""
    if backend == "vlm":
        from .zeroshot_vlm import classify_zeroshot

        return classify_zeroshot(image)
    if backend == "cnn":
        raise NotImplementedError("M5: CNN backend wiring not implemented yet")
    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    raise NotImplementedError("M5: CLI not implemented yet")
