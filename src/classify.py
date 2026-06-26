"""統一分類インターフェース (M5).

backend に応じて Nemotron VLM / 高速 CNN のいずれかで顔画像を判定する。

    classify(image, backend="nemotron" | "cnn") -> "A" | "B"

- "nemotron": llama-server 経由の VLM 高速判定（~2s/回、解説デモ向き）
- "cnn":      MobileNetV3-small によるサブミリ秒判定（速度勝負向き、任意）

TODO(M5): nemotron_client / train_cnn の成果物を束ねる。
"""

from __future__ import annotations


def classify(image, backend: str = "nemotron") -> str:
    """顔画像を "A" または "B" に分類する。"""
    if backend == "nemotron":
        from .nemotron_client import judge

        return judge(image)
    if backend == "cnn":
        raise NotImplementedError("M5: CNN backend wiring not implemented yet")
    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    raise NotImplementedError("M5: CLI not implemented yet")
