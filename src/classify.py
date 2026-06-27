"""統一分類インターフェース (M5).

backend に応じて Nemotron VLM / 高速 CNN のいずれかで顔画像を判定する。

    classify(image, backend="nemotron" | "cnn", refs_a=..., refs_b=...) -> "A" | "B"

- "nemotron": llama-server 経由の VLM 判定。refs_a/refs_b を渡すと few-shot
              照合（形態①の本命デモ）。省略すると ill-posed なベースライン。
- "cnn":      MobileNetV3-small によるサブミリ秒判定（速度勝負向き、任意）

TODO(M5): cnn backend を train_cnn の成果物に配線する。
"""

from __future__ import annotations


def classify(image, backend: str = "nemotron", refs_a=None, refs_b=None) -> str:
    """顔画像を "A" または "B" に分類する。

    refs_a / refs_b は VLM の few-shot 見本（形態①）。cnn backend では未使用。
    """
    if backend == "nemotron":
        from .nemotron_client import judge

        return judge(image, refs_a=refs_a, refs_b=refs_b)
    if backend == "cnn":
        raise NotImplementedError("M5: CNN backend wiring not implemented yet")
    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    raise NotImplementedError("M5: CLI not implemented yet")
