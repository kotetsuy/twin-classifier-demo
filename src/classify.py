"""統一分類インターフェース (M5).

backend に応じて Nemotron VLM / 高速 CNN のいずれかで顔画像を判定する。

    classify(image, backend="nemotron" | "cnn", refs_a=..., refs_b=...) -> "A" | "B"

- "nemotron": llama-server 経由の VLM 判定。refs_a/refs_b を渡すと few-shot
              照合（形態①の本命デモ）。省略すると ill-posed なベースライン。
- "cnn":      MobileNetV3-small によるサブミリ秒判定（速度勝負向き、要 results/cnn.pt）

backend 配下のモジュール (nemotron_client / train_cnn) は遅延 import する
（VLM だけ使うときに torch を読み込まない、等のため）。
"""

from __future__ import annotations

_CNN = None  # (model, classes, transform, device) のキャッシュ（初回 load のみ）


def _cnn():
    """学習済み CNN を一度だけロードしてキャッシュする。"""
    global _CNN
    if _CNN is None:
        try:
            from .train_cnn import load_classifier
        except ImportError:  # スクリプト実行（パッケージ外）からの利用
            from train_cnn import load_classifier
        _CNN = load_classifier()
    return _CNN


def classify(image, backend: str = "nemotron", refs_a=None, refs_b=None) -> str:
    """顔画像を "A" または "B" に分類する。

    refs_a / refs_b は VLM の few-shot 見本（形態①）。cnn backend では未使用。
    """
    if backend == "nemotron":
        try:
            from .nemotron_client import judge
        except ImportError:
            from nemotron_client import judge

        return judge(image, refs_a=refs_a, refs_b=refs_b)
    if backend == "cnn":
        try:
            from .train_cnn import predict_label
        except ImportError:
            from train_cnn import predict_label

        model, classes, tf, device = _cnn()
        return predict_label(model, classes, tf, device, image)
    raise ValueError(f"unknown backend: {backend!r}")


if __name__ == "__main__":
    raise NotImplementedError("M5: CLI not implemented yet")
