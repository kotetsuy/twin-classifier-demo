"""データセット切替の単一スイッチ.

環境変数 `TWIN_DATASET`（既定 "synthetic"）ひとつで、学習・評価・分類・リアルタイムが
参照する **データディレクトリと CNN 重み** を一括で切り替える。

  synthetic … 合成双子（主データ・既定）        : data/synthetic , results/cnn.pt
  the_touch … 実写ザ・たっち（ローカル限定・非公開）: data/the_touch , results/cnn_thetouch.pt

使い方:
    python src/evaluate.py --with-cnn                       # 既定=合成
    TWIN_DATASET=the_touch python src/evaluate.py --with-cnn  # 実写に切替
    TWIN_DATASET=the_touch python src/realtime.py --mode speed ...

各スクリプトの --data / --weights の既定値がこのスイッチに従う（明示指定すれば上書き可）。
新しいデータセットは REGISTRY に1行足すだけで増やせる。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_VAR = "TWIN_DATASET"
DEFAULT = "synthetic"

REGISTRY: dict[str, dict[str, Path]] = {
    "synthetic": {
        "data": ROOT / "data" / "synthetic",
        "weights": ROOT / "results" / "cnn.pt",
    },
    "the_touch": {
        "data": ROOT / "data" / "the_touch",
        "weights": ROOT / "results" / "cnn_thetouch.pt",
    },
}


def name() -> str:
    n = os.environ.get(ENV_VAR, DEFAULT)
    if n not in REGISTRY:
        raise SystemExit(f"未知の {ENV_VAR}={n!r}. 選択肢: {list(REGISTRY)}")
    return n


def data_dir() -> Path:
    return REGISTRY[name()]["data"]


def weights() -> Path:
    return REGISTRY[name()]["weights"]
