"""Nemotron VLM 判定クライアント (M3).

NVIDIA Nemotron 3 Nano Omni を、自前ビルドの ROCm llama.cpp で常駐させた
`llama-server`（OpenAI 互換 API, 既定 :8080）経由で叩く薄い HTTP クライアント。
VLM 本体の Python 依存は持たない（モデルはサーバ側にロード済み前提）。

2 モード:
  - judge(img)   -> "A" | "B"
        高速判定。reasoning OFF、max_tokens 最小、grammars/ab.gbnf で出力を
        A|B に強制。レイテンシ最小化が目的（とはいえ ~2s/回はアーキ上の下限）。
  - explain(img) -> ExplainResult
        解説。reasoning ON で「どちらか＋根拠（生え際・眉・ほくろ・非対称 等）」を
        日本語で語らせ、[Start thinking]…[End thinking] の思考トレースも返す。

サーバ起動は scripts/serve_nemotron.sh を参照。

TODO(M3): judge / explain / ping を実装（HANDOFF の参考実装ベース）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = os.environ.get("NEMOTRON_URL", "http://localhost:8080/v1/chat/completions")
# 出力を A|B に強制する GBNF（高速判定パス用）
AB_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "grammars" / "ab.gbnf"


@dataclass
class ExplainResult:
    """解説モードの戻り値。"""

    answer: str          # "A" | "B"
    rationale: str       # 日本語の根拠説明
    thinking: str        # [Start thinking]…[End thinking] の思考トレース（あれば）


def ping(url: str = DEFAULT_URL) -> bool:
    """llama-server が応答するか確認する。未起動なら False。"""
    raise NotImplementedError("M3: server health check not implemented yet")


def judge(image, url: str = DEFAULT_URL) -> str:
    """顔画像を "A" または "B" に高速判定する（grammar 拘束）。"""
    raise NotImplementedError("M3: fast judge not implemented yet")


def explain(image, url: str = DEFAULT_URL) -> ExplainResult:
    """顔画像を判定し、根拠と思考トレースつきで返す（解説デモ用）。"""
    raise NotImplementedError("M3: explain mode not implemented yet")
