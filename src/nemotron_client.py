"""Nemotron VLM 判定クライアント (M3).

NVIDIA Nemotron 3 Nano Omni を、自前ビルドの ROCm llama.cpp で常駐させた
`llama-server`（OpenAI 互換 API, 既定 :8080）経由で叩く薄い HTTP クライアント。
VLM 本体の Python 依存は持たない（モデルはサーバ側にロード済み前提）。

2 モード:
  - judge(img)   -> "A" | "B"
        高速判定。grammars/ab.gbnf で出力を A|B に強制し、reasoning を切って
        答えを直接 content に出させる（~0.9s/回）。レイテンシ最小化が目的。
  - explain(img) -> ExplainResult
        解説。「どちらか＋根拠（生え際・眉・ほくろ・非対称 等）」を日本語で語らせる。
        既定は thinking OFF（content に直接・簡潔・約2秒で良質な根拠が出る）。
        think=True で思考トレースを取れるが、このモデルは思考が収束せず
        content が空のまま max_tokens に達することがある（下記注記）。

このサーバ（Nemotron Reasoning）は思考を `reasoning_content`、最終回答を
`content` に分けて返す。judge は chat_template_kwargs.enable_thinking=False で
思考を抑止して content に直接出させる（grammar と併用）。

⚠️ reasoning ON のまま explain すると、思考が終わらず content が空のまま
max_tokens 到達（思考 3000字超）になりやすい。よって explain も既定は
thinking OFF とし、信頼できる非空の根拠を優先する。

サーバ起動は scripts/serve_nemotron.sh を参照。
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_URL = os.environ.get("NEMOTRON_URL", "http://localhost:8080/v1/chat/completions")
# 出力を A|B に強制する GBNF（高速判定パス用）
AB_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "grammars" / "ab.gbnf"

JUDGE_PROMPT = (
    "These are identical twins, person A or person B. Answer only 'A' or 'B'."
)
EXPLAIN_PROMPT = (
    "この人物は一卵性双生児のうち person A か person B です。"
    "どちらか答え、見分けた根拠（生え際・眉・ほくろ・輪郭の非対称など"
    "観察できる特徴）を日本語で簡潔に述べてください。"
)

_SERVE_HINT = (
    "llama-server に接続できません ({url})。\n"
    "別ターミナルで `bash scripts/serve_nemotron.sh` を起動してください "
    "（既定で http://localhost:8080 に OpenAI 互換 API を出します）。"
)


@dataclass
class ExplainResult:
    """解説モードの戻り値。"""

    answer: str          # "A" | "B"（抽出できなければ ""）
    rationale: str       # 日本語の根拠説明（content）
    thinking: str        # 思考トレース（reasoning_content）


def _grammar() -> str:
    return AB_GRAMMAR_PATH.read_text()


def _to_data_url(image) -> str:
    """画像（パス / numpy BGR 配列 / PIL）を JPEG の data URL に変換する。"""
    if isinstance(image, (str, Path)):
        raw = Path(image).read_bytes()
    else:
        import numpy as np

        if isinstance(image, np.ndarray):
            import cv2

            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("failed to JPEG-encode image array")
            raw = buf.tobytes()
        else:  # PIL.Image など save() を持つもの
            import io

            bio = io.BytesIO()
            image.convert("RGB").save(bio, format="JPEG")
            raw = bio.getvalue()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def _message(prompt: str, image) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
        ],
    }]


def _post(payload: dict, url: str, timeout: float) -> dict:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(_SERVE_HINT.format(url=url)) from e
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def _base_url(url: str) -> str:
    return url.split("/v1/", 1)[0]


def ping(url: str = DEFAULT_URL) -> bool:
    """llama-server が応答するか確認する。未起動なら False。"""
    try:
        r = requests.get(f"{_base_url(url)}/health", timeout=3)
        return r.ok
    except requests.exceptions.RequestException:
        return False


def _extract_ab(*texts: str) -> str:
    """与えた文字列群から最初に現れる 'A'/'B' を取り出す。なければ ""。"""
    for t in texts:
        for ch in (t or ""):
            if ch in ("A", "B"):
                return ch
    return ""


def judge(image, url: str = DEFAULT_URL, prompt: str = JUDGE_PROMPT, timeout: float = 30.0) -> str:
    """顔画像を "A" または "B" に高速判定する（grammar 拘束 + reasoning OFF）。"""
    payload = {
        "messages": _message(prompt, image),
        "max_tokens": 4,
        "temperature": 0,
        "grammar": _grammar(),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    msg = _post(payload, url, timeout)
    ans = _extract_ab(msg.get("content", ""), msg.get("reasoning_content", ""))
    if ans not in ("A", "B"):
        raise RuntimeError(f"unexpected judge output: {msg!r}")
    return ans


def explain(
    image,
    url: str = DEFAULT_URL,
    prompt: str = EXPLAIN_PROMPT,
    think: bool = False,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> ExplainResult:
    """顔画像を判定し、根拠（と任意で思考トレース）つきで返す（解説デモ用）。

    think=False（既定）: thinking OFF。content に簡潔な日本語の根拠が直接出る。
    think=True: reasoning ON。reasoning_content に思考トレースが出るが、この
        モデルは思考が収束せず content が空のまま max_tokens に達することがある。
        デモで思考過程を見せたい場合のみ使う。
    """
    payload = {
        "messages": _message(prompt, image),
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    msg = _post(payload, url, timeout)
    rationale = msg.get("content", "") or ""
    thinking = msg.get("reasoning_content", "") or ""
    return ExplainResult(
        answer=_extract_ab(rationale, thinking),
        rationale=rationale,
        thinking=thinking,
    )
