"""ザ・たっち（双子芸人）の実写を Web 画像検索で収集する（route C 実写・ローカル専用）.

ddgs(DuckDuckGo) の画像検索で「ザ・たっち」関連語を引き、顔が検出できた画像のみを
`data/raw/the_touch/<weak_label>/` に保存し、出典(URL/タイトル/クエリ)を attribution.csv に残す。

重要な前提・限界:
- これは芸能人の実写であり権利クリアではない。**公開リポにコミットしない**（data/ は .gitignore 済み）。
  ローカルの個人検証用途に限る。
- ザ・たっちは一卵性双生児で、検索語「…たくや」「…かずや」でもコンビ両方が写る画像が大半。
  よって <weak_label> は**弱いヒントに過ぎず、A/B の正解ラベルにはならない**。確実な手掛かりは
  「たくやは鼻の横にほくろ」。最終的な A/B 仕分けは人手レビューが必要（本スクリプトは収集のみ）。

CLI:
    python scripts/fetch_the_touch.py                 # 全クエリ群を収集
    python scripts/fetch_the_touch.py --per-query 30  # 1クエリあたり最大枚数
    python scripts/fetch_the_touch.py --no-faces-only  # 顔フィルタ無効
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import requests
from ddgs import DDGS

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (X11; Linux x86_64) twin-classifier-demo/0.1 (local research)"

# 弱ラベル -> 検索クエリ群。「たくや/かずや」狙いでもコンビ両方が写る点に注意。
QUERY_GROUPS = {
    "both": ["ザ・たっち 双子 芸人", "ザ・たっち コンビ", "ザ・たっち お笑い", "The Touch 双子 芸人"],
    "takuya": ["ザ・たっち たくや", "ザ・たっち 角田拓也", "ザ・たっち 兄 たくや"],
    "kazuya": ["ザ・たっち かずや", "ザ・たっち 角田和也", "ザ・たっち 弟 かずや"],
}

ATTR_FIELDS = ["filename", "weak_label", "query", "title", "page_url", "image_url"]


def face_filter():
    """顔検出関数を返す（mediapipe 未導入なら None）。"""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import cv2
        import numpy as np
        import face_align
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 顔フィルタ無効化（{e}）")
        return None

    def has_face(raw: bytes) -> bool:
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 64:
            return False
        return face_align.detect_eye_centers(img) is not None

    return has_face


def download(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "image" not in ct and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return None
        return r.content
    except requests.exceptions.RequestException as e:
        print(f"[skip] {url[:70]} ({e})")
        return None


def ext_of(url: str, raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    low = url.lower()
    for e in ("jpg", "jpeg", "png", "webp"):
        if e in low:
            return "jpg" if e == "jpeg" else e
    return "jpg"


def collect(per_query: int, faces_only: bool, sleep: float) -> None:
    out_root = ROOT / "data" / "raw" / "the_touch"
    has_face = face_filter() if faces_only else None
    seen: set[str] = set()  # md5 で重複排除
    attr_path = out_root / "attribution.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    if attr_path.exists():
        with attr_path.open(newline="") as f:
            for row in csv.DictReader(f):
                p = out_root / row["weak_label"] / row["filename"]
                if p.exists():
                    seen.add(hashlib.md5(p.read_bytes()).hexdigest())

    rows: list[dict] = []
    totals = {k: 0 for k in QUERY_GROUPS}
    for label, queries in QUERY_GROUPS.items():
        out = out_root / label
        out.mkdir(parents=True, exist_ok=True)
        for q in queries:
            print(f"\n=== [{label}] '{q}' ===")
            try:
                with DDGS() as d:
                    results = list(d.images(q, max_results=per_query, region="jp-jp", safesearch="off"))
            except Exception as e:  # noqa: BLE001
                print(f"[search err] {e}")
                continue
            for item in results:
                url = item.get("image") or ""
                if not url:
                    continue
                raw = download(url)
                if raw is None:
                    continue
                h = hashlib.md5(raw).hexdigest()
                if h in seen:
                    continue
                if has_face is not None and not has_face(raw):
                    continue
                fname = f"{h[:16]}.{ext_of(url, raw)}"
                (out / fname).write_bytes(raw)
                seen.add(h)
                totals[label] += 1
                rows.append({
                    "filename": fname, "weak_label": label, "query": q,
                    "title": (item.get("title") or "")[:120],
                    "page_url": item.get("url", ""), "image_url": url,
                })
                print(f"  [ok {totals[label]}] {fname}  {(item.get('title') or '')[:50]}")
                time.sleep(sleep)

    if rows:
        write_header = not attr_path.exists()
        with attr_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ATTR_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(rows)
    print("\n=== 収集サマリ ===")
    for k, v in totals.items():
        print(f"  {k}: {v} 枚")
    print(f"  出典: {attr_path}")
    print("注: weak_label は検索語ベースの弱ヒント。A/B 正解は人手レビューが必要"
          "（たくや=鼻横にほくろ）。")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect ザ・たっち real photos (local only)")
    ap.add_argument("--per-query", type=int, default=25, help="1クエリあたり最大取得数")
    ap.add_argument("--no-faces-only", dest="faces_only", action="store_false",
                    help="顔検出フィルタを無効化")
    ap.add_argument("--sleep", type=float, default=0.25, help="ダウンロード間隔 秒")
    ap.set_defaults(faces_only=True)
    args = ap.parse_args()
    collect(args.per_query, args.faces_only, args.sleep)


if __name__ == "__main__":
    main()
