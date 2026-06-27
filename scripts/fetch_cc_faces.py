"""権利クリアな顔画像を Openverse から取得する補助スクリプト。

Openverse API（https://api.openverse.org/）でライセンスを絞り込み（既定 CC0 +
パブリックドメインマーク）、検索語にヒットした画像を `data/` 配下にダウンロードし、
出典・ライセンスを `attribution.csv` に残す。実画像はリポジトリにコミットしない
（`data/` は .gitignore 済み）。記事・README の「検証は権利処理済み素材のみ」を
担保するための取得経路。

Openverse の検索結果はノイズが多い（"identical twins" に MRI 画像などが混ざる）。
既定で mediapipe FaceLandmarker による顔検出フィルタを通し、顔が取れた画像のみ残す
（`--no-faces-only` で無効化）。

注意:
- CC0/PDM は著作権上クリアだが、被写体の肖像・プライバシー権までは保証しない。
  公開リポにコミットしない・記事掲載時は別途配慮する、を前提に使うこと。
- API はキー不要だが匿名はレート制限あり。大量取得は控えめに。

CLI:
    python scripts/fetch_cc_faces.py --query "identical twins" --count 40
    python scripts/fetch_cc_faces.py -q twins -n 20 --out data/raw/twins --no-faces-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

API = "https://api.openverse.org/v1/images/"
UA = "twin-classifier-demo/0.1 (research; rights-cleared data only)"
ROOT = Path(__file__).resolve().parent.parent

# 出典CSVに残す列（filename を先頭に付与）。
ATTR_FIELDS = [
    "filename", "id", "title", "creator", "license", "license_version",
    "license_url", "foreign_landing_url", "source", "url",
]


def _face_filter():
    """顔検出フィルタ関数を返す。mediapipe 未導入なら None（フィルタ無効）。"""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        import face_align
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 顔フィルタを無効化（依存の読み込み失敗: {e})")
        return None

    def has_face(raw: bytes) -> bool:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        return face_align.detect_eye_centers(img) is not None

    return has_face


def search(query: str, license_: str, page: int, page_size: int, source: str | None) -> dict:
    params = {
        "q": query,
        "license": license_,
        "page": page,
        "page_size": page_size,
        "mature": "false",
    }
    # 例: source="wikimedia" は実在双子の写真が当たりやすい（汎用クエリは
    # アルバム名一致のノイズが多い）。
    if source:
        params["source"] = source
    r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def download(url: str, timeout: float = 30.0) -> bytes | None:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        return r.content
    except requests.exceptions.RequestException as e:
        print(f"[skip] download failed: {url} ({e})")
        return None


def fetch(
    query: str,
    out: Path,
    count: int,
    license_: str,
    faces_only: bool,
    source: str | None = None,
    page_size: int = 20,  # 匿名アクセスの上限（>20 は 401）
    max_pages: int = 30,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    has_face = _face_filter() if faces_only else None

    attr_path = out / "attribution.csv"
    seen: set[str] = set()
    if attr_path.exists():  # 再実行時は既存IDを尊重して追記
        with attr_path.open(newline="") as f:
            seen = {row["id"] for row in csv.DictReader(f)}

    saved = 0
    rows: list[dict] = []
    page = 1
    while saved < count and page <= max_pages:
        try:
            data = search(query, license_, page, page_size, source)
        except requests.exceptions.RequestException as e:
            print(f"[stop] API error on page {page}: {e}")
            break
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            if saved >= count:
                break
            iid = item.get("id", "")
            if not iid or iid in seen:
                continue
            # Openverse 経由のサムネイルを優先（プロバイダ CDN の直叩きは
            # wikimedia 等で 429 になる。サムネは ~600px で顔クロップには十分）。
            src_url = item.get("thumbnail") or item.get("url")
            if not src_url:
                continue
            raw = download(src_url)
            if raw is None and item.get("url") and item["url"] != src_url:
                raw = download(item["url"])  # サムネ失敗時は原本にフォールバック
            if raw is None:
                continue
            if has_face is not None and not has_face(raw):
                print(f"[no face] {item.get('title')!r} -> skip")
                continue
            ext = (item.get("filetype") or "jpg").lower().split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
            fname = f"{iid}.{ext}"
            (out / fname).write_bytes(raw)
            seen.add(iid)
            rows.append({k: item.get(k, "") for k in ATTR_FIELDS} | {"filename": fname})
            saved += 1
            print(f"[ok {saved}/{count}] {item.get('license')} {fname}  {item.get('title')!r}")
            time.sleep(0.3)  # 控えめに
        page += 1

    if rows:
        write_header = not attr_path.exists()
        with attr_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ATTR_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(rows)
    print(f"\n{saved} 枚を {out} に保存。出典: {attr_path}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch rights-cleared face images from Openverse")
    ap.add_argument("-q", "--query", default="identical twins", help="検索語")
    ap.add_argument("-n", "--count", type=int, default=40, help="取得枚数")
    ap.add_argument("--license", default="cc0,pdm", help="ライセンス絞り込み（既定: cc0,pdm。by,by-sa は要帰属）")
    ap.add_argument("--source", default=None, help="プロバイダ絞り込み（例: wikimedia。実在双子に当たりやすい）")
    ap.add_argument("--out", type=Path, default=None, help="出力先（既定: data/raw/<query>）")
    ap.add_argument(
        "--no-faces-only", dest="faces_only", action="store_false",
        help="顔検出フィルタを無効化（既定は有効）",
    )
    ap.set_defaults(faces_only=True)
    args = ap.parse_args()

    out = args.out or (ROOT / "data" / "raw" / args.query.replace(" ", "_"))
    fetch(args.query, out, args.count, args.license, args.faces_only, args.source)


if __name__ == "__main__":
    main()
