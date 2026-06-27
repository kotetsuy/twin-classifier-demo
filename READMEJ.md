# READMEJ — インストール & 実行手順

`twin-classifier-demo` を `git clone` した状態から、双子判定デモを動かすまでの手順。
技術的な設計の解説は [`TECHNICALJ.md`](./TECHNICALJ.md)、読み物（Qiita記事）は
[`the-touch-classifier.md`](./the-touch-classifier.md) を参照。

> **権利・プライバシー**: 本リポジトリは実在人物の画像も学習済み重みも同梱しない。
> 検証は権利クリアな素材（合成画像／CC ライセンス画像）のみで行う。`data/` と
> `*.pt` 等は `.gitignore` 済み。

---

## 0. 前提環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 24.04 |
| GPU | AMD Ryzen AI MAX+ 395（gfx1151, 統合メモリ 48GB）|
| ROCm | 7.2.x（`/opt/rocm`）|
| Python | 3.12 |
| 必須環境変数 | `HSA_OVERRIDE_GFX_VERSION=11.5.1`（gfx1151 を HIP に認識させる）|

ROCm 版 PyTorch（`torch 2.9.x+rocm7.2.1`）が **Python 3.12 のユーザーサイト
(`~/.local`)** に導入済みであること。PyPI の既定 torch は iGPU を認識しないので使わない。

VLM ルートを使う場合は、別途 **ROCm 対応の llama.cpp**（`-DGGML_HIP=ON`, `gfx1151`）を
ビルドし、`llama-server` バイナリと Nemotron の GGUF を用意しておくこと
（モデル入手は記事のリンク先参照）。

---

## 1. クローンと仮想環境

```bash
git clone https://github.com/kotetsuy/twin-classifier-demo.git
cd twin-classifier-demo

# ユーザーサイトの ROCm torch を継承する venv を作る（torch は入れ直さない）
python3 -m venv --system-site-packages .venv

# プロジェクト依存だけ追加
.venv/bin/pip install -r requirements.txt
```

## 2. ROCm 疎通確認

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash scripts/verify_rocm.sh        # gfx1151 が見え、matmul が通れば PASSED
```

> ROCm でも PyTorch の device 名は `"cuda"`（`torch.cuda.is_available()` が True）。

---

## 3. データを用意する（2 ルート）

実在人物画像は同梱しないので、自分で用意する。用途で使い分ける。

### route A — 合成双子（学習・評価用 / 確実）

権利クリアなラベル付き A/B データを手元生成する。CNN 学習と評価はこれを使う。

```bash
# data/synthetic/{train,val}/{A,B} に生成（seed から再現可能・gitignore 対象）
.venv/bin/python scripts/make_synthetic_twins.py --n-train 300 --n-val 20 --diff 0.7
#   --diff を下げる(例 0.4)ほど A/B が似て難しくなる
```

### route C — 実写の双子（VLM 解説デモ用 / 任意）

CC ライセンスの実在双子写真を Openverse から取得（VLM の定性デモ用ギャラリー）。

```bash
.venv/bin/python scripts/fetch_cc_faces.py -q "identical twins" \
    --source wikimedia --license "cc0,pdm,by,by-sa" -n 40
#   data/raw/ に保存。出典は attribution.csv に記録（CC-BY の帰属に対応）
```

---

## 4. 高速 CNN を学習（route A データ）

```bash
.venv/bin/python src/train_cnn.py --epochs 12
#   results/cnn.pt にベスト重みを保存（gitignore 対象・コミットしない）
#   1枚判定: .venv/bin/python src/train_cnn.py --predict path/to/face.png
```

> CNN は学習データが少ないと過学習する。`--n-train` は 200〜300/クラス以上を推奨。

---

## 5. Nemotron VLM を常駐起動（別ターミナル）

```bash
bash scripts/serve_nemotron.sh        # OpenAI 互換 API を :8080 に出す
# モデル/バイナリのパスは環境変数で上書き可:
#   LLAMA_SERVER, NEMOTRON_MODEL, NEMOTRON_MMPROJ
curl -s localhost:8080/health         # {"status":"ok"} になれば準備完了
```

VLM の judge / explain を直接叩く例:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
import nemotron_client as nc
D = "data/synthetic"
refs_a = [f"{D}/train/A/0000.png", f"{D}/train/A/0001.png"]
refs_b = [f"{D}/train/B/0000.png", f"{D}/train/B/0001.png"]
q = f"{D}/val/B/0000.png"
print("few-shot:", nc.judge(q, refs_a=refs_a, refs_b=refs_b))   # 見本つき = 形態①
print("zero-shot:", nc.judge(q))                                # 参照なし(ill-posed)
print(nc.explain(q, refs_a=refs_a, refs_b=refs_b).rationale)    # 判定根拠つき
PY
```

---

## 6. 評価（手法比較表を出す）

```bash
.venv/bin/python src/evaluate.py --with-cnn
#   fewshot / zeroshot / cnn を同一 val で比較
#   results/eval.{json,csv} と confusion.png（混同行列チャート）を出力
#   --limit N で高速確認、--refs-per-class K で見本数を調整
```

出力例（合成 val n=40, train=300/クラス, diff=0.7）:

| 手法 | 精度 | ms/回(中央値) |
|---|---|---|
| cnn | 100% | ~4 |
| fewshot（見本つき VLM）| 97.5% | ~2818 |
| zeroshot（参照なし VLM）| 55% | ~764 |

---

## 7. リアルタイム判定（キャプチャ→判定→クリック）

X11 環境（`DISPLAY` 必要）。**既定は dry-run**（クリックせずログのみ）で安全。

```bash
# 解説デモモード（見本つき VLM・形態①）。まず dry-run で領域とクリック先を確認
.venv/bin/python src/realtime.py --mode explain --refs-dir data/synthetic/train \
    --region 100 100 400 400 --a-xy 300 800 --b-xy 900 800

# 速度勝負モード（CNN 連打）。実クリックは --no-dry-run
.venv/bin/python src/realtime.py --mode speed --no-dry-run --interval 0.1 --max-frames 30
```

- `--region X Y W H`: キャプチャ領域（既定はプライマリモニタ全体）
- `--a-xy` / `--b-xy`: A/B 判定時のクリック先（既定は領域の左1/3・右2/3）
- `--mode explain` は `--refs-dir`（`A/` `B/` を含む）必須

---

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `torch.cuda.is_available()` が False | `HSA_OVERRIDE_GFX_VERSION=11.5.1` を export。`verify_rocm.sh` で確認 |
| llama-server に繋がらない | `scripts/serve_nemotron.sh` を起動し `/health` が ok か確認 |
| CNN の val 精度が 50% | 学習データ不足の過学習。`--n-train` を増やす |
| realtime で実クリックしない | 既定 dry-run のため。`--no-dry-run` を付ける（pynput / X11 が要る）|
| Openverse 取得が少ない | 顔フィルタが厳しめ。`--no-faces-only` や `--source wikimedia` を調整 |

各マイルストーンの実装状況は [`README.md`](./README.md) の「ステータス」を参照。
