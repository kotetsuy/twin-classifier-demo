# twin-classifier-demo

ローカル AI **だけ** で一卵性双生児を見分けられるかを検証する技術デモ。
クラウド推論は一切使わず、AMD ROCm の iGPU 上で完結させる。

最終的に「画面キャプチャ → 判定 → 左右クリック」までリアルタイム自動化することを目標とする。

> **権利・プライバシーへの配慮**
> 本リポジトリには実在人物の画像・テレビキャプチャ・宣材、およびそれらから学習した重みを **一切同梱しない**。
> 検証は権利処理済みの素材（自前撮影 / 公開データセット / 生成画像）のみで行う。
> コード上のクラス名は中立 (`A` / `B`) とし、固有名詞には依存しない汎用の双子分類器として実装している。

## 構成

| ファイル | 役割 |
|---|---|
| `src/face_align.py` | ランドマーク基準で顔をクロップ・224x224 正規化 |
| `src/zeroshot_vlm.py` | moondream2 等の VLM でゼロショット判定 |
| `src/train_cnn.py` | MobileNetV3-small を 2 クラスに付け替えて学習 |
| `src/classify.py` | 統一インターフェース `classify(image, backend) -> "A"\|"B"` |
| `src/evaluate.py` | accuracy / 混同行列 / 手法比較表 |
| `src/realtime.py` | `mss` で画面キャプチャ → `classify` → 左右クリック |
| `scripts/verify_rocm.sh` | torch + ROCm 疎通チェック |

データセットは `data/train/{A,B}` および `data/val/{A,B}`（ImageFolder 形式、gitignore 対象）。

## 環境

- Ubuntu 24.04 / AMD Ryzen AI MAX+ 395（gfx1151, 統合メモリ 48GB）
- ROCm 7.2.x / Python 3.12
- 重要: `HSA_OVERRIDE_GFX_VERSION=11.5.1`（gfx1151 を HIP に認識させるため）

### セットアップ

```bash
# 1. PyTorch は ROCm ホイールを使う（PyPI の既定ビルドは iGPU を認識しない）
#    本マシンには ROCm ビルドが導入済み (torch 2.9.x+rocm7.2.1)。

# 2. アプリ依存をインストール
pip install -r requirements.txt

# 3. ROCm 疎通確認（gfx1151 が見えること）
bash scripts/verify_rocm.sh
```

PyTorch では ROCm でも device 名に `"cuda"` を使う（`torch.cuda.is_available()` が True）。

## 使い方

> M2 以降の実装に合わせて追記予定。

```bash
# ゼロショット VLM 判定（予定）
python src/classify.py --image path/to/face.png --backend vlm

# CNN 学習（予定）
python src/train_cnn.py --data data --epochs 20

# 手法比較・評価（予定）
python src/evaluate.py --data data/val

# リアルタイム（予定）
python src/realtime.py --region 0,0,1280,720
```

## ステータス

- [x] M1: スキャフォールド / ROCm 疎通確認
- [ ] M2: 顔アライメント (`face_align.py`)
- [ ] M3: ゼロショット VLM (`zeroshot_vlm.py`)
- [ ] M4: CNN 学習 (`train_cnn.py`)
- [ ] M5: 統一インターフェース・評価 (`classify.py` / `evaluate.py`)
- [ ] M6: リアルタイム化 (`realtime.py`)
- [ ] M7: 記事 (`the-touch-classifier.md`) への結果反映
