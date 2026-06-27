# TECHNICALJ — 技術解説

`twin-classifier-demo` の設計判断・実装の勘所・実測知見をまとめる。
セットアップ手順は [`READMEJ.md`](./READMEJ.md)、読み物は
[`the-touch-classifier.md`](./the-touch-classifier.md)。

---

## 1. 全体構成

```
入力画像 ─┬─→ [face_align] 目基準で 224x224 正規化（実写用・鏡像なし）
          │
          ├─→ [nemotron_client] llama-server(:8080, OpenAI互換) 経由で VLM 判定
          │        judge()  : A|B を grammar 拘束で高速返答
          │        explain(): A|B + 日本語の根拠（思考トレースも任意で）
          │        ※ refs_a/refs_b を渡すと few-shot 照合（形態①）
          │
          └─→ [train_cnn] MobileNetV3-small（合成データ学習・サブms）

      [classify] backend=nemotron|cnn を束ねる統一 IF
      [evaluate] 精度・混同行列・レイテンシで手法比較
      [realtime] mss キャプチャ → classify → クリック（explain/speed の2モード）
```

データ生成は `scripts/make_synthetic_twins.py`（route A）と
`scripts/fetch_cc_faces.py`（route C）。

---

## 2. いちばんの肝：単一画像 A/B はそのままでは解けない（ill-posed）

「双子の写真1枚を見せて A か B か当てる」は、**A と B が誰なのかをモデルに与えない限り
原理的に不良設定（ill-posed）**。"A"/"B" は中身のないラベルで、1枚だけ見ても基準がない。

実測でもこれは明確に出る:

- **zero-shot（参照なし）VLM の精度 ≒ 55%**（ほぼ偶然）。しかも出力が片方（"A"）に
  退化し、B の recall は 10% まで落ちる。混同行列は「常に A と答える」縦一列になる。

→ これを **well-posed** にするには「A と B が誰か」をモデルに教える必要があり、教え方が
2 通りある。それがそのまま「判定バックエンドの二段構え」になる。

| 教え方 | 実装 | 特徴 |
|---|---|---|
| **見本をその場で提示**（in-context）| few-shot VLM（形態①）| 学習不要・解説が出る・遅い |
| **重みに学習させる** | CNN（route A データ）| 速い・正確・要ラベル学習 |

---

## 3. 形態①：見本つき VLM 照合（few-shot）

本命デモ。A の見本画像・B の見本画像をプロンプトに同梱し、最後に出題画像を置いて
「最後のはどっち?」と照合させる。OpenAI 互換 API の content 配列に複数画像を並べる:

```
content = [
  {text: "person A:"}, {image: A見本1}, {image: A見本2},
  {text: "person B:"}, {image: B見本1}, {image: B見本2},
  {text: "最後の画像は A か B か。根拠も述べよ"},
  {image: 出題画像},          # ← 必ず最後（プロンプトの "the LAST image" と対応）
]
```

実装は `nemotron_client._ref_message()`。`judge()/explain()` は `refs_a, refs_b`（各
画像リスト）を受け取り、与えられたら few-shot、省略したら単一画像（ill-posed
ベースライン）になる。`classify()` も refs を透過する。

---

## 4. Nemotron 3 Nano Omni を llama-server で叩く際の実測知見

モデル: `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning`（UD-Q4_K_XL ≒ 22GB）+
`mmproj-F16`（vision）。ROCm llama.cpp の `llama-server` を `-ngl 99 -c 8192` で常駐。

### 4.1 reasoning モデルの出力分離

このモデルは **思考を `reasoning_content`、最終回答を `content`** に分けて返す。

- **judge（高速判定）**: reasoning ON のまま grammar 拘束すると、答えが
  `reasoning_content` 側に出て `content` が空になる。→ `chat_template_kwargs={"enable_thinking": False}`
  で thinking を切り、`content` に直接 A/B を出させる。`grammars/ab.gbnf`（`root ::= "A" | "B"`）
  で出力を1トークンに強制。
- **explain（解説）**: reasoning ON のままだと **思考が収束せず、`content` が空のまま
  `max_tokens` に到達**する（思考 3000字超、`finish_reason: length`）。「3文以内で」等の
  指示も思考フェーズは無視される。→ explain も **既定 thinking OFF**。`content` に簡潔で
  非空の根拠が約2秒で出る。`think=True` はトレース取得用オプション（結論まで至らない
  場合あり、と明記）。

### 4.2 レイテンシ

- 単一画像 judge: ~0.8s/回
- few-shot judge（見本2＋出題＝3画像）: ~2.8s/回（画像枚数に比例して伸びる）

画像入力は path / numpy(BGR) / PIL を `_to_data_url()` で JPEG data URL 化して渡す。

---

## 5. 高速 CNN（route A データで学習）

`MobileNetV3-small`（ImageNet 事前学習）の分類ヘッドを 2 クラスに付け替え。

- **左右反転 augmentation は禁止**。双子の識別手がかりは微細な左右非対称（ほくろ・眉の
  角度・生え際）なので、鏡像化すると手がかりが消える。`face_align` の「相似変換のみ・
  鏡像なし（det(R)=s²>0）」と同じ思想。augmentation は明度/コントラストの軽い揺らぎのみ。
- 推論ヘルパ `load_classifier()/predict_label()` を `classify`/`evaluate`/`realtime` で
  共有。`classify` 側はモデルを一度だけロードしてキャッシュする。

### データ量の効き（実測）

合成データの量が generalization を直接左右する:

| train枚数/クラス | 挙動 |
|---|---|
| 40 | train_loss → 0 に落ちるが **val 50%**（過学習・丸暗記。汎化せず）|
| 300 | **epoch 2 で val 100%**（識別特徴を学習できる）|

→ 「少量データでは VLM(few-shot) の方が強く、十分なデータがあれば CNN が圧勝」という、
データ量に依存した綺麗な逆転が観察できる。

---

## 6. 評価方法（`evaluate.py`）

ImageFolder の val 全体に各手法の predict を適用し、精度・per-class accuracy・
混同行列・レイテンシ（ms/回、中央値）を集計。few-shot の見本は **train から取る**
ので val とリークしない。`results/eval.{json,csv}` と `confusion.png` を出力。

### 3 手法比較（合成 val n=40, train=300/クラス, diff=0.7, seed=0, 見本2/クラス）

| 手法 | 精度 | A recall | B recall | ms/回(中央値) | 性質 |
|---|---|---|---|---|---|
| cnn | **100%** | 100% | 100% | **~4** | 速い＋正確。要ラベル学習 |
| fewshot VLM | 97.5% | 95% | 100% | ~2818 | 学習不要・解説つき。約700倍遅い |
| zeroshot VLM | 55% | 100% | 10% | ~764 | 参照なしで破綻（ill-posed の実証）|

> レイテンシ中央値は CNN の初回モデルロードを外して定常推論を反映する（mean には載る）。
> few-shot の弱点として A recall 75〜95% の取りこぼし（「目立つ特徴=B」へのバイアス）が
> 出る場合がある。

---

## 7. データ取得の2ルート（設計判断）

最初は研究用の実写双子データ（ND-TWINS-2009-2010 等）も検討したが、機関署名が要る・
250GB・個人のデモ用途に重い、という理由で見送り。代わりに **C+A の二本立て**:

- **route A（合成 / `make_synthetic_twins.py`）**: A・B は同一「ゲノム」を共有し、
  HANDOFF が挙げる手がかり（ほくろ・眉の角度・生え際）だけが安定して異なる。各画像に
  撮影ゆらぎ（回転・並進・スケール・明度・背景・ノイズ）を加え、同一人物内のばらつきを
  作る。`--diff` で難易度可変。**権利完全クリア・ラベル付き・seed 再現可能**なので学習と
  評価の主データに最適。写実性は問わない（漫画的でよい）。
- **route C（実写 / `fetch_cc_faces.py`）**: Openverse で CC0/PDM（+任意で CC-BY）を絞り、
  顔検出（mediapipe FaceLandmarker 再利用）で非顔を除外、出典を `attribution.csv` に記録。
  `--source wikimedia` で実在双子が当たる。ただし「2人1枚もの・別ペア・少数」なので
  **教師あり A/B には不向き**で、VLM の定性的な解説ギャラリー用と割り切る。
  原本直叩きは 429 になるため Openverse のサムネ経由で取得。

> mediapipe は実写顔用の検出器なので、合成顔（route A）は検出しない。合成は生成時点で
> 正規化済みなので face_align は不要。

---

## 8. リアルタイム化（`realtime.py`）

`mss` で領域キャプチャ → `classify` → A/B を `--a-xy`/`--b-xy` のクリック先に対応づけて
クリック。2 モード:

- **explain**（既定）: few-shot VLM で判定＋根拠表示。自分でペースを握る（~3s/枚）。
- **speed**: CNN でサブ ms 判定して連打。

安全のため**既定 dry-run**（クリックせずログ）。実クリックは `--no-dry-run`（pynput / X11）。
クリック手段が無ければ自動的に dry-run にフォールバックする。

---

## 9. 既知の限界・拡張余地

- 静止画分類なので、双子が**動きを揃える**と手がかりが消える。Nemotron 3 Nano Omni は
  動画入力も持つので、時系列特徴での改善余地あり。
- few-shot VLM は見本の質に精度が依存する。見本枚数（`--refs-per-class`）や見本の
  選び方で変わる。
- 合成データは識別が綺麗すぎる（`diff=0.7` で CNN 100%）。`--diff` を下げると現実的な
  難易度に近づき、手法間の差がより見える。
