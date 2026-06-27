# DATA_PRIVATEJ — データを公開 GitHub に出さずに保存する方法

実写画像・学習済み重み・私的メモなどを **公開リポジトリに一切出さず**、ローカルや
非公開リポに安全に保存・管理するための方針と手順。コード・設計・集計値だけを公開し、
データ本体は非公開、という本プロジェクトの前提を担保するためのドキュメント。

> 関連: セットアップは [`READMEJ.md`](./READMEJ.md)、設計解説は [`TECHNICALJ.md`](./TECHNICALJ.md)。

---

## 0. 公開してはいけないもの（非公開対象）

| 種別 | 例 | 理由 |
|---|---|---|
| 実在人物の画像 | `data/raw/`, `data/the_touch/`, `data/train|val/` | 肖像・プライバシー・著作権 |
| 学習済み重み | `results/cnn.pt`, `results/cnn_thetouch.pt`, `*.pt/*.pth/*.onnx/*.safetensors` | 学習データの間接的な含有 |
| モデルバイナリ | `.models/`, `*.task`（初回 DL 取得物）| 再取得可能・サイズ大 |
| 私的な原稿・メモ | `the-touch-classifier.md`, `HANDOFF.md`, 作業ログ | 公開前提でない素材を含む |

→ **公開するのはコード・設計・集計値（精度・混同行列の数値）のみ**。画像やラベル本体は出さない。

---

## 1. いまの防御（`.gitignore`）

以下は既に ignore 済み（コミットされない）:

```gitignore
data/
*.pt
*.pth
*.onnx
*.safetensors
results/*.png
results/*.jpg
.models/
*.task
the-touch-classifier.md
HANDOFF.md
.claude/
```

確認コマンド:

```bash
git check-ignore -v data/the_touch/train/A/foo.png   # 行が出れば ignore されている
git status --porcelain                                # 追跡候補に画像/重みが無いか目視
```

> **注意（穴）**: `PROGRESS.md` と `results/eval.{json,csv}` は現状 **ignore されていない**
> （まだ add していないだけ）。私的にしておきたいなら §2 の要領で gitignore に追加すること。

---

## 2. 「非公開 markdown」でデータ・記録を保存する規約

データ台帳・ラベルの控え・収集メモなどを **gitignore された markdown** に書いて手元に残す。
公開リポには出さないが、markdown なので可読・差分管理しやすい。

### 2.1 規約を有効化する（`.gitignore` に追記）

```gitignore
# 非公開 markdown / 私的データ置き場（コミットしない）
private/
*.private.md
results/eval.json
results/eval.csv
PROGRESS.md
```

追記後の確認:

```bash
git check-ignore -v private/dataset-log.md notes.private.md
```

### 2.2 使い方（例）

`private/dataset-log.md`（または `*.private.md`）に markdown で記録する。例:

```markdown
# the_touch データ台帳（非公開）

## 収集
- 取得日: 2026-06-27 / クエリ: ザ・たっち 双子 ほか
- 収集 69 枚（both 38 / takuya 15 / kazuya 16）、ノイズ 22 枚は _rejected/

## ラベル（A=たくや / B=かずや）
| crop | label | 手掛かり | 出典 |
|---|---|---|---|
| both__xxxx__f0.png | A | 鼻横ほくろ | <URL> |
| both__xxxx__f1.png | B | 面長 | <URL> |

## 評価メモ
- fewshot 47.6% / zeroshot 47.6% / cnn 71.4%（val n=21・ローカル限定）
```

画像そのものは `data/`（ignore 済み）に置き、**この markdown には参照とメタ情報のみ**を書く。
バイナリを base64 で markdown に埋めるのは避ける（肥大化・差分が壊れる）。

---

## 3. 非公開のまま保存・バックアップする手段

| 手段 | 共有 | 暗号 | 向き |
|---|---|---|---|
| A. ローカルのみ（tar/rsync を外部ドライブ・私的クラウドへ）| × | 任意 | 最簡・単独運用 |
| B. **別の Private GitHub リポ**にデータを置く | 限定 | GitHub 管理 | 複数マシンで使う |
| C. git submodule で private データリポを参照 | 限定 | GitHub 管理 | 公開リポにポインタのみ |
| D. git-crypt で公開リポ内に暗号化して置く | 公開リポ越し | あり（鍵管理要）| 1リポで完結したい |
| E. リポ全体を private にする | 限定 | GitHub 管理 | 全部非公開でよい |

### A. ローカル退避（最簡）

```bash
tar czf ~/backups/twin-data-$(date +%F).tgz data/ results/*.pt
#   外部ドライブや私的クラウド（自分しかアクセスできない場所）へ。公開先には置かない。
```

### B. 別の Private GitHub リポにデータを置く

```bash
gh repo create twin-data --private        # 非公開リポを作成
cd ~/twin-data && git init && cp -r ~/twin-classifier-demo/data . && \
  git add . && git commit -m "private data" && \
  git remote add origin git@github.com:<you>/twin-data.git && git push -u origin main
```

公開リポ側は `data/` を ignore したまま。データは private リポにだけ存在する。

### C. submodule（公開リポにはポインタのみ）

```bash
# 公開リポ内に private データリポを submodule として紐付け（中身は private のまま）
git submodule add git@github.com:<you>/twin-data.git data
#   クローンした他人は private リポにアクセスできなければ data を取得できない＝非公開を維持
```

### D. git-crypt（公開リポ内で暗号化）

```bash
git-crypt init
printf 'data/** filter=git-crypt diff=git-crypt\n*.pt filter=git-crypt diff=git-crypt\n' >> .gitattributes
git-crypt add-gpg-user <your-gpg-id>      # 鍵を持つ人だけ復号できる
#   鍵を失うと復号不能。鍵は別管理。公開リポに push されるのは暗号文。
```

### E. リポ全体を private に

```bash
gh repo edit <you>/twin-classifier-demo --visibility private --accept-visibility-change-consequences
```

---

## 4. 流出防止チェックリスト

push する前に:

```bash
git status                       # 画像/重み/私的mdが staged に無いか
git diff --staged --stat         # 大きいバイナリが混じっていないか
git check-ignore -v data/ results/cnn_thetouch.pt   # ignore されているか
```

任意の pre-commit フック（データ混入を機械的に弾く）`.git/hooks/pre-commit`:

```bash
#!/usr/bin/env bash
# data/ 配下や重み/画像が staged なら commit を止める
if git diff --cached --name-only | grep -E '^(data/)|\.(pt|pth|onnx|safetensors|png|jpg|jpeg)$'; then
  echo "ERROR: 非公開対象が staged です。git restore --staged で外してください。" >&2
  exit 1
fi
```

### 既に誤って commit / push してしまったら

```bash
git rm --cached -r data/ ; echo 'data/' >> .gitignore ; git commit -m "stop tracking data"
#   履歴からも消すには git filter-repo / BFG。ただし公開済み＝漏洩前提で、
#   画像の権利者対応・必要なら再アップ依頼の停止等を別途検討する。
```

---

## 5. このリポでの運用まとめ

- **公開 OK**: コード、設計ドキュメント（READMEJ/TECHNICALJ）、合成データ生成スクリプト、集計値。
- **非公開（本ドキュメントの対象）**: 実写画像、学習済み重み、モデルバイナリ、私的原稿・台帳。
- 既定は `.gitignore` でローカル限定。共有や複数マシン運用が要るなら §3 の B/C/E（private リポ）を使う。
- データの記録・台帳は §2 の「非公開 markdown」（`private/` または `*.private.md`）に残す。

> このドキュメント自体は方法論なので公開可。私的にしたい場合は `*.private.md` 規約に従い
> リネーム（例 `DATA_PRIVATEJ.private.md`）するか `private/` に移す。
