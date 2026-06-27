"""評価・手法比較 (M5).

ImageFolder 形式の val（既定 data/synthetic/val/{A,B}）に対して各手法の
accuracy・混同行列・レイテンシ(ms/回) を集計し、手法比較表を出力する。
記事の比較表に流用できるよう results/ に CSV / JSON / 混同行列 PNG も保存する。

対象手法:
  - fewshot : Nemotron VLM + 見本（train/{A,B} を参照）。本命デモ（形態①）。
  - zeroshot: Nemotron VLM 単一画像（参照なし。ill-posed なベースライン）。
  - cnn     : 高速 CNN（M4 の重みがあれば。未実装なら自動スキップ）。

参照（見本）は train 分割から取る。val とは別画像なのでリークしない。

CLI:
    python src/evaluate.py                               # fewshot + zeroshot
    python src/evaluate.py --methods fewshot --limit 20  # 速く確認
    python src/evaluate.py --data data/synthetic --refs-per-class 3 --with-cnn
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nemotron_client as nc  # noqa: E402

LABELS = ("A", "B")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_refs(data: Path, n: int) -> tuple[list[Path], list[Path]]:
    """train/{A,B} の先頭 n 枚ずつを few-shot 見本として返す。"""
    refs = {}
    for label in LABELS:
        imgs = _list_images(data / "train" / label)
        if len(imgs) < n:
            raise SystemExit(f"train/{label} に見本が足りない（{len(imgs)} < {n}）")
        refs[label] = imgs[:n]
    return refs["A"], refs["B"]


def load_val(data: Path, limit: int | None) -> list[tuple[str, Path]]:
    """(正解ラベル, 画像パス) のリストを val から作る。limit でクラス毎に上限。"""
    samples: list[tuple[str, Path]] = []
    for label in LABELS:
        imgs = _list_images(data / "val" / label)
        if limit is not None:
            imgs = imgs[:limit]
        samples += [(label, p) for p in imgs]
    if not samples:
        raise SystemExit(f"val 画像が見つからない: {data}/val/{{A,B}}")
    return samples


@dataclass
class Result:
    method: str
    truths: list[str] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)


def run_method(method: str, predict, samples: list[tuple[str, Path]]) -> Result:
    """predict(path)->'A'|'B' を全 val に適用して結果を集める。"""
    res = Result(method=method)
    for i, (truth, path) in enumerate(samples, 1):
        t = time.time()
        try:
            pred = predict(path)
        except Exception as e:  # noqa: BLE001  判定不能は誤りとして "?" 記録
            print(f"  [{method}] {path.name}: error {e}")
            pred = "?"
        dt = (time.time() - t) * 1000
        res.truths.append(truth)
        res.preds.append(pred)
        res.latencies_ms.append(dt)
        print(f"  [{method} {i}/{len(samples)}] truth={truth} pred={pred} ({dt:.0f}ms)")
    return res


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def metrics(res: Result) -> dict:
    n = len(res.truths)
    correct = sum(t == p for t, p in zip(res.truths, res.preds))
    # 混同行列: confusion[truth][pred]。pred は A/B 以外（"?"）も other に集計。
    conf = {t: {"A": 0, "B": 0, "other": 0} for t in LABELS}
    for t, p in zip(res.truths, res.preds):
        conf[t][p if p in ("A", "B") else "other"] += 1
    per_class = {
        t: (conf[t][t] / sum(conf[t].values()) if sum(conf[t].values()) else 0.0)
        for t in LABELS
    }
    lat = res.latencies_ms
    return {
        "method": res.method,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "per_class_acc": per_class,
        "confusion": conf,
        "latency_ms_mean": sum(lat) / n if n else 0.0,
        "latency_ms_median": _median(lat),
    }


def print_report(all_metrics: list[dict]) -> None:
    print("\n=== 比較表 ===")
    head = f"{'method':<10} {'acc':>6} {'A acc':>6} {'B acc':>6} {'ms/回(median)':>14} {'n':>4}"
    print(head)
    print("-" * len(head))
    for m in all_metrics:
        print(
            f"{m['method']:<10} {m['accuracy']*100:>5.1f}% "
            f"{m['per_class_acc']['A']*100:>5.1f}% {m['per_class_acc']['B']*100:>5.1f}% "
            f"{m['latency_ms_median']:>13.0f} {m['n']:>4}"
        )
    print("\n=== 混同行列 (行=正解, 列=予測) ===")
    for m in all_metrics:
        c = m["confusion"]
        print(f"[{m['method']}]            pred A  pred B  other")
        for t in LABELS:
            print(f"  truth {t}        {c[t]['A']:>6} {c[t]['B']:>6} {c[t]['other']:>6}")


def save_results(all_metrics: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    with (out / "eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "accuracy", "A_acc", "B_acc",
                    "latency_ms_mean", "latency_ms_median"])
        for m in all_metrics:
            w.writerow([m["method"], m["n"], f"{m['accuracy']:.4f}",
                        f"{m['per_class_acc']['A']:.4f}", f"{m['per_class_acc']['B']:.4f}",
                        f"{m['latency_ms_mean']:.1f}", f"{m['latency_ms_median']:.1f}"])
    _save_confusion_png(all_metrics, out)
    print(f"\n保存: {out}/eval.json, eval.csv" + (", confusion.png" if _HAS_MPL else ""))


_HAS_MPL = False
try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    _HAS_MPL = True
except Exception:  # noqa: BLE001
    pass


def _save_confusion_png(all_metrics: list[dict], out: Path) -> None:
    if not _HAS_MPL:
        return
    k = len(all_metrics)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.0), squeeze=False)
    for ax, m in zip(axes[0], all_metrics):
        c = m["confusion"]
        mat = [[c["A"]["A"], c["A"]["B"]], [c["B"]["A"], c["B"]["B"]]]
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1], ["pred A", "pred B"])
        ax.set_yticks([0, 1], ["true A", "true B"])
        ax.set_title(f"{m['method']}\nacc={m['accuracy']*100:.0f}%")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)
    plt.close(fig)


def build_predictors(methods: list[str], data: Path, refs_per_class: int,
                     weights: Path | None = None):
    """method 名 -> predict(path)->'A'|'B' の辞書を作る。

    weights を指定すると cnn はその重みを使う（既定の results/cnn.pt を上書きせず
    別データ学習の重みを評価できる）。None なら classify 既定（results/cnn.pt）。
    """
    predictors = {}
    if "fewshot" in methods or "zeroshot" in methods:
        if not nc.ping():
            raise SystemExit(
                "llama-server に接続できません。`bash scripts/serve_nemotron.sh` を起動してください。"
            )
    if "fewshot" in methods:
        refs_a, refs_b = load_refs(data, refs_per_class)
        predictors["fewshot"] = lambda p: nc.judge(str(p), refs_a=refs_a, refs_b=refs_b)
    if "zeroshot" in methods:
        predictors["zeroshot"] = lambda p: nc.judge(str(p))
    if "cnn" in methods:
        if weights is not None:
            from train_cnn import load_classifier, predict_label

            m, classes, tf, dev = load_classifier(weights)
            predictors["cnn"] = lambda p: predict_label(m, classes, tf, dev, str(p))
        else:
            from classify import classify

            predictors["cnn"] = lambda p: classify(str(p), backend="cnn")
    return predictors


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate twin classifiers (M5)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--methods", default="fewshot,zeroshot",
                    help="カンマ区切り: fewshot,zeroshot,cnn")
    ap.add_argument("--with-cnn", action="store_true", help="cnn を手法に追加")
    ap.add_argument("--weights", type=Path, default=None,
                    help="cnn の重みパス（既定 results/cnn.pt を上書きせず別重みで評価）")
    ap.add_argument("--refs-per-class", type=int, default=2, help="few-shot 見本枚数/クラス")
    ap.add_argument("--limit", type=int, default=None, help="val のクラス毎上限（高速確認用）")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.with_cnn and "cnn" not in methods:
        methods.append("cnn")

    samples = load_val(args.data, args.limit)
    predictors = build_predictors(methods, args.data, args.refs_per_class, args.weights)

    all_metrics = []
    for method in methods:
        if method not in predictors:
            print(f"[skip] {method}: 予測器なし")
            continue
        print(f"\n--- {method} ---")
        res = run_method(method, predictors[method], samples)
        all_metrics.append(metrics(res))

    if all_metrics:
        print_report(all_metrics)
        save_results(all_metrics, args.out)


if __name__ == "__main__":
    main()
