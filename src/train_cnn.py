"""高速 CNN 判定器の学習・推論 (M4).

MobileNetV3-small（ImageNet 事前学習）の分類ヘッドを 2 クラス(A/B)に付け替え、
ImageFolder（既定 data/synthetic/{train,val}/{A,B}）で学習する。Nemotron VLM
（~数百ms〜数秒/枚）に対し、CNN はサブミリ秒で判定できる「速い実行系」。
評価 (evaluate.py) の cnn 手法、リアルタイム化 (realtime.py) の速度モードに使う。

双子の識別は微細な左右非対称が手がかりなので、**左右反転 augmentation は入れない**
（face_align と同じ方針。鏡像は識別手がかりを消す）。

重みは results/cnn.pt に保存し、リポジトリにはコミットしない（.gitignore: *.pt）。
ROCm では device 名は "cuda"（HSA_OVERRIDE_GFX_VERSION 設定済み前提）。

CLI:
    python src/train_cnn.py                          # 学習して results/cnn.pt 保存
    python src/train_cnn.py --epochs 15 --batch-size 16
    python src/train_cnn.py --predict path/to/face.png   # 学習済みで1枚判定
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import data_config as _dc
except ImportError:  # パッケージ(src.train_cnn)として import された場合
    from . import data_config as _dc

ROOT = Path(__file__).resolve().parent.parent
# 既定データ/重みは TWIN_DATASET スイッチに従う（src/data_config.py）。
DEFAULT_DATA = _dc.data_dir()
DEFAULT_WEIGHTS = _dc.weights()
CLASSES = ("A", "B")  # ImageFolder はアルファベット順 → A=0, B=1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_transform(size: int) -> transforms.Compose:
    # 反転は禁止。明度/コントラスト等の軽い揺らぎのみ（生成側でも揺らしている）。
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def train(
    data: Path = DEFAULT_DATA,
    out: Path = DEFAULT_WEIGHTS,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 3e-4,
    size: int = 224,
    device_name: str | None = None,
) -> Path:
    device = _device(device_name)
    train_ds = datasets.ImageFolder(data / "train", transform=_train_transform(size))
    val_ds = datasets.ImageFolder(data / "val", transform=_eval_transform(size))
    if train_ds.classes != list(CLASSES):
        raise SystemExit(f"想定クラス {CLASSES} と不一致: {train_ds.classes}")
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"device={device} train={len(train_ds)} val={len(val_ds)} epochs={epochs}")
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        acc = _accuracy(model, val_ld, device)
        print(f"epoch {ep:>2}/{epochs}  train_loss={running/len(train_ds):.4f}  val_acc={acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "classes": list(CLASSES),
         "arch": "mobilenet_v3_small", "size": size, "val_acc": best_acc},
        out,
    )
    print(f"best val_acc={best_acc*100:.1f}% -> saved {out}")
    return out


# --- 推論（classify.py / evaluate.py / realtime.py から再利用）---

def load_classifier(weights: Path = DEFAULT_WEIGHTS, device_name: str | None = None):
    """学習済み重みを読み、(model, classes, transform, device) を返す。"""
    if not Path(weights).exists():
        raise SystemExit(f"重みがありません: {weights}（先に train_cnn.py で学習）")
    device = _device(device_name)
    ckpt = torch.load(weights, map_location=device)
    model = build_model(num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"], _eval_transform(ckpt.get("size", 224)), device


def _to_pil(image):
    from PIL import Image
    import numpy as np

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):  # BGR (cv2) 前提 → RGB
        return Image.fromarray(image[:, :, ::-1].copy())
    return image.convert("RGB")


@torch.no_grad()
def predict_label(model, classes, transform, device, image) -> str:
    """1 枚を "A"/"B" に判定する。"""
    x = transform(_to_pil(image)).unsqueeze(0).to(device)
    idx = int(model(x).argmax(1).item())
    return classes[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train / run fast CNN twin classifier (M4)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default=None, help='例 "cuda" / "cpu"')
    ap.add_argument("--predict", type=Path, default=None, help="学習せず1枚判定")
    args = ap.parse_args()

    if args.predict is not None:
        model, classes, tf, device = load_classifier(args.out, args.device)
        t = time.time()
        label = predict_label(model, classes, tf, device, args.predict)
        print(f"{args.predict} -> {label}  ({(time.time()-t)*1000:.1f}ms incl. load-warm)")
        return
    train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.size, args.device)


if __name__ == "__main__":
    main()
