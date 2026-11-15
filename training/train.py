"""Train a VGG19 breast-cancer classifier on BreakHis.

Two modes are available:

``frozen``
    ImageNet VGG19 convolutions are frozen and reused as a feature extractor;
    only the compact classifier head is trained. This runs comfortably on a CPU
    laptop and is the model shipped with the demo application.

``finetune``
    The last two convolutional blocks plus the head are fine-tuned end-to-end
    with mixed precision. Recommended on a GPU (Google Colab, ~15 epochs).

Examples
--------
    python training/train.py --mode frozen
    python training/train.py --mode finetune --epochs 15 --batch-size 32

Artifacts written to ``models/``:
    vgg19_breakhis.pt     best checkpoint consumed by the API
    metrics.json          test metrics, threshold and run metadata
    history.json          per-epoch training history
    test_predictions.csv  per-image predicted probabilities
    figures/*.png         publication-ready evaluation figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.architecture import (  # noqa: E402
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    build_model,
)
from app.core.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from training.dataset import BreakHisDataset, build_transforms  # noqa: E402

COLORS = {"benign": "#2E86AB", "malignant": "#D1495B"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / max(len(y_true), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    pr_curve = precision_recall_curve(y_true, y_prob, pos_label=1)
    pr_auc = average_precision_score(y_true, y_prob)

    # 95% Wilson confidence interval for accuracy.
    n = len(y_true)
    z = 1.959963984540054
    denom = 1 + z**2 / n
    centre = (accuracy + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(accuracy * (1 - accuracy) / n + z**2 / (4 * n**2)) / denom

    return {
        "accuracy": round(float(accuracy), 4),
        "accuracy_ci95": [round(float(centre - margin), 4), round(float(centre + margin), 4)],
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": round(float(threshold), 4),
        "n_samples": int(n),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": pr_curve[0].tolist(), "recall": pr_curve[1].tolist()},
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J statistic, computed on the validation split only."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    scores = tpr - fpr
    index = int(np.argmax(scores))
    return float(np.clip(thresholds[index], 0.05, 0.95))


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probs)


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gathered: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, targets in loader:
        activations = model.features(images.to(device))
        pooled = model.pool(activations).flatten(1)
        gathered.append(pooled.cpu())
        labels.append(targets)
    return torch.cat(gathered), torch.cat(labels)


def train_head_on_features(
    model: nn.Module,
    manifest: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    augment_copies: int,
    patience: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Freeze the backbone, cache VGG19 features, then train the classifier head."""
    model.eval()
    model.features.requires_grad_(False)

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.time()
    # VGG19 activations are memory hungry on CPU, so cache features in small batches.
    extract_batch = min(batch_size, 16)
    for copy_index in range(augment_copies):
        loader = DataLoader(
            BreakHisDataset(manifest, "train", train=True),
            batch_size=extract_batch,
            shuffle=False,
            num_workers=0,
        )
        x, y = extract_features(loader, model, device)
        features.append(x)
        labels.append(y)
        print(f"  extracted augmentation pass {copy_index + 1}/{augment_copies} "
              f"({len(y)} images, {time.time() - started:.0f}s)")

    train_x = torch.cat(features)
    train_y = torch.cat(labels)
    val_loader = DataLoader(BreakHisDataset(manifest, "val", train=False), batch_size=extract_batch)
    val_x, val_y = extract_features(val_loader, model, device)

    head = model.classifier
    head.train()
    class_counts = torch.bincount(train_y, minlength=2).float()
    class_weights = (len(train_y) / (2 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_auc": []}
    best_auc, best_state, stale = -1.0, None, 0
    generator = torch.Generator().manual_seed(42)

    for epoch in range(1, epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        total_loss, correct = 0.0, 0
        for start in range(0, len(train_x), batch_size):
            index = permutation[start : start + batch_size]
            batch_x = train_x[index].to(device)
            batch_y = train_y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            correct += (logits.argmax(1) == batch_y).sum().item()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x.to(device))
            val_loss = criterion(val_logits, val_y.to(device)).item()
            val_prob = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(int)
            val_acc = float((val_pred == val_y.numpy()).mean())
            fpr, tpr, _ = roc_curve(val_y.numpy(), val_prob, pos_label=1)
            val_auc = float(auc(fpr, tpr))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(total_loss / len(train_x), 4))
        history["train_accuracy"].append(round(correct / len(train_x), 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_auc"].append(round(val_auc, 4))
        print(f"  epoch {epoch:02d}/{epochs}  train_loss={total_loss / len(train_x):.4f}  "
              f"train_acc={correct / len(train_x):.4f}  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stopping at epoch {epoch}")
                break

    head.load_state_dict(best_state)
    return history, val_x, val_y


def finetune(
    model: nn.Module,
    manifest: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fine-tune the last two VGG19 blocks and the classifier head."""
    for parameter in model.features[:24].parameters():
        parameter.requires_grad = False

    train_loader = DataLoader(
        BreakHisDataset(manifest, "train", train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(BreakHisDataset(manifest, "val", train=False), batch_size=batch_size)

    class_counts = torch.bincount(torch.tensor(BreakHisDataset(manifest, "train", False).labels), minlength=2).float()
    class_weights = (class_counts.sum() / (2 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.features[24:].parameters() if p.requires_grad], "lr": lr / 10},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_auc": []}
    best_auc, best_state, stale = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(1) == targets).sum().item()
        scheduler.step()

        val_y, val_prob = predict_loader(model, val_loader, device)
        val_logits = torch.logit(torch.tensor(val_prob).clamp(1e-6, 1 - 1e-6))
        val_loss = nn.functional.binary_cross_entropy(val_logits, torch.tensor(val_y, dtype=torch.float32)).item()
        val_acc = float(((val_prob >= 0.5).astype(int) == val_y).mean())
        fpr, tpr, _ = roc_curve(val_y, val_prob, pos_label=1)
        val_auc = float(auc(fpr, tpr))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(total_loss / len(train_loader.dataset), 4))
        history["train_accuracy"].append(round(correct / len(train_loader.dataset), 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_auc"].append(round(val_auc, 4))
        print(f"  epoch {epoch:02d}/{epochs}  train_loss={total_loss / len(train_loader.dataset):.4f}  "
              f"train_acc={correct / len(train_loader.dataset):.4f}  val_acc={val_acc:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    val_y, val_prob = predict_loader(model, val_loader, device)
    return history, val_y, val_prob


def plot_training_curves(history: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="Training", color="#1F4E79", linewidth=1.6)
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="#D1495B", linewidth=1.6, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("(a) Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_accuracy"], label="Training", color="#1F4E79", linewidth=1.6)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation", color="#D1495B", linewidth=1.6, linestyle="--")
    axes[1].plot(epochs, history["val_auc"], label="Validation AUC", color="#4C956C", linewidth=1.4, linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title("(b) Accuracy and AUC")
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(matrix: dict, path: Path) -> None:
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    figure, axis = plt.subplots(figsize=(3.4, 3.0))
    axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            colour = "white" if values[row, column] > values.max() / 2 else "#1F2937"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=12, color=colour)
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.spines[:].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(metrics: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(3.6, 3.2))
    axis.plot(metrics["roc"]["fpr"], metrics["roc"]["tpr"], color="#1F4E79", linewidth=1.8,
              label=f"VGG19 (AUC = {metrics['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Receiver operating characteristic")
    axis.legend(frameon=False, loc="lower right", fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pr(metrics: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(3.6, 3.2))
    axis.plot(metrics["pr"]["recall"], metrics["pr"]["precision"], color="#D1495B", linewidth=1.8,
              label=f"VGG19 (AP = {metrics['pr_auc']:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curve")
    axis.legend(frameon=False, loc="lower left", fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(manifest: Path, path: Path) -> None:
    counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["split"]][0 if row["label"] == "benign" else 1] += 1
    splits = ["train", "val", "test"]
    benign = [counts[split][0] for split in splits]
    malignant = [counts[split][1] for split in splits]
    x = np.arange(len(splits))
    figure, axis = plt.subplots(figsize=(4.2, 2.8))
    axis.bar(x - 0.18, benign, 0.36, label="Benign", color=COLORS["benign"])
    axis.bar(x + 0.18, malignant, 0.36, label="Malignant", color=COLORS["malignant"])
    for index, value in enumerate(benign):
        axis.text(index - 0.18, value + 8, str(value), ha="center", fontsize=7.5)
    for index, value in enumerate(malignant):
        axis.text(index + 0.18, value + 8, str(value), ha="center", fontsize=7.5)
    axis.set_xticks(x, ["Training", "Validation", "Test"])
    axis.set_ylabel("Number of patches")
    axis.set_ylim(0, max(malignant) * 1.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_samples(manifest: Path, path: Path) -> None:
    root = manifest.parent
    rows: dict[str, list[dict]] = {"benign": [], "malignant": []}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and len(rows[row["label"]]) < 4:
                rows[row["label"]].append(row)
    figure, axes = plt.subplots(2, 4, figsize=(8.0, 4.1))
    for row_index, label in enumerate(("benign", "malignant")):
        for column, row in enumerate(rows[label]):
            image = Image.open(root / row["path"]).convert("RGB")
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
    figure.text(0.02, 0.73, "Benign", rotation=90, va="center", fontsize=10, color=COLORS["benign"])
    figure.text(0.02, 0.29, "Malignant", rotation=90, va="center", fontsize=10, color=COLORS["malignant"])
    figure.tight_layout(rect=(0.04, 0, 1, 1))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_gradcam(
    model: nn.Module,
    manifest: Path,
    predictions: list[dict],
    path: Path,
    device: torch.device,
) -> None:
    root = manifest.parent
    # Balance the panel: two correctly classified benign and two malignant patches.
    benign = [row for row in predictions if row["correct"] and row["true"] == 0][:2]
    malignant = [row for row in predictions if row["correct"] and row["true"] == 1][:2]
    picks = benign + malignant
    if len(picks) < 4:  # pragma: no cover
        picks = predictions[:4]
    cam = GradCAM(model)
    figure, axes = plt.subplots(3, len(picks), figsize=(2.0 * len(picks), 5.6))
    for index, row in enumerate(picks):
        image = Image.open(root / row["path"]).convert("RGB")
        original = np.array(image.resize((INPUT_SIZE, INPUT_SIZE)))
        tensor = build_transforms(False)(image).unsqueeze(0).to(device)
        heatmap = cam.generate(tensor, class_index=row["predicted"])
        overlay = overlay_heatmap(original, heatmap)

        axes[0, index].imshow(original)
        axes[0, index].set_title(CLASS_NAMES[row["true"]].capitalize(), fontsize=8)
        axes[1, index].imshow(heatmap, cmap="jet")
        axes[2, index].imshow(overlay)
        for axis in axes[:, index]:
            axis.axis("off")
        axes[2, index].set_xlabel(f"p = {row['probability']:.2f}", fontsize=7)
    for axis, label in zip(axes[:, 0], ("Input patch", "Grad-CAM", "Overlay")):
        axis.set_ylabel(label, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/breakhis_40x_manifest.csv"))
    parser.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    parser.add_argument("--epochs", type=int, default=None, help="Defaults: 60 frozen, 15 finetune")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--augment-copies", type=int, default=3, help="Augmented feature passes (frozen mode)")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=BACKEND_ROOT / "models")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs or (60 if args.mode == "frozen" else 15)
    batch_size = args.batch_size or (256 if args.mode == "frozen" else 32)
    lr = args.lr or (1e-3 if args.mode == "frozen" else 3e-4)

    set_seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode} | Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    model = build_model(pretrained=True, torch_home=BACKEND_ROOT / "models" / "torch_cache").to(device)

    started = time.time()
    if args.mode == "frozen":
        history, val_x, val_y = train_head_on_features(
            model, args.manifest, device, epochs, batch_size, lr, args.augment_copies, args.patience
        )
        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model.classifier(val_x.to(device)), dim=1)[:, 1].cpu().numpy()
        val_labels = val_y.numpy()
        checkpoint = {
            "format_version": 1,
            "architecture": "vgg19",
            "mode": "frozen",
            "num_classes": 2,
            "class_names": CLASS_NAMES,
            "input_size": INPUT_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "classifier": model.classifier.state_dict(),
        }
    else:
        history, val_labels, val_prob = finetune(
            model, args.manifest, device, epochs, batch_size, lr, args.patience
        )
        checkpoint = {
            "format_version": 1,
            "architecture": "vgg19",
            "mode": "finetune",
            "num_classes": 2,
            "class_names": CLASS_NAMES,
            "input_size": INPUT_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "state_dict": model.state_dict(),
        }

    threshold = best_threshold(val_labels, val_prob)
    test_loader = DataLoader(
        BreakHisDataset(args.manifest, "test", train=False),
        batch_size=16,
        shuffle=False,
    )
    test_y, test_prob = predict_loader(model, test_loader, device)
    metrics = compute_metrics(test_y, test_prob, threshold)
    metrics.update(
        {
            "mode": args.mode,
            "epochs_trained": len(history["epoch"]),
            "train_seconds": round(time.time() - started, 1),
            "validation_accuracy_at_0.5": round(float(((val_prob >= 0.5).astype(int) == val_labels).mean()), 4),
            "validation_auc": round(
                float(auc(*roc_curve(val_labels, val_prob, pos_label=1)[:2])), 4
            ),
            "dataset": {
                "name": "BreakHis",
                "magnification": "40X",
                "split": "patient-level 70/15/15",
                "images": {"train": 1451, "val": 283, "test": 261},
                "patients": 82,
            },
        }
    )

    checkpoint["threshold"] = threshold
    checkpoint["metrics"] = metrics
    checkpoint["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checkpoint_path = args.out_dir / "vgg19_breakhis.pt"
    torch.save(checkpoint, checkpoint_path)

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    test_dataset = BreakHisDataset(args.manifest, "test", train=False)
    with (args.out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true", "probability", "predicted", "correct"])
        for row, label, probability in zip(test_dataset.rows, test_y, test_prob):
            predicted = int(probability >= threshold)
            writer.writerow([row["path"], int(label), f"{probability:.6f}", predicted, int(predicted == label)])

    if not args.skip_figures:
        predictions = [
            {
                "path": row["path"],
                "true": int(label),
                "probability": float(probability),
                "predicted": int(probability >= threshold),
                "correct": bool((probability >= threshold) == label),
            }
            for row, label, probability in zip(test_dataset.rows, test_y, test_prob)
        ]
        model.eval()
        plot_training_curves(history, figures_dir / "training_curves.png")
        plot_confusion(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
        plot_roc(metrics, figures_dir / "roc_curve.png")
        plot_pr(metrics, figures_dir / "pr_curve.png")
        plot_class_distribution(args.manifest, figures_dir / "class_distribution.png")
        plot_samples(args.manifest, figures_dir / "sample_images.png")
        plot_gradcam(model, args.manifest, predictions, figures_dir / "gradcam_examples.png", device)

    print("\nTest metrics")
    for key in ("accuracy", "accuracy_ci95", "sensitivity", "specificity", "precision", "f1_score", "roc_auc"):
        print(f"  {key:12s}: {metrics[key]}")
    print(f"  threshold   : {threshold:.4f}")
    print(f"  checkpoint  : {checkpoint_path}")
    print(f"  elapsed     : {metrics['train_seconds']}s")


if __name__ == "__main__":
    main()
