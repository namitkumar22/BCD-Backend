"""Re-evaluate a saved checkpoint and regenerate all evaluation artifacts.

Useful when the checkpoint already exists and only the reports/figures need to be
refreshed. The threshold is always recomputed on the validation split, never on
the test set.

Usage:
    python training/evaluate.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.architecture import build_model, load_checkpoint  # noqa: E402
from training.dataset import BreakHisDataset  # noqa: E402
from training.train import (  # noqa: E402
    best_threshold,
    compute_metrics,
    plot_class_distribution,
    plot_confusion,
    plot_gradcam,
    plot_pr,
    plot_roc,
    plot_samples,
    plot_training_curves,
    predict_loader,
)


def main() -> None:
    models_dir = BACKEND_ROOT / "models"
    figures_dir = models_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest = BACKEND_ROOT / "data" / "breakhis_40x_manifest.csv"
    checkpoint_path = models_dir / "vgg19_breakhis.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(pretrained=True, torch_home=models_dir / "torch_cache")
    blob = load_checkpoint(model, checkpoint_path)
    model.eval()
    device = torch.device("cpu")

    val_loader = DataLoader(BreakHisDataset(manifest, "val", train=False), batch_size=16)
    test_loader = DataLoader(BreakHisDataset(manifest, "test", train=False), batch_size=16)

    val_y, val_prob = predict_loader(model, val_loader, device)
    threshold = best_threshold(val_y, val_prob)
    test_y, test_prob = predict_loader(model, test_loader, device)
    metrics = compute_metrics(test_y, test_prob, threshold)
    metrics.update(
        {
            "mode": blob.get("mode", "frozen"),
            "epochs_trained": blob.get("metrics", {}).get("epochs_trained"),
            "train_seconds": blob.get("metrics", {}).get("train_seconds"),
            "validation_accuracy_at_0.5": round(float(((val_prob >= 0.5).astype(int) == val_y).mean()), 4),
            "validation_auc": round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(val_y, val_prob)), 4),
            "dataset": blob.get("metrics", {}).get(
                "dataset",
                {
                    "name": "BreakHis",
                    "magnification": "40X",
                    "split": "patient-level 70/15/15",
                    "images": {"train": 1451, "val": 283, "test": 261},
                    "patients": 82,
                },
            ),
        }
    )

    blob["threshold"] = threshold
    blob["metrics"] = metrics
    torch.save(blob, checkpoint_path)
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"threshold={threshold:.4f} accuracy={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f}")

    dataset = BreakHisDataset(manifest, "test", train=False)
    predictions = []
    with (models_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true", "probability", "predicted", "correct"])
        for row, label, probability in zip(dataset.rows, test_y, test_prob):
            predicted = int(probability >= threshold)
            writer.writerow([row["path"], int(label), f"{probability:.6f}", predicted, int(predicted == label)])
            predictions.append(
                {
                    "path": row["path"],
                    "true": int(label),
                    "probability": float(probability),
                    "predicted": predicted,
                    "correct": predicted == int(label),
                }
            )

    history_path = models_dir / "history.json"
    if history_path.is_file():
        plot_training_curves(json.loads(history_path.read_text(encoding="utf-8")), figures_dir / "training_curves.png")
    plot_confusion(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    plot_roc(metrics, figures_dir / "roc_curve.png")
    plot_pr(metrics, figures_dir / "pr_curve.png")
    plot_class_distribution(manifest, figures_dir / "class_distribution.png")
    plot_samples(manifest, figures_dir / "sample_images.png")
    plot_gradcam(model, manifest, predictions, figures_dir / "gradcam_examples.png", device)
    print("figures refreshed")


if __name__ == "__main__":
    main()
