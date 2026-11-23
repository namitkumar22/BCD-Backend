"""Build reference statistics for the input-domain (OOD) gate.

The API rejects uploads that do not resemble the BreakHis patches the classifier
was trained on. Two reference distributions are estimated here:

* a 512-dimensional VGG19 feature embedding (Mahalanobis distance), and
* an 8-dimensional colour/stain descriptor of the 224x224 analysis crop.

Run after training, or whenever the backbone/checkpoint changes:

    python training/build_reference_stats.py

Artifacts written to ``models/``:

    reference_stats.npz   means, precision matrices and thresholds read by the API
    reference_stats.json  human-readable calibration report

The model itself is never trained or modified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.architecture import IMAGENET_MEAN, IMAGENET_STD, build_model, load_checkpoint  # noqa: E402
from app.core.input_quality import (  # noqa: E402
    COLOR_FEATURE_NAMES,
    analyse_crop,
    crop_to_tensor,
    model_crop,
)

METRIC_FIELDS = (
    "brightness_mean",
    "saturation_mean",
    "saturated_fraction",
    "dark_fraction",
    "clipped_bright_fraction",
    "background_fraction",
    "purple_fraction",
    "pink_fraction",
    "colorfulness",
    "grayscale_fraction",
    "sharpness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(BACKEND_ROOT / "data" / "breakhis_40x_manifest.csv"))
    parser.add_argument("--model", default=str(BACKEND_ROOT / "models" / "vgg19_breakhis.pt"))
    parser.add_argument("--out-dir", default=str(BACKEND_ROOT / "models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--feature-percentile",
        type=float,
        default=99.0,
        help="Percentile of validation domain distances used as the rejection threshold.",
    )
    parser.add_argument(
        "--stain-percentile",
        type=float,
        default=99.0,
        help="Percentile of validation stain distances used as the rejection threshold.",
    )
    parser.add_argument(
        "--cache",
        default=str(BACKEND_ROOT / "models" / ".reference_feature_cache.npz"),
        help="Scratch cache so thresholds can be re-tuned without re-extracting features.",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--probe", action="append", default=[], help="Extra image to score, as name=path.")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits: dict[str, list[dict]] = {}
    for row in rows:
        splits.setdefault(row["split"], []).append(row)
    return splits


def load_metrics(splits: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    for split, rows in splits.items():
        values = []
        for row in rows:
            with Image.open(BACKEND_ROOT / "data" / row["path"]) as image:
                crop = model_crop(image)
                values.append(analyse_crop(crop, original_size=image.size).as_dict())
        metrics[split] = np.asarray([[item[name] for name in METRIC_FIELDS] for item in values], dtype=np.float64)
    return metrics


@torch.no_grad()
def extract_features(
    splits: dict[str, list[dict]], model, device: torch.device, batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    features: dict[str, np.ndarray] = {}
    colors: dict[str, np.ndarray] = {}
    metrics: dict[str, np.ndarray] = {}

    for split, rows in splits.items():
        feature_batches: list[np.ndarray] = []
        color_rows: list[np.ndarray] = []
        metric_rows: list[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            crops = []
            for row in batch:
                with Image.open(BACKEND_ROOT / "data" / row["path"]) as image:
                    crop = model_crop(image)
                    metric = analyse_crop(crop, original_size=image.size)
                crops.append(crop)
                color_rows.append(metric.color_features())
                metric_rows.append([metric.as_dict()[name] for name in METRIC_FIELDS])
            tensors = torch.stack([crop_to_tensor(crop, IMAGENET_MEAN, IMAGENET_STD) for crop in crops]).to(device)
            embedded = model.embed(tensors).cpu().numpy().astype(np.float32)
            feature_batches.append(embedded)
            done = min(start + batch_size, len(rows))
            print(f"  {split}: {done}/{len(rows)}", flush=True)
        features[split] = np.concatenate(feature_batches, axis=0)
        colors[split] = np.asarray(color_rows, dtype=np.float64)
        metrics[split] = np.asarray(metric_rows, dtype=np.float64)
    return features, colors, metrics


def fit_reference(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    estimator = LedoitWolf().fit(data)
    return estimator.location_.astype(np.float64), estimator.precision_.astype(np.float64)


def distances(data: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    difference = data - mean
    squared = np.einsum("ij,jk,ik->i", difference, precision, difference)
    return np.sqrt(np.maximum(squared, 0.0))


def probe_images(extra: list[str]) -> list[tuple[str, Path]]:
    probes: list[tuple[str, Path]] = []
    try:
        import matplotlib

        sample_dir = Path(matplotlib.get_data_path()) / "sample_data"
        for name, filename in (
            ("face (Grace Hopper)", "grace_hopper.jpg"),
            ("photograph (Minduka)", "Minduka_Present_Blue_Pack.png"),
            ("logo (matplotlib)", "logo2.png"),
        ):
            if (sample_dir / filename).is_file():
                probes.append((name, sample_dir / filename))
    except Exception:  # pragma: no cover - matplotlib is a project dependency
        pass

    # Synthetic non-histology probes.
    from PIL import ImageDraw

    sample_dir = BACKEND_ROOT / ".tmp" / "probes"
    sample_dir.mkdir(parents=True, exist_ok=True)
    document = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(document)
    rng = np.random.default_rng(0)
    for index in range(18):
        draw.rectangle(
            [60, 50 + index * 34, 60 + int(rng.integers(120, 760)), 62 + index * 34],
            fill=(int(rng.integers(20, 80)),) * 3,
        )
    document_path = sample_dir / "document.png"
    document.save(document_path)
    noise_path = sample_dir / "noise.png"
    Image.fromarray((rng.random((224, 224, 3)) * 255).astype(np.uint8)).save(noise_path)
    probes += [("synthetic document", document_path), ("synthetic noise", noise_path)]

    for item in extra:
        name, _, raw_path = item.partition("=")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"Probe not found: {path}")
        probes.append((name or path.stem, path))
    return probes


def probe_scores(
    probes: list[tuple[str, Path]], model, device: torch.device, color_mean, color_precision, feature_mean, feature_precision
) -> list[dict]:
    results = []
    with torch.no_grad():
        for name, path in probes:
            with Image.open(path) as image:
                crop = model_crop(image)
                metric = analyse_crop(crop, original_size=image.size)
                tensor = crop_to_tensor(crop, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
            feature = model.embed(tensor).cpu().numpy()[0].astype(np.float64)
            results.append(
                {
                    "name": name,
                    "path": str(path),
                    "stain_distance": round(float(distances(metric.color_features()[None, :], color_mean, color_precision)[0]), 2),
                    "domain_distance": round(float(distances(feature[None, :], feature_mean, feature_precision)[0]), 2),
                    "metrics": metric.as_dict(),
                }
            )
    return results


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    manifest_path = Path(args.manifest)
    splits = read_manifest(manifest_path)
    cache_path = Path(args.cache)

    if args.reuse_cache and cache_path.is_file():
        print(f"Reusing cached features: {cache_path}")
        cached = np.load(cache_path)
        features = {split: cached[f"{split}__features"] for split in splits}
        colors = {split: cached[f"{split}__colors"] for split in splits}
        metrics = {split: cached[f"{split}__metrics"] for split in splits}
    else:
        model = build_model(pretrained=True, torch_home=BACKEND_ROOT / "models" / "torch_cache")
        load_checkpoint(model, args.model)
        model.to(device).eval()
        print("Extracting VGG19 features and colour metrics ...", flush=True)
        features, colors, metrics = extract_features(splits, model, device, args.batch_size)
        np.savez_compressed(
            cache_path,
            **{f"{split}__features": features[split] for split in splits},
            **{f"{split}__colors": colors[split] for split in splits},
            **{f"{split}__metrics": metrics[split] for split in splits},
        )

    model = build_model(pretrained=True, torch_home=BACKEND_ROOT / "models" / "torch_cache")
    load_checkpoint(model, args.model)
    model.to(device).eval()

    reference = "train"
    if reference not in features:
        raise SystemExit("The manifest must contain a 'train' split.")

    feature_mean, feature_precision = fit_reference(features[reference])
    color_mean, color_precision = fit_reference(colors[reference])

    feature_distances = {split: distances(values, feature_mean, feature_precision) for split, values in features.items()}
    color_distances = {split: distances(values, color_mean, color_precision) for split, values in colors.items()}

    calibration_split = "val" if "val" in feature_distances else reference
    feature_threshold = float(np.percentile(feature_distances[calibration_split], args.feature_percentile))
    color_threshold = float(np.percentile(color_distances[calibration_split], args.stain_percentile))

    sharpness_index = METRIC_FIELDS.index("sharpness")
    sharpness = metrics[reference][:, sharpness_index]
    quality_thresholds = {
        "min_sharpness": float(round(np.percentile(sharpness, 0.1) * 0.5, 8)),
        "min_brightness": 0.15,
        "max_brightness": 0.99,
        "max_dark_fraction": 0.85,
        "max_clipped_bright_fraction": 0.90,
        "max_grayscale_fraction": 0.995,
        "max_grayscale_colorfulness": 0.012,
    }

    splits_report = {}
    for split in features:
        splits_report[split] = {
            "n": int(len(features[split])),
            "feature_distance_percentiles": {
                str(p): round(float(np.percentile(feature_distances[split], p)), 2) for p in (50, 90, 95, 99, 99.9)
            },
            "stain_distance_percentiles": {
                str(p): round(float(np.percentile(color_distances[split], p)), 2) for p in (50, 90, 95, 99, 99.9)
            },
            "false_reject_rate_at_threshold": round(
                float((feature_distances[split] > feature_threshold).mean()), 4
            ),
            "false_reject_rate_from_stain": round(float((color_distances[split] > color_threshold).mean()), 4),
        }

    # Hard acquisition checks must not reject genuine patches; report how they behave.
    reference_metrics = metrics[reference]
    brightness = reference_metrics[:, METRIC_FIELDS.index("brightness_mean")]
    dark = reference_metrics[:, METRIC_FIELDS.index("dark_fraction")]
    bright = reference_metrics[:, METRIC_FIELDS.index("clipped_bright_fraction")]
    grayscale = reference_metrics[:, METRIC_FIELDS.index("grayscale_fraction")]
    print(
        "Reference ranges: "
        f"brightness {brightness.min():.3f}-{brightness.max():.3f}, "
        f"max dark {dark.max():.3f}, max clipped {bright.max():.3f}, "
        f"max grayscale {grayscale.max():.3f}, sharpness p0.1 {np.percentile(sharpness, 0.1):.6f}"
    )

    probes = probe_scores(
        probe_images(args.probe), model, device, color_mean, color_precision, feature_mean, feature_precision
    )
    for probe in probes:
        probe["rejected_by_stain"] = probe["stain_distance"] > color_threshold
        probe["rejected_by_domain"] = probe["domain_distance"] > feature_threshold
        print(
            f"Probe {probe['name']}: stain {probe['stain_distance']:.1f}/{color_threshold:.1f} "
            f"domain {probe['domain_distance']:.1f}/{feature_threshold:.1f} "
            f"rejected={probe['rejected_by_stain'] or probe['rejected_by_domain']}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    np.savez_compressed(
        out_dir / "reference_stats.npz",
        version=np.array("1"),
        generated_at=np.array(generated_at),
        feature_mean=feature_mean.astype(np.float32),
        feature_precision=feature_precision.astype(np.float32),
        feature_threshold=np.array(feature_threshold),
        color_mean=color_mean.astype(np.float32),
        color_precision=color_precision.astype(np.float32),
        color_threshold=np.array(color_threshold),
        color_feature_names=np.asarray(COLOR_FEATURE_NAMES),
        quality_thresholds=np.array(json.dumps(quality_thresholds)),
        n_reference=np.array(len(features[reference])),
        feature_percentile=np.array(args.feature_percentile),
        stain_percentile=np.array(args.stain_percentile),
    )

    report = {
        "version": 1,
        "generated_at": generated_at,
        "checkpoint": str(Path(args.model)),
        "manifest": str(manifest_path),
        "reference_split": reference,
        "n_reference": int(len(features[reference])),
        "feature_dim": int(features[reference].shape[1]),
        "color_features": list(COLOR_FEATURE_NAMES),
        "thresholds": {
            "feature_threshold": round(feature_threshold, 4),
            "feature_percentile": args.feature_percentile,
            "stain_threshold": round(color_threshold, 4),
            "stain_percentile": args.stain_percentile,
            "calibration_split": calibration_split,
            "quality": quality_thresholds,
        },
        "splits": splits_report,
        "probes": probes,
    }
    (out_dir / "reference_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'reference_stats.npz'} and {out_dir / 'reference_stats.json'}")
    print(
        "Feature threshold "
        f"{feature_threshold:.2f} ({args.feature_percentile}th percentile of {calibration_split}); "
        f"stain threshold {color_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
