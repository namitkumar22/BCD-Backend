"""Build a patient-level split manifest for the BreakHis dataset.

BreakHis images are named like ``SOB_B_A-14-22549AB-40-001.jpg`` where
``14-22549AB`` identifies the patient/slide. Multiple patches from the same
patient appear in the dataset, so splitting at *image* level leaks information
between train and test. This script groups by patient and produces a 70/15/15
train/validation/test manifest with stratification on the benign/malignant label.

Usage:
    python training/prepare_dataset.py --data-dir data/breakhis
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path

MAGNIFICATIONS = {"40X", "100X", "200X", "400X"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_patient(image_path: Path) -> str:
    """Extract a patient identifier from the slide folder name."""
    slide_folder = image_path.parent.parent.name  # .../<slide>/40X/image.jpg
    return re.sub(r"[^A-Za-z0-9-]", "-", slide_folder)


def collect_samples(data_dir: Path, magnification: str) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for label in ("benign", "malignant"):
        class_dir = data_dir / "histology_slides" / "breast" / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if magnification not in path.parts:
                continue
            samples.append(
                {
                    "path": path.as_posix(),
                    "label": label,
                    "patient": parse_patient(path),
                    "magnification": magnification,
                }
            )
    return samples


def split_patients(
    samples: list[dict[str, str]], seed: int = 42
) -> dict[str, str]:
    """Assign whole patients to train/val/test, stratified by class."""
    by_class: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        by_class[sample["label"]].add(sample["patient"])

    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for label, patients in sorted(by_class.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        for index, patient in enumerate(patients):
            if index < n_test:
                assignment[patient] = "test"
            elif index < n_test + n_val:
                assignment[patient] = "val"
            else:
                assignment[patient] = "train"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/breakhis"))
    parser.add_argument("--magnification", default="40X", choices=sorted(MAGNIFICATIONS))
    parser.add_argument("--out", type=Path, default=Path("data/breakhis_40x_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = collect_samples(args.data_dir, args.magnification)
    if not samples:
        raise SystemExit(
            f"No images found for {args.magnification} in {args.data_dir}. "
            "Download BreakHis and extract it into data/breakhis first."
        )

    assignment = split_patients(samples, args.seed)
    for sample in samples:
        sample["split"] = assignment[sample["patient"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Store paths relative to the manifest location so the dataset can be moved
    # together with the manifest without rewriting every row.
    for sample in samples:
        sample["path"] = Path(os.path.relpath(sample["path"], args.out.parent)).as_posix()
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "patient", "magnification", "split"])
        writer.writeheader()
        writer.writerows(samples)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        counts[sample["split"]][sample["label"]] += 1

    print(f"Wrote {len(samples)} samples to {args.out}")
    print(f"Patients: {len(set(s['patient'] for s in samples))}")
    for split in ("train", "val", "test"):
        benign = counts[split]["benign"]
        malignant = counts[split]["malignant"]
        print(f"  {split:5s}: {benign + malignant:4d} images ({benign} benign, {malignant} malignant)")


if __name__ == "__main__":
    main()
