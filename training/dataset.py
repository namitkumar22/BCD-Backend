"""Dataset and transforms for BreakHis histopathology patches."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.core.architecture import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class BreakHisDataset(Dataset):
    """Reads a manifest CSV with columns: path, label, patient, split."""

    def __init__(self, manifest: str | Path, split: str, train: bool) -> None:
        self.root = Path(manifest).parent
        self.transform = build_transforms(train)
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not rows:
            raise ValueError(f"No rows for split '{split}' in {manifest}")
        self.rows = rows
        self.labels = [0 if row["label"] == "benign" else 1 for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(image), self.labels[index]
