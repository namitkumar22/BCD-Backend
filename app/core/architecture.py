"""VGG19 model definition shared by training and inference.

The backbone is torchvision's VGG19 with ImageNet weights. Two training
strategies are supported:

* ``frozen``   - the convolutional backbone stays fixed and only the lightweight
                 classification head is trained. Fast, runs on a CPU laptop.
* ``finetune`` - the last two convolutional blocks and the head are fine-tuned
                 end-to-end (recommended on a GPU, e.g. Google Colab).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

CLASS_NAMES = ["benign", "malignant"]
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VGG19Classifier(nn.Module):
    """VGG19 feature extractor followed by a compact binary classifier head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = vgg19(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dimensional pooled backbone feature vector."""
        return self.pool(self.features(x)).flatten(1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Map a pooled feature vector to class logits."""
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embed(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    torch_home: str | Path | None = None,
) -> VGG19Classifier:
    """Create a VGG19 classifier with ImageNet-pretrained convolutional weights."""
    if torch_home is not None:
        torch.hub.set_dir(str(torch_home))
    model = VGG19Classifier(num_classes=num_classes)
    if pretrained:
        weights = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.state_dict()
        model.features.load_state_dict(weights)
    return model


def load_checkpoint(model: VGG19Classifier, checkpoint_path: str | Path) -> dict:
    """Load a training checkpoint produced by ``training/train.py``."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in blob:
        model.load_state_dict(blob["state_dict"])
    elif "classifier" in blob:
        model.classifier.load_state_dict(blob["classifier"])
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unrecognised checkpoint format: {checkpoint_path}")
    model.eval()
    return blob
