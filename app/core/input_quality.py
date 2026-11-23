"""Image statistics used to verify that an upload is histopathology.

The checks in :mod:`app.services.input_gate` compare these descriptors with a
reference distribution estimated from the BreakHis training patches. Keeping the
crop geometry here in one place also guarantees that the quality metrics,
Grad-CAM overlay and the classifier all see exactly the same 224 x 224 crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

INPUT_SIZE = 224
RESIZE_SHORTER_SIDE = 256

# Colour descriptors, in the fixed order stored in the reference statistics.
COLOR_FEATURE_NAMES = (
    "saturation_mean",
    "saturated_fraction",
    "brightness_mean",
    "background_fraction",
    "purple_fraction",
    "pink_fraction",
    "colorfulness",
    "grayscale_fraction",
)


def model_crop(image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """Return the exact square crop the classifier receives.

    Mirrors ``Resize(256)`` + ``CenterCrop(224)`` from the training pipeline: the
    shorter side is scaled to 256 px and the centre 224 x 224 pixels are cropped.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= height:
        new_width, new_height = RESIZE_SHORTER_SIDE, int(RESIZE_SHORTER_SIDE * height / width)
    else:
        new_width, new_height = int(RESIZE_SHORTER_SIDE * width / height), RESIZE_SHORTER_SIDE
    resized = rgb.resize((max(1, new_width), max(1, new_height)), Image.Resampling.BILINEAR)
    left = int(round((resized.width - size) / 2.0))
    top = int(round((resized.height - size) / 2.0))
    return resized.crop((left, top, left + size, top + size))


def crop_to_tensor(crop: Image.Image, mean: list[float], std: list[float]):
    """Convert an RGB crop to a normalised ``(3, H, W)`` float tensor."""
    import torch

    array = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


@dataclass(frozen=True)
class ImageMetrics:
    """Measurements taken on the 224 x 224 analysis crop."""

    width: int
    height: int
    brightness_mean: float
    saturation_mean: float
    saturated_fraction: float
    dark_fraction: float
    clipped_bright_fraction: float
    background_fraction: float
    purple_fraction: float
    pink_fraction: float
    colorfulness: float
    grayscale_fraction: float
    sharpness: float

    def color_features(self) -> np.ndarray:
        """Return the colour descriptor vector in ``COLOR_FEATURE_NAMES`` order."""
        return np.asarray(
            [
                self.saturation_mean,
                self.saturated_fraction,
                self.brightness_mean,
                self.background_fraction,
                self.purple_fraction,
                self.pink_fraction,
                self.colorfulness,
                self.grayscale_fraction,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "width": self.width,
            "height": self.height,
            "brightness_mean": round(self.brightness_mean, 4),
            "saturation_mean": round(self.saturation_mean, 4),
            "saturated_fraction": round(self.saturated_fraction, 4),
            "dark_fraction": round(self.dark_fraction, 4),
            "clipped_bright_fraction": round(self.clipped_bright_fraction, 4),
            "background_fraction": round(self.background_fraction, 4),
            "purple_fraction": round(self.purple_fraction, 4),
            "pink_fraction": round(self.pink_fraction, 4),
            "colorfulness": round(self.colorfulness, 4),
            "grayscale_fraction": round(self.grayscale_fraction, 4),
            "sharpness": round(self.sharpness, 6),
        }


def analyse_crop(crop: Image.Image, original_size: tuple[int, int] | None = None) -> ImageMetrics:
    """Compute colour and focus statistics for an analysis crop.

    Hue is expressed in degrees. Saturated pixels (S >= 0.12) are split into the
    purple range typical of haematoxylin-stained nuclei (250-330 degrees) and the
    pink range typical of eosin-stained cytoplasm/connective tissue (330-360).
    """
    array = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]

    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    delta = maximum - minimum
    saturation = np.where(maximum > 0, delta / np.maximum(maximum, 1e-6), 0.0)

    coloured = delta > 1e-6
    safe_delta = np.where(coloured, delta, 1.0)
    red_chroma = np.where(coloured, (maximum - red) / safe_delta, 0.0)
    green_chroma = np.where(coloured, (maximum - green) / safe_delta, 0.0)
    blue_chroma = np.where(coloured, (maximum - blue) / safe_delta, 0.0)
    hue = np.where(
        maximum == red,
        blue_chroma - green_chroma,
        np.where(maximum == green, 2.0 + red_chroma - blue_chroma, 4.0 + green_chroma - red_chroma),
    )
    hue = np.where(coloured, (hue / 6.0) % 1.0 * 360.0, 0.0)

    saturated = saturation >= 0.12
    saturated_count = float(saturated.sum())
    purple = saturated & (hue >= 250.0) & (hue < 330.0)
    pink = saturated & (hue >= 330.0)

    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )

    grayscale = (maximum - minimum) <= (8.0 / 255.0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    laplacian = (
        4.0 * luminance[1:-1, 1:-1]
        - luminance[:-2, 1:-1]
        - luminance[2:, 1:-1]
        - luminance[1:-1, :-2]
        - luminance[1:-1, 2:]
    )

    width, height = original_size or crop.size
    total = max(float(array.shape[0] * array.shape[1]), 1.0)
    return ImageMetrics(
        width=int(width),
        height=int(height),
        brightness_mean=float(maximum.mean()),
        saturation_mean=float(saturation.mean()),
        saturated_fraction=float(saturated.sum() / total),
        dark_fraction=float((maximum < 0.12).sum() / total),
        clipped_bright_fraction=float((maximum > 0.985).sum() / total),
        background_fraction=float(((maximum > 0.88) & (saturation < 0.10)).sum() / total),
        purple_fraction=float(purple.sum() / saturated_count) if saturated_count else 0.0,
        pink_fraction=float(pink.sum() / saturated_count) if saturated_count else 0.0,
        colorfulness=colorfulness,
        grayscale_fraction=float(grayscale.sum() / total),
        sharpness=float(laplacian.var()),
    )
