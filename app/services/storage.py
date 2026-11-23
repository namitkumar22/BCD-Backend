"""Upload validation and file storage helpers."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_DIMENSION = 64


class InvalidImageError(ValueError):
    """Raised for unsupported, corrupted or oversized uploads."""


def load_upload(payload: bytes, filename: str, settings: Settings) -> tuple[Image.Image, str]:
    """Validate an uploaded file and return a decoded RGB image and safe suffix."""
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise InvalidImageError(f"Image exceeds the {settings.max_upload_mb} MB limit.")

    suffix = Path(filename or "image.jpg").suffix.lower() or ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            "Unsupported file type. Please upload a JPG, PNG, WEBP, BMP or TIFF image."
        )

    try:
        Image.MAX_IMAGE_PIXELS = 60_000_000
        image = Image.open(io.BytesIO(payload))
        image.load()
        # Phone/camera EXIF orientation should not silently rotate a slide patch.
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file could not be read as a valid image.") from exc

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise InvalidImageError(f"Image must be at least {MIN_DIMENSION}x{MIN_DIMENSION} pixels.")
    return image.convert("RGB"), suffix


def save_upload(image: Image.Image, suffix: str, settings: Settings) -> str:
    """Persist the uploaded image and return its filename."""
    name = f"{uuid.uuid4().hex}{suffix if suffix != '.tiff' else '.tif'}"
    path = settings.storage_dir / "uploads" / name
    image.save(path)
    return name


def save_gradcam(overlay_png: bytes, settings: Settings) -> str:
    name = f"{uuid.uuid4().hex}_gradcam.png"
    (settings.storage_dir / "uploads" / name).write_bytes(overlay_png)
    return name
