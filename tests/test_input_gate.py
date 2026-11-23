"""Tests for out-of-domain upload rejection.

Genuine BreakHis patches must still be accepted, while photographs and
non-microscopy images must be rejected before any prediction is produced.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.inference import ModelService
from app.services.input_gate import InputRejectedError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = BACKEND_ROOT / "data" / "breakhis_40x_manifest.csv"


@pytest.fixture(scope="module")
def service() -> ModelService:
    model_service = ModelService(Settings())
    model_service.load()
    if not model_service.ready:
        pytest.skip(f"Checkpoint unavailable: {model_service.error}")
    if not model_service.gate.ready:
        pytest.skip(f"Reference statistics unavailable: {model_service.gate.error}")
    return model_service


def _histopathology_rows(split: str, limit: int) -> list[dict]:
    if not MANIFEST.is_file():
        pytest.skip("BreakHis manifest is not available")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        pytest.skip(f"No '{split}' rows in the manifest")
    return rows[:limit]


@pytest.mark.parametrize("offset", range(0, 10, 2))
def test_real_histopathology_patches_are_accepted(service: ModelService, offset: int) -> None:
    rows = _histopathology_rows("test", 10)
    row = rows[offset]
    with Image.open(BACKEND_ROOT / "data" / row["path"]) as image:
        result = service.predict(image.convert("RGB"), want_grad_cam=False)
    assert result["predicted_label"] in {"benign", "malignant"}
    assert result["probability"] == pytest.approx(
        result["probabilities"][result["predicted_label"]]
    )
    assert result["input_check"]


def test_decision_follows_tuned_threshold(service: ModelService) -> None:
    """A malignant probability above the (sub-0.5) threshold must be labelled malignant."""
    rows = _histopathology_rows("test", 40)
    threshold = service.threshold
    checked = 0
    for row in rows:
        with Image.open(BACKEND_ROOT / "data" / row["path"]) as image:
            try:
                result = service.predict(image.convert("RGB"), want_grad_cam=False)
            except InputRejectedError:
                # The gate can legitimately reject a tail-case patch; the
                # threshold behaviour is tested on accepted predictions.
                continue
        expected = "malignant" if result["probabilities"]["malignant"] >= threshold else "benign"
        assert result["predicted_label"] == expected
        checked += 1
    assert checked >= 10


def test_face_photo_is_rejected(service: ModelService) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    face = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    if not face.is_file():
        pytest.skip("matplotlib sample data not installed")

    with Image.open(face) as image:
        with pytest.raises(InputRejectedError) as caught:
            service.predict(image.convert("RGB"), want_grad_cam=False)

    payload = caught.value.result.payload()
    assert payload["code"] == "invalid_input"
    assert payload["reasons"]
    assert payload["guidance"]


def test_blank_image_is_rejected(service: ModelService) -> None:
    with pytest.raises(InputRejectedError):
        service.predict(Image.new("RGB", (320, 240), "white"), want_grad_cam=False)


def test_rejection_payload_is_json_serialisable(service: ModelService) -> None:
    import json

    from app.core.input_quality import analyse_crop, model_crop

    image = Image.new("RGB", (320, 240), "white")
    metrics = analyse_crop(model_crop(image), original_size=image.size)
    verdict = service.gate.evaluate(metrics, np.zeros_like(service.gate.feature_mean))
    payload = verdict.payload()
    assert json.loads(json.dumps(payload))["reasons"]
