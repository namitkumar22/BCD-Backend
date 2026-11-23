"""Input-domain verification for uploaded histopathology images.

A classifier trained on BreakHis patches will happily assign a benign/malignant
probability to a face, a document or a screenshot. This gate runs *before* the
classifier and rejects images that are clearly outside the input domain.

Two calibrated distances are used, both estimated from the training patches by
``training/build_reference_stats.py``:

1. **Deep domain distance** - Mahalanobis distance of the 512-dimensional VGG19
   feature vector from the BreakHis feature distribution. This is the primary
   signal: natural photographs (faces, objects, scenes) fall far outside it.
2. **Stain-colour distance** - Mahalanobis distance in an 8-dimensional HSV /
   colourfulness / greyness descriptor space, which catches images that the
   feature space might accept but that clearly lack H&E staining.

Conservative acquisition checks (resolution, exposure, focus, greyscale) run
first and give specific, user-actionable messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.input_quality import ImageMetrics

GUIDANCE = (
    "Upload a single H&E-stained breast histopathology patch at 40X magnification "
    "(BreakHis-style JPG, PNG or TIFF). Photographs, faces, documents, screenshots "
    "and other non-microscopy images are rejected before the model runs."
)

# Used until the calibrated reference statistics are available.
DEFAULT_QUALITY_THRESHOLDS = {
    "min_sharpness": 0.0008,
    "min_brightness": 0.15,
    "max_brightness": 0.99,
    "max_dark_fraction": 0.85,
    "max_clipped_bright_fraction": 0.90,
    "max_grayscale_fraction": 0.995,
    "max_grayscale_colorfulness": 0.012,
}


class InputRejectedError(RuntimeError):
    """Raised when an upload fails the input-domain verification."""

    def __init__(self, result: "ScreeningResult") -> None:
        self.result = result
        super().__init__(result.message)


@dataclass
class ScreeningResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def payload(self) -> dict:
        """JSON body returned with HTTP 422 for a rejected upload."""
        return {
            "code": "invalid_input",
            "message": self.message
            or (
                "The image was rejected before analysis because it does not look like "
                "an H&E-stained breast tissue patch."
            ),
            "reasons": self.reasons,
            "guidance": GUIDANCE,
            "details": self.details,
            "checks": self.checks,
        }


def _mahalanobis(vector: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    difference = np.asarray(vector, dtype=np.float64) - mean
    squared = float(difference @ precision @ difference)
    return float(np.sqrt(max(squared, 0.0)))


class InputGate:
    """Calibrated histopathology-domain gate for uploaded images."""

    def __init__(self, stats_path: Path, enabled: bool = True) -> None:
        self.stats_path = Path(stats_path)
        self.enabled = enabled
        self.error: str | None = None
        self.reference_patches = 0
        self.feature_mean: np.ndarray | None = None
        self.feature_precision: np.ndarray | None = None
        self.feature_threshold: float | None = None
        self.color_mean: np.ndarray | None = None
        self.color_precision: np.ndarray | None = None
        self.color_threshold: float | None = None
        self.quality = dict(DEFAULT_QUALITY_THRESHOLDS)

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        if not self.enabled:
            self.error = "Input validation is disabled in the configuration."
            return
        if not self.stats_path.is_file():
            self.error = (
                f"Reference statistics not found: {self.stats_path}. "
                "Run training/build_reference_stats.py to enable the full domain check."
            )
            return
        try:
            with np.load(self.stats_path, allow_pickle=False) as data:
                self.feature_mean = data["feature_mean"].astype(np.float64)
                self.feature_precision = data["feature_precision"].astype(np.float64)
                self.feature_threshold = float(data["feature_threshold"])
                self.color_mean = data["color_mean"].astype(np.float64)
                self.color_precision = data["color_precision"].astype(np.float64)
                self.color_threshold = float(data["color_threshold"])
                self.reference_patches = int(data["n_reference"])
                if "quality_thresholds" in data:
                    stored = json.loads(str(data["quality_thresholds"]))
                    self.quality.update(
                        {key: float(value) for key, value in stored.items() if key in self.quality}
                    )
            self.error = None
        except Exception as exc:  # noqa: BLE001 - surfaced through /health and /model
            self.error = f"{type(exc).__name__}: {exc}"
            self.feature_mean = None
            self.feature_precision = None
            self.feature_threshold = None
            self.color_mean = None
            self.color_precision = None
            self.color_threshold = None

    @property
    def ready(self) -> bool:
        return (
            self.enabled
            and self.feature_mean is not None
            and self.feature_precision is not None
            and self.feature_threshold is not None
            and self.color_mean is not None
            and self.color_precision is not None
            and self.color_threshold is not None
        )

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "reference_patches": self.reference_patches,
            "feature_threshold": self.feature_threshold,
            "stain_threshold": self.color_threshold,
            "error": self.error,
        }

    # --------------------------------------------------------------- screening
    def evaluate(self, metrics: ImageMetrics, feature_vector: np.ndarray | None = None) -> ScreeningResult:
        """Check an image and return an accepted/rejected verdict with reasons."""
        if not self.enabled:
            return ScreeningResult(
                accepted=True,
                checks=[
                    {
                        "name": "input_validation",
                        "passed": True,
                        "value": 0.0,
                        "limit": 0.0,
                        "note": "Input validation is disabled in the configuration.",
                    }
                ],
            )

        checks: list[dict] = []
        reasons: list[str] = []
        details: dict[str, float] = {}

        def record(name: str, passed: bool, value: float, limit: float) -> None:
            checks.append(
                {"name": name, "passed": bool(passed), "value": round(float(value), 4), "limit": round(float(limit), 4)}
            )

        # ---------------------------------------------------------- acquisition
        bright = metrics.brightness_mean
        record("brightness", self.quality["min_brightness"] <= bright <= self.quality["max_brightness"], bright,
               self.quality["min_brightness"] if bright < self.quality["min_brightness"] else self.quality["max_brightness"])
        if bright < self.quality["min_brightness"]:
            reasons.append(f"The image is too dark to analyse (mean brightness {bright:.2f}).")
        elif bright > self.quality["max_brightness"]:
            reasons.append(f"The image is almost completely white (mean brightness {bright:.2f}).")

        record("dark_fraction", metrics.dark_fraction <= self.quality["max_dark_fraction"], metrics.dark_fraction,
               self.quality["max_dark_fraction"])
        if metrics.dark_fraction > self.quality["max_dark_fraction"]:
            reasons.append("Most of the image is black or heavily underexposed.")

        record(
            "clipped_bright_fraction",
            metrics.clipped_bright_fraction <= self.quality["max_clipped_bright_fraction"],
            metrics.clipped_bright_fraction,
            self.quality["max_clipped_bright_fraction"],
        )
        if metrics.clipped_bright_fraction > self.quality["max_clipped_bright_fraction"]:
            reasons.append("Most of the image is overexposed (clipped white pixels).")

        record("sharpness", metrics.sharpness >= self.quality["min_sharpness"], metrics.sharpness,
               self.quality["min_sharpness"])
        if metrics.sharpness < self.quality["min_sharpness"]:
            reasons.append(
                "The image is too blurred to analyse reliably. Upload a sharply focused tissue patch."
            )

        greyscale = (
            metrics.grayscale_fraction > self.quality["max_grayscale_fraction"]
            and metrics.colorfulness < self.quality["max_grayscale_colorfulness"]
        )
        record("greyscale", not greyscale, metrics.grayscale_fraction, self.quality["max_grayscale_fraction"])
        if greyscale:
            reasons.append(
                "The image is effectively greyscale; H&E-stained tissue shows pink and purple colour."
            )

        # ---------------------------------------------------- colour/stain domain
        if self.color_precision is not None and self.color_mean is not None and self.color_threshold is not None:
            stain_distance = _mahalanobis(metrics.color_features(), self.color_mean, self.color_precision)
            details["stain_distance"] = round(stain_distance, 2)
            details["stain_threshold"] = round(self.color_threshold, 2)
            record("stain_colour", stain_distance <= self.color_threshold, stain_distance, self.color_threshold)
            if stain_distance > self.color_threshold:
                reasons.append(
                    "The colour signature is unlike H&E-stained tissue "
                    f"(stain distance {stain_distance:.1f} vs. limit {self.color_threshold:.1f})."
                )

        # ------------------------------------------------------- deep domain OOD
        if (
            feature_vector is not None
            and self.feature_precision is not None
            and self.feature_mean is not None
            and self.feature_threshold is not None
        ):
            domain_distance = _mahalanobis(feature_vector, self.feature_mean, self.feature_precision)
            details["domain_distance"] = round(domain_distance, 2)
            details["domain_threshold"] = round(self.feature_threshold, 2)
            record("deep_domain", domain_distance <= self.feature_threshold, domain_distance, self.feature_threshold)
            if domain_distance > self.feature_threshold:
                reasons.append(
                    "The image's VGG19 feature signature lies outside the distribution of the "
                    f"{self.reference_patches:,} BreakHis training patches "
                    f"(domain distance {domain_distance:.1f} vs. limit {self.feature_threshold:.1f})."
                )

        if self.enabled and not self.ready and not reasons:
            # No calibrated references: acquisition checks only. Do not pretend
            # that a domain decision was made.
            checks.append(
                {
                    "name": "calibrated_domain_check",
                    "passed": True,
                    "value": 0.0,
                    "limit": 0.0,
                    "note": self.error or "Reference statistics unavailable.",
                }
            )

        return ScreeningResult(
            accepted=not reasons,
            reasons=reasons,
            checks=checks,
            details=details,
        )
