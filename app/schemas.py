"""Pydantic response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    version: str
    device: str


class ModelMetrics(BaseModel):
    accuracy: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    precision: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None
    n_samples: int | None = None


class ModelInfo(BaseModel):
    name: str
    architecture: str
    input_size: int
    classes: list[str]
    mode: str
    threshold: float
    parameters: int
    trained_at: str | None = None
    dataset: str
    metrics: ModelMetrics
    total_predictions: int


class Probabilities(BaseModel):
    benign: float
    malignant: float


class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_label: Literal["benign", "malignant"]
    display_label: str
    probability: float = Field(description="Probability of the predicted class")
    probabilities: Probabilities
    threshold: float
    confidence: float
    risk_level: Literal["low", "moderate", "high"]
    latency_ms: int
    image_width: int
    image_height: int
    model_mode: str
    created_at: datetime
    image_url: str
    grad_cam_url: str | None = None
    disclaimer: str


class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    predicted_label: str
    display_label: str
    probability: float
    malignant_probability: float
    risk_level: str
    latency_ms: int
    created_at: datetime
    image_url: str
    grad_cam_url: str | None = None


class PredictionHistory(BaseModel):
    items: list[PredictionHistoryItem]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    total: int
    benign: int
    malignant: int
    high_risk: int
    average_malignant_probability: float
    average_latency_ms: int
