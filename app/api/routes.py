"""HTTP routes exposed by the backend."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Prediction
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelMetrics,
    PredictionHistory,
    PredictionHistoryItem,
    PredictionResponse,
    Probabilities,
    StatsResponse,
)
from app.services.inference import DISCLAIMER, ModelNotReadyError, ModelService
from app.services.report import build_report
from app.services.storage import InvalidImageError, load_upload, save_gradcam, save_upload

router = APIRouter()
settings = get_settings()

_model_service: ModelService | None = None


def set_model_service(service: ModelService) -> None:
    global _model_service
    _model_service = service


def get_model_service() -> ModelService:
    if _model_service is None:  # pragma: no cover - guarded by application lifespan
        raise HTTPException(status_code=503, detail="Model service is not initialised.")
    return _model_service


def _image_url(prediction_id: int, kind: str) -> str:
    return f"/api/predictions/{prediction_id}/{kind}"


def _to_response(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        id=prediction.id,
        filename=prediction.filename,
        predicted_label=prediction.predicted_label,  # type: ignore[arg-type]
        display_label=prediction.predicted_label.capitalize(),
        probability=prediction.probability,
        probabilities=Probabilities(
            benign=prediction.benign_probability,
            malignant=prediction.malignant_probability,
        ),
        threshold=prediction.threshold,
        confidence=prediction.probability,
        risk_level=prediction.risk_level,  # type: ignore[arg-type]
        latency_ms=prediction.latency_ms,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        model_mode=prediction.model_mode,
        created_at=prediction.created_at,
        image_url=_image_url(prediction.id, "image"),
        grad_cam_url=_image_url(prediction.id, "gradcam") if prediction.grad_cam_image else None,
        disclaimer=DISCLAIMER,
    )


def _to_history_item(prediction: Prediction) -> PredictionHistoryItem:
    return PredictionHistoryItem(
        id=prediction.id,
        filename=prediction.filename,
        predicted_label=prediction.predicted_label,
        display_label=prediction.predicted_label.capitalize(),
        probability=prediction.probability,
        malignant_probability=prediction.malignant_probability,
        risk_level=prediction.risk_level,
        latency_ms=prediction.latency_ms,
        created_at=prediction.created_at,
        image_url=_image_url(prediction.id, "image"),
        grad_cam_url=_image_url(prediction.id, "gradcam") if prediction.grad_cam_image else None,
    )


# --------------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: Annotated[ModelService, Depends(get_model_service)]) -> HealthResponse:
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        model_ready=service.ready,
        version=settings.app_version,
        device=str(service.device),
    )


@router.get("/model", response_model=ModelInfo, tags=["system"])
def model_info(
    service: Annotated[ModelService, Depends(get_model_service)],
    db: Annotated[Session, Depends(get_db)],
) -> ModelInfo:
    metrics = service.metrics or service.load_metrics_file()
    total = db.scalar(select(func.count(Prediction.id))) or 0
    return ModelInfo(
        name="VGG19 Breast Cancer Classifier",
        architecture="VGG19 (ImageNet transfer learning) + 2-layer classifier head",
        input_size=int(service.metadata.get("input_size", 224)),
        classes=list(service.metadata.get("class_names", ["benign", "malignant"])),
        mode=str(service.metadata.get("mode", "frozen")),
        threshold=service.threshold,
        parameters=service.parameters,
        trained_at=service.metadata.get("trained_at"),
        dataset="BreakHis 40X (patient-level split)",
        metrics=ModelMetrics(
            accuracy=metrics.get("accuracy"),
            sensitivity=metrics.get("sensitivity"),
            specificity=metrics.get("specificity"),
            precision=metrics.get("precision"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
            n_samples=metrics.get("n_samples"),
        ),
        total_predictions=total,
    )


# -------------------------------------------------------------------- predict
@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    service: Annotated[ModelService, Depends(get_model_service)],
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(description="Histopathology image (JPG, PNG, WEBP)")],
    generate_gradcam: Annotated[bool, Form()] = True,
    notes: Annotated[str | None, Form()] = None,
) -> PredictionResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail=f"Model is not available: {service.error}")

    payload = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        image, suffix = load_upload(payload, file.filename or "image.jpg", settings)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = service.predict(image, want_grad_cam=generate_gradcam and settings.generate_gradcam)
    except ModelNotReadyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    stored_image = save_upload(image, suffix, settings)
    grad_cam_name = None
    if result["overlay"] is not None:
        buffer = io.BytesIO()
        Image.fromarray(result["overlay"]).save(buffer, format="PNG")
        grad_cam_name = save_gradcam(buffer.getvalue(), settings)

    record = Prediction(
        filename=file.filename or "image.jpg",
        stored_image=str(settings.storage_dir / "uploads" / stored_image),
        grad_cam_image=str(settings.storage_dir / "uploads" / grad_cam_name) if grad_cam_name else None,
        predicted_label=result["predicted_label"],
        probability=result["probability"],
        benign_probability=result["probabilities"]["benign"],
        malignant_probability=result["probabilities"]["malignant"],
        threshold=result["threshold"],
        risk_level=result["risk_level"],
        image_width=image.width,
        image_height=image.height,
        model_mode=str(service.metadata.get("mode", "frozen")),
        latency_ms=result["latency_ms"],
        notes=(notes or "").strip() or None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_response(record)


# -------------------------------------------------------------------- history
@router.get("/predictions", response_model=PredictionHistory, tags=["history"])
def list_predictions(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    label: Annotated[str | None, Query(pattern="^(benign|malignant)$")] = None,
) -> PredictionHistory:
    query = select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc())
    count_query = select(func.count(Prediction.id))
    if label:
        query = query.where(Prediction.predicted_label == label)
        count_query = count_query.where(Prediction.predicted_label == label)
    total = db.scalar(count_query) or 0
    rows = db.scalars(query.offset(offset).limit(limit)).all()
    return PredictionHistory(
        items=[_to_history_item(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse, tags=["history"])
def get_prediction(prediction_id: int, db: Annotated[Session, Depends(get_db)]) -> PredictionResponse:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return _to_response(record)


@router.delete("/predictions/{prediction_id}", status_code=204, tags=["history"])
def delete_prediction(prediction_id: int, db: Annotated[Session, Depends(get_db)]) -> Response:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    for stored in (record.stored_image, record.grad_cam_image):
        if stored:
            path = Path(stored)
            if path.is_file():
                path.unlink(missing_ok=True)
    db.delete(record)
    db.commit()
    return Response(status_code=204)


def _file_response(path: Path, prediction_id: int, kind: str) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    suffix = path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=f"prediction-{prediction_id}-{kind}{suffix}")


@router.get("/predictions/{prediction_id}/image", tags=["history"])
def get_image(prediction_id: int, db: Annotated[Session, Depends(get_db)]) -> FileResponse:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return _file_response(Path(record.stored_image), prediction_id, "image")


@router.get("/predictions/{prediction_id}/gradcam", tags=["history"])
def get_gradcam(prediction_id: int, db: Annotated[Session, Depends(get_db)]) -> FileResponse:
    record = db.get(Prediction, prediction_id)
    if record is None or not record.grad_cam_image:
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return _file_response(Path(record.grad_cam_image), prediction_id, "gradcam")


@router.get("/predictions/{prediction_id}/report", tags=["history"])
def download_report(prediction_id: int, db: Annotated[Session, Depends(get_db)]) -> Response:
    record = db.get(Prediction, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    pdf = build_report(record, settings)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="BCD-report-{prediction_id:05d}.pdf"'},
    )


# ---------------------------------------------------------------------- stats
@router.get("/stats", response_model=StatsResponse, tags=["history"])
def stats(db: Annotated[Session, Depends(get_db)]) -> StatsResponse:
    total = db.scalar(select(func.count(Prediction.id))) or 0
    malignant = db.scalar(select(func.count(Prediction.id)).where(Prediction.predicted_label == "malignant")) or 0
    high_risk = db.scalar(select(func.count(Prediction.id)).where(Prediction.risk_level == "high")) or 0
    average_probability = db.scalar(select(func.avg(Prediction.malignant_probability))) or 0.0
    average_latency = db.scalar(select(func.avg(Prediction.latency_ms))) or 0.0
    return StatsResponse(
        total=total,
        benign=total - malignant,
        malignant=malignant,
        high_risk=high_risk,
        average_malignant_probability=round(float(average_probability), 4),
        average_latency_ms=int(average_latency),
    )
