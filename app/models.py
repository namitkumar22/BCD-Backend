"""Database models."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_image: Mapped[str] = mapped_column(String(512))
    grad_cam_image: Mapped[str | None] = mapped_column(String(512), nullable=True)

    predicted_label: Mapped[str] = mapped_column(String(20), index=True)
    probability: Mapped[float] = mapped_column(Float)
    benign_probability: Mapped[float] = mapped_column(Float)
    malignant_probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20))

    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    model_mode: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
