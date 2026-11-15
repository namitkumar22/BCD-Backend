"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.config import get_settings
from app.database import init_db
from app.services.inference import ModelService

settings = get_settings()
PACKAGE_DIR = Path(__file__).resolve().parent
if not settings.model_path.is_file():
    # Also accept the checkpoint in app/models for simpler deployments.
    alternative = PACKAGE_DIR / "models" / "vgg19_breakhis.pt"
    if alternative.is_file():
        settings.model_path = alternative


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    service = ModelService(settings)
    service.load()
    routes.set_model_service(service)
    app.state.model_service = service
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "REST API for breast cancer histopathology screening with a VGG19 "
        "transfer-learning model. Upload a tissue patch to obtain a "
        "benign/malignant probability, a Grad-CAM explanation and a PDF report.\n\n"
        "**Research and educational use only - not a medical device.**"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")


@app.get("/", tags=["system"])
def index() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }
