"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", env_prefix="BCD_", extra="ignore")

    app_name: str = "Breast Cancer Detection API"
    app_version: str = "1.0.0"
    environment: str = "development"

    model_path: Path = BACKEND_ROOT / "models" / "vgg19_breakhis.pt"
    metrics_path: Path = BACKEND_ROOT / "models" / "metrics.json"
    reference_stats_path: Path = BACKEND_ROOT / "models" / "reference_stats.npz"
    torch_home: Path = BACKEND_ROOT / "models" / "torch_cache"

    # Verify that an upload is histopathology before running the classifier.
    input_validation: bool = True

    storage_dir: Path = BACKEND_ROOT / "storage"
    database_url: str = f"sqlite:///{(BACKEND_ROOT / 'storage' / 'app.db').as_posix()}"

    max_upload_mb: int = 10
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    generate_gradcam: bool = True
    torch_threads: int = 4

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "reports").mkdir(parents=True, exist_ok=True)
    return settings
