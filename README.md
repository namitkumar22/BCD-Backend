# Breast Cancer Detection using VGG19 — Backend

FastAPI service that classifies breast histopathology patches as **benign** or
**malignant** with a VGG19 transfer-learning model, returns Grad-CAM attention
maps, stores a screening history in SQLite and generates PDF reports.

> Research and educational use only. This is not a medical device.

## Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI 0.115 + Uvicorn |
| Model | PyTorch 2.x, torchvision VGG19 (ImageNet weights) |
| Explainability | Grad-CAM on the last convolutional block |
| Database | SQLite + SQLAlchemy 2.0 |
| Reports | ReportLab |
| Environment | `uv` virtual environment, Python 3.12 |

## Project layout

```
Backend/
├── app/
│   ├── main.py                 FastAPI app, CORS, lifespan model loading
│   ├── config.py               Settings (BCD_* environment variables)
│   ├── database.py             SQLAlchemy engine/session
│   ├── models.py               Prediction table
│   ├── schemas.py              Pydantic response models
│   ├── api/routes.py           All HTTP endpoints
│   ├── core/
│   │   ├── architecture.py     VGG19Classifier + checkpoint loading
│   │   └── gradcam.py          Grad-CAM implementation
│   └── services/
│       ├── inference.py        Preprocessing, prediction, risk levels
│       ├── storage.py          Upload validation and file storage
│       └── report.py           PDF report generation
├── training/
│   ├── prepare_dataset.py      Patient-level BreakHis split manifest
│   ├── dataset.py              Dataset + augmentation pipeline
│   ├── train.py                Frozen / fine-tune training entry point
│   └── colab_train.ipynb       Ready-to-run Google Colab notebook
├── models/                     vgg19_breakhis.pt, metrics.json, figures/
├── storage/                    Uploaded patches, Grad-CAM images, SQLite DB
├── data/                       BreakHis dataset + split manifest (not committed)
└── pyproject.toml
```

## Quick start

```bash
cd Backend
uv sync
uv run uvicorn app.main:app --port 8000
```

One command after cloning:

```bash
cd Backend && uv sync && uv run uvicorn app.main:app --port 8000
```

* Swagger UI: <http://localhost:8000/docs>
* Health check: <http://localhost:8000/api/health>
* Model card JSON: <http://localhost:8000/api/model>

The first startup downloads the ImageNet VGG19 backbone (~548 MB) into
`models/torch_cache/` unless it is already present. A demo CPU prediction takes
roughly 1–2 seconds including Grad-CAM.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service and model readiness |
| `GET` | `/api/model` | Architecture, threshold and test metrics |
| `POST` | `/api/predict` | Multipart upload; returns prediction, probabilities and Grad-CAM URL |
| `GET` | `/api/predictions` | Paginated history (`limit`, `offset`, `label`) |
| `GET` | `/api/predictions/{id}` | Single prediction |
| `GET` | `/api/predictions/{id}/image` | Original uploaded patch |
| `GET` | `/api/predictions/{id}/gradcam` | Grad-CAM overlay |
| `GET` | `/api/predictions/{id}/report` | One-page PDF report |
| `DELETE` | `/api/predictions/{id}` | Delete record and stored images |
| `GET` | `/api/stats` | Dashboard counters |

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@sample.jpg" \
  -F "generate_gradcam=true"
```

```json
{
  "id": 1,
  "predicted_label": "malignant",
  "probabilities": { "benign": 0.07, "malignant": 0.93 },
  "risk_level": "high",
  "latency_ms": 1840,
  "grad_cam_url": "/api/predictions/1/gradcam"
}
```

## Configuration

Copy `.env.example` to `.env` and adjust. Every key is prefixed with `BCD_`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BCD_MODEL_PATH` | `models/vgg19_breakhis.pt` | Trained checkpoint |
| `BCD_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `BCD_GENERATE_GRADCAM` | `true` | Global Grad-CAM switch |
| `BCD_TORCH_THREADS` | `4` | CPU inference threads |
| `BCD_ALLOWED_ORIGINS` | `localhost:3000` | CORS origins |
| `BCD_DATABASE_URL` | `sqlite:///storage/app.db` | Database URL |

## Training the model

The dataset is **BreakHis** (breast histopathology, 40X magnification). Put the
extracted dataset in `data/breakhis/`, then:

```bash
# Patient-level 70/15/15 split (82 patients, 1995 patches)
uv run python training/prepare_dataset.py --data-dir data/breakhis

# Fast CPU training: frozen VGG19 backbone + classifier head
uv run python training/train.py --mode frozen

# GPU / Colab: fine-tune the last two VGG19 blocks
uv run python training/train.py --mode finetune --epochs 15
```

Both modes write `models/vgg19_breakhis.pt`, `metrics.json`, `history.json`,
`test_predictions.csv` and evaluation figures to `models/figures/`.

See `training/colab_train.ipynb` for a Colab-ready workflow and
`training/README.md` for details.

## Notes on the split

BreakHis has multiple patches per patient. A random image-level split leaks
patient information into the test set and inflates accuracy, so this project
splits **by patient** and keeps the test set untouched until the final
evaluation. The decision threshold is selected on the validation split with
Youden's J statistic.
