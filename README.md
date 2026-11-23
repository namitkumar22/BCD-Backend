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
│   │   ├── gradcam.py          Grad-CAM implementation
│   │   └── input_quality.py    Shared 224 crop, colour/focus metrics
│   └── services/
│       ├── inference.py        Verification, prediction, risk levels
│       ├── input_gate.py       Out-of-domain (non-histopathology) rejection
│       ├── storage.py          Upload validation and file storage
│       └── report.py           PDF report generation
├── training/
│   ├── prepare_dataset.py      Patient-level BreakHis split manifest
│   ├── dataset.py              Dataset + augmentation pipeline
│   ├── train.py                Frozen / fine-tune training entry point
│   ├── build_reference_stats.py  Calibrates the input-domain gate
│   └── colab_train.ipynb       Ready-to-run Google Colab notebook
├── models/                     Checkpoint, metrics, reference_stats.*, figures/
├── tests/                      Input-gate regression tests
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
| `GET` | `/api/health` | Service, model and input-gate readiness |
| `GET` | `/api/model` | Architecture, threshold and test metrics |
| `POST` | `/api/predict` | Multipart upload; returns prediction, probabilities and Grad-CAM URL. Non-histopathology images return HTTP 422 with the rejection reasons and are not stored. |
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
| `BCD_REFERENCE_STATS_PATH` | `models/reference_stats.npz` | Input-domain reference statistics |
| `BCD_INPUT_VALIDATION` | `true` | Reject images outside the histopathology domain |
| `BCD_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `BCD_GENERATE_GRADCAM` | `true` | Global Grad-CAM switch |
| `BCD_TORCH_THREADS` | `4` | CPU inference threads |
| `BCD_ALLOWED_ORIGINS` | `localhost:3000` | CORS origins |
| `BCD_DATABASE_URL` | `sqlite:///storage/app.db` | Database URL |

## Input validation (out-of-domain rejection)

A VGG19 classifier trained on BreakHis still returns a confident-looking
probability for a face, a document or a screenshot. The API therefore verifies
the upload before inference:

1. **Acquisition checks** - decode, minimum size, exposure, focus and greyscale.
2. **Stain-colour distance** - Mahalanobis distance of an 8-dimensional HSV /
   colourfulness / greyness descriptor from the training patches.
3. **Deep domain distance** - Mahalanobis distance of the 512-dimensional VGG19
   feature vector from the training patches.

Both distances use means and Ledoit-Wolf precision matrices estimated from the
1,451 training patches. Rejection thresholds are the 99.9th percentile of the
validation split, so the gate is calibrated on unseen data and not on the test
set. With the shipped checkpoint the thresholds are 50.33 (VGG19 domain) and
5.69 (stain colour). On the held-out test split each distance rejects 0.4% of
genuine patches, while a face (domain distance 318.9), a photograph (244.4), a
logo (174.1), a document (122.0) and noise (stain 9.7) are rejected. A rejected
upload returns `422` with a machine-readable body and never reaches the
classifier, the database or the report generator:

```json
{
  "detail": {
    "code": "invalid_input",
    "message": "The image was rejected before analysis ...",
    "reasons": ["The image's VGG19 feature signature lies outside ..."],
    "guidance": "Upload a single H&E-stained breast histopathology patch ...",
    "details": { "domain_distance": 41.2, "domain_threshold": 27.5, "stain_distance": 18.4, "stain_threshold": 9.1 }
  }
}
```

Rebuild the reference statistics after changing the checkpoint or backbone:

```bash
uv run python training/build_reference_stats.py
```

This also writes `models/reference_stats.json`, which records distance
percentiles per split, false-rejection rates and scores for probe images
(a face, a photograph, a logo, a document and noise). Set
`BCD_INPUT_VALIDATION=false` to disable the gate; `/api/health` and `/api/model`
report whether it is active.

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
`test_predictions.csv` and evaluation figures to `models/figures/`. Finish a
training run with `uv run python training/build_reference_stats.py` so the input
validation matches the new checkpoint.

## Tests

```bash
uv run pytest tests -q
```

The tests confirm that real BreakHis patches are accepted, that a face photo and
a blank image are rejected before inference, and that the predicted label
follows the tuned decision threshold.

See `training/colab_train.ipynb` for a Colab-ready workflow and
`training/README.md` for details.

## Notes on the split

BreakHis has multiple patches per patient. A random image-level split leaks
patient information into the test set and inflates accuracy, so this project
splits **by patient** and keeps the test set untouched until the final
evaluation. The decision threshold is selected on the validation split with
Youden's J statistic.
