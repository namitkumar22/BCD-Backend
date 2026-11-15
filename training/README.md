# Model training

The pipeline uses the **BreakHis** breast histopathology dataset at **40X**
magnification: 1,995 patches from 82 patients, labelled benign or malignant.

## 1. Get the data

Download `breakhis.zip` (224 MB) from
<https://huggingface.co/datasets/hirundo-io/BreaKHis-original> and extract it so
that the following paths exist:

```
Backend/data/breakhis/histology_slides/breast/benign/...
Backend/data/breakhis/histology_slides/breast/malignant/...
```

## 2. Build the patient-level split

```bash
cd Backend
.venv/Scripts/python training/prepare_dataset.py --data-dir data/breakhis
```

Output: `data/breakhis_40x_manifest.csv`

| Split | Images | Benign | Malignant |
| --- | --- | --- | --- |
| Train | 1,451 | 450 | 1,001 |
| Validation | 283 | 80 | 203 |
| Test | 261 | 95 | 166 |

Whole patients are assigned to a single split, so patches from the same slide
never appear on both sides of the evaluation.

## 3. Train

### Option A — frozen backbone (CPU, ~30 min for 3 augmented passes)

```bash
.venv/Scripts/python training/train.py --mode frozen
```

VGG19's convolutional weights stay fixed; only a 2-layer head is trained on
cached 512-dimensional features. This is the checkpoint shipped with the demo
and runs comfortably on a laptop.

### Option B — fine-tuning (GPU / Google Colab, recommended for best accuracy)

```bash
python training/train.py --mode finetune --epochs 15 --batch-size 32
```

The last two convolutional blocks (from `features[24]`) and the head are
fine-tuned with AdamW, cosine annealing, class weighting and mixed precision.
Use `training/colab_train.ipynb` for a ready-made Colab session.

### Useful flags

| Flag | Default | Description |
| --- | --- | --- |
| `--mode` | `frozen` | `frozen` or `finetune` |
| `--epochs` | 60 / 15 | Epochs for frozen / fine-tune |
| `--batch-size` | 256 / 32 | Feature or image batch size |
| `--lr` | 1e-3 / 3e-4 | Learning rate |
| `--augment-copies` | 3 | Augmented feature passes (frozen mode) |
| `--patience` | 15 | Early-stopping patience on validation AUC |
| `--device` | auto | `cpu` or `cuda` |
| `--out-dir` | `models` | Artifact directory |

## 4. Outputs

```
models/
├── vgg19_breakhis.pt      best checkpoint (head-only for frozen, full for finetune)
├── metrics.json           test metrics, threshold, run metadata
├── history.json           per-epoch loss / accuracy / AUC
├── test_predictions.csv   per-image probability
└── figures/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── class_distribution.png
    ├── sample_images.png
    └── gradcam_examples.png
```

## 5. Use the checkpoint in the API

Copy `vgg19_breakhis.pt` and `metrics.json` into `Backend/models/` and restart
the API. The checkpoint stores its own threshold, class names, mode and metrics,
and the API reads them automatically.

## Reproducibility

* Seed: 42 (`--seed`)
* Split seed: 42 (`prepare_dataset.py --seed`)
* Selection: best validation ROC-AUC
* Test set: evaluated once, after training
* Threshold: Youden's J on the validation split
