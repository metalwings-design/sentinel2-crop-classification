# Crop Classification Using Multi-Temporal Sentinel-2 Imagery

## Project Overview

This project classifies crop types using multi-temporal Sentinel-2 satellite imagery from the PASTIS dataset. The workflow extracts temporal features (mean, std, max, min) from 46 time steps, trains a LightGBM model, and applies spatial smoothing for improved results.

**Key Results:** 60.72% validation accuracy using 30 patches.

---

## Dataset Structure

Place the dataset in the following structure:

PASTIS_subset/
├── DATA_S2/
│ └── S2_<patch_id>.npy # Sentinel-2 images (46, 10, 128, 128)
├── ANNOTATIONS/
│ └── TARGET_<patch_id>.npy # Crop labels (1, 128, 128)
└── metadata.geojson # Acquisition dates



---

## Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2 install dependancies

pip install -r requirements.txt

### 3 Update configuration
Edit configs/config.yaml with your data path:
data:
  data_dir: "C:/path/to/PASTIS_subset"  

### 4 Run notebook

jupyter notebook notebooks/exploration_and_training.ipynb

## Repository structure
sentinel2-crop-classification/
├── README.md
├── report.md
├── requirements.txt
├── notebooks/
│   └── exploration_and_training.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── test_patch_comparison.png
│   ├── metrics/
│   │   └── classification_report.txt
│   └── predictions/
│       ├── test_patch_predictions.npy
│       └── test_patch_smoothed.npy
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── configs/
    └── config.yaml


## Key files

notebooks/exploration_and_training.ipynb	Main workflow (load, explore, train, evaluate)
src/preprocessing.py	                    Feature extraction and dataset building
src/train.py	                            LightGBM training
configs/config.yaml	                        All configurable parameters
report.md	                                Full analysis and results