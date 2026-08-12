# Crop Type Classification: Technical Analysis and Evaluation Report

## 1. Model Selection Rationale
LightGBM (`LGBMClassifier`) was selected as the baseline model architecture for the following reasons:
* **Computational Efficiency:** Enables fast training and evaluation across multi-temporal raster datasets without requiring dedicated GPU acceleration.
* **Tabular Feature Handling:** Efficiently handles tabular feature representations derived from temporal summary statistics (mean, std, min, max) calculated across Sentinel-2 band time series.
* **Non-Linear Decision Boundaries:** Tree-based gradient boosting captures complex non-linear spectral interactions across multi-spectral bands.

---

## 2. Approach Strengths and Limitations

### Strengths
* **Low Computational Overhead:** Fast training times allow rapid iteration and execution on standard multi-core CPU hardware.
* **Explicit Feature Compression:** Reduces memory requirements by collapsing 4D spatial-temporal tensors (46, 10, 128, 128) into a compressed 2D feature matrix (N_{\text{pixels}}, 40).
* **Background Noise Exclusion:** Pre-filtering class `0` (Background) and class `19` (Void label) prevents non-crop pixels (38.91% of total spatial pixels) from dominating optimization loss gradients.

### Limitations
* **Loss of Spatial Context:** Pixel-wise tabular classification treats adjacent pixels independently, leading to high-frequency "salt-and-pepper" noise in output spatial predictions.
* **Phenological Aggregation:** Collapsing the 46-timestamp sequence into static summary statistics removes time-series sequential patterns essential for separating spectrally similar crop calendars.
* **Subsampling Bias:** Sampling 1,000 pixels per patch reduces the representation of rare minority classes in training partitions.

---

## 3. Evaluation Results and Interpretation

### Performance Overview
* **Validation Accuracy:** 0.5993
* **Test Overall Accuracy:** 0.5679
* **Test Mean IoU (mIoU):** 0.1744


### Class-Wise Metric Breakdown

| Class ID | Class Name | Precision | Recall | F1-Score | IoU | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Meadow | 0.7397 | 0.9057 | 0.8143 | 0.6868 | 43,483 |
| **2** | Soft winter wheat | 0.5426 | 0.5246 | 0.5335 | 0.3638 | 31,692 |
| **3** | Corn | 0.3117 | 0.4630 | 0.3726 | 0.2289 | 23,374 |
| **4** | Winter barley | 0.4712 | 0.0798 | 0.1365 | 0.0732 | 12,634 |
| **5** | Winter rapeseed | 0.8424 | 0.7905 | 0.8156 | 0.6886 | 10,537 |
| **6** | Spring barley | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1,345 |
| **7** | Sunflower | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2,618 |
| **10** | Winter triticale | 0.1723 | 0.0355 | 0.0589 | 0.0000 | 4,419 |
| **12** | Fruits/veg/flowers | 0.0000 | 0.0000 | 0.0000 | 0.0304 | 174 |
| **14** | Leguminous fodder | 0.0465 | 0.0067 | 0.0118 | 0.0000 | 1,486 |
| **15** | Soybeans | 0.4999 | 0.5275 | 0.5133 | 0.0059 | 18,106 |
| **17** | Mixed cereal | 0.1915 | 0.0211 | 0.0379 | 0.3453 | 855 |
| **18** | Sorghum | 0.0000 | 0.0000 | 0.0000 | 0.0193 | 558 |

---

## 4. Performance Drivers and Misclassification Analysis

### High-Performing Classes
* **Meadow (Class 1) & Winter Rapeseed (Class 5):** Achieved highest F1-scores (0.8143 and 0.8156). Meadow benefits from high pixel availability (Support: 43,483). Winter rapeseed exhibits unique spectral signatures during its flowering phase (high Red-Edge/NIR response).

### Poor-Performing / Unclassified Classes
* **Spring Barley (6), Sunflower (7), Sorghum (18):** Yielded 0.0000 F1-scores. These minority categories represent <1 % of dataset pixels, causing the model decision boundaries to default to majority classes.

### Inter-Class Confusion Drivers
* **Cereal Crop Overlap:** Soft winter wheat (Class 2), Winter barley (Class 4), and Winter triticale (Class 10) share similar canopy structures and phenological cycles. Aggregating temporal features into summary statistics removes the timing details needed to distinguish these species.

---

## 5. Dataset and Operational Artifacts

* **Class Imbalance & Sample Size:** Severe class imbalance skews loss minimization toward majority classes (Meadow, Wheat, Corn). Random pixel sampling further reduces minority class representation.
* **Spatial Resolution & Parcel Boundaries:** At 10m spatial resolution, boundary pixels along field margins contain mixed spectral signals (e.g., crop mixed with soil or road), leading to misclassifications along field borders.
* **Cloud Contamination & Seasonal Timing:** Missing acquisitions or cloud cover during critical growth phases (e.g., green-up or senescence) impair the temporal summary statistics.


---

## 6. Recommendations for Model and Pipeline Improvement

1. **Feature Engineering:** Calculate explicit multi-temporal vegetation indices (NDVI, NDWI, NDRE) across all 46 timestamps to preserve phenological curves.
2. **Stratified Sampling:** Replace random pixel sampling with class-stratified sampling to ensure rare crops are sufficiently represented in training sets.
3. **Spatial-Temporal Deep Learning:** Transition from pixel-level tabular models to deep learning segmentation architectures (e.g. U-Net) to capture spatial field context and temporal dependencies simultaneously.
