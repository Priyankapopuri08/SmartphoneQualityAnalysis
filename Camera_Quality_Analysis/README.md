# Camera Quality Analysis

This directory contains the complete source code and experimental framework for **Camera Quality Analysis**, matching the methodology, formulas, and results presented in:

> **"How to Choose your Pre-owned Smartphone?": A Multi-Dimensional Benchmarking of Performance and Quality**  
> *Priyanka Chowdary Popuri, Dipanjan Chakraborty*  
> *Department of Computer Science and Information Systems, BITS Pilani, Hyderabad Campus*

---

## Overview

Smartphone camera evaluation in this study assesses intra-model performance degradation in aging Android smartphones across three standardized capture scenarios (Section 3.8):
1. **Daylight Outdoor**
2. **Indoor Lighting**
3. **Low-Light Conditions**

It combines **No-Reference Image Quality Assessment (NR-IQA)** with **Subjective Mean Rank (MR)**, **Non-Linear Logistic Mapping $Q(x)$**, and **Correlation Analysis (SRCC, PLCC, RMSE)**.

---

## Directory & Branch Layout

Following the structure of other evaluation modules in this repository (such as `OPVQ`), Camera Quality Analysis is organized in one directory with dedicated modular branches for each camera quality metric:

```
Camera_Quality_Analysis/
├── camera_quality_pipeline.py            # Master Unified Pipeline (replicates Tables 8, 9, 10)
├── brisque.py                            # BRISQUE core calculation module
├── niqe.py                               # NIQE core calculation module
├── il_niqe.py                            # IL-NIQE core calculation module
├── subjective_mr.py                      # Subjective Mean Rank (MR) module (Table 8)
├── logistic_mapping.py                   # 5-parameter logistic mapping Q(x) (Table 9)
├── correlation_analysis.py               # SRCC, PLCC, and RMSE module (Table 10)
│
├── branches/                             # Dedicated metric branch directories
│   ├── BRISQUE/                          # Branch: BRISQUE implementation & runner
│   │   ├── run_brisque.py
│   │   └── README.md
│   ├── NIQE/                             # Branch: NIQE implementation & runner
│   │   ├── run_niqe.py
│   │   └── README.md
│   └── IL_NIQE/                          # Branch: IL-NIQE implementation & runner
│       ├── run_il_niqe.py
│       └── README.md
│
├── sample_images/                        # Test captures across 9 phones (27 unique images: daylight, indoor, lowlight)
│   ├── Oppo_A37_{2016,2017,2018}_{daylight,indoor,lowlight}.jpg
│   ├── Vivo_Y67_{2016,2017,2018}_{daylight,indoor,lowlight}.jpg
│   └── Redmi_5A_{2016,2017,2018}_{daylight,indoor,lowlight}.jpg
│
├── Camera_Quality_Installation_Guide.docx # Formatted Word Installation Guide (matches OPVQ guide)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Mathematical Formulas Implemented (from Paper)

### 1. Subjective Mean Rank (MR) (Section 3.4 & Table 8)
$$MR_j = \frac{1}{N} \sum_{i=1}^N r_{ij}$$
Where $r_{ij}$ is the rank assigned by participant $i$ to image $j$, and $N$ is total evaluators ($N=5$).  
*Lower MR values indicate superior perceived image quality.*

### 2. No-Reference Image Quality Assessment (NR-IQA)
- **BRISQUE**: Extracts 36 Natural Scene Statistics (NSS) features from MSCN coefficients and directional pairwise AGGD products across 2 scales.
- **NIQE**: Measures Mahalanobis-like statistical distance between the test image MVG model and a natural pristine image model:
  $$D(\nu_1, \nu_2, \Sigma_1, \Sigma_2) = \sqrt{(\nu_1 - \nu_2)^T \left( \frac{\Sigma_1 + \Sigma_2}{2} \right)^{-1} (\nu_1 - \nu_2)}$$
- **IL-NIQE**: Integrates luminance MSCN, chrominance statistics ($C_b, C_r$), and gradient edge statistics.

### 3. 5-Parameter Logistic Mapping $Q(x)$ (Section 3.4 & Table 9)
$$Q(x) = \beta_1 \left( \frac{1}{2} - \frac{1}{1 + e^{\beta_2(x - \beta_3)}} \right) + \beta_4 x + \beta_5$$
Maps raw objective scores $x$ to perceptual quality scale $Q(x) \in [1, 5]$ (higher denotes better quality).

### 4. Correlation Analysis (Section 3.4 & Table 10)
- **Spearman Rank-Order Correlation Coefficient (SRCC)**:
  $$SRCC = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
- **Pearson Linear Correlation Coefficient (PLCC)**:
  $$PLCC = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Root Mean Squared Error (RMSE)**:
  $$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - y_i)^2}$$

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Unified Paper Replication Benchmark (Tables 8, 9, and 10)
```bash
python3 camera_quality_pipeline.py
```

### 3. Evaluate Any Single Image File
```bash
python3 camera_quality_pipeline.py --image sample_images/daylight_outdoor_2018.jpg
```

### 4. Run an Individual Branch
```bash
cd branches/BRISQUE && python3 run_brisque.py ../../sample_images/daylight_outdoor_2018.jpg
cd branches/NIQE && python3 run_niqe.py ../../sample_images/indoor_lighting_2018.jpg
cd branches/IL_NIQE && python3 run_il_niqe.py ../../sample_images/low_light_2018.jpg
```
