# OPVQ (Objective Perceptual Video Quality)

## Overview

**OPVQ** is a full-reference perceptual video quality evaluation metric designed to model the human visual system's sensitivity to spatial and temporal video distortions.

In this study (*"How to Choose your Pre-owned Smartphone?: A Multi-Dimensional Benchmarking of Performance and Quality"*), OPVQ is evaluated alongside **VMAF** to assess perceptual video degradation in smartphone screen recordings. Evaluating video fidelity across multiple complementary metrics ensures that observed intra-model performance degradation trends are robust and not biased by a single perceptual model.

---

## Directory Layout

```
OPVQ/
├── opvq_mos.py                  # Core OPVQ & VMAF evaluation module + Section 3.11 statistical framework
├── test_compression.py          # Benchmark runner against synthetic compression artifacts
├── reference.mp4                # Pristine reference screen recording
├── distorted.mp4                # Captured / degraded screen recording
├── requirements.txt             # Python dependencies (numpy, matplotlib, pandas, scipy)
├── OPVQ_Installation_Guide.docx # Detailed system & environment setup document
└── README.md                    # Module documentation & methodology guide
```

---

## Methodology & Mathematical Formulation (from Paper)

### 1. Video Quality Metric Scales (Section 3.4 & 3.6)
- **VMAF**: $\text{VMAF} \in [0, 100]$ (higher scores indicate greater similarity to reference video).
- **OPVQ**: Reported on the standardized ITU 5-point perceptual quality scale $\text{OPVQ} \in [1.0, 5.0]$ (Figure 5 & Table 6), where higher scores indicate superior perceptual quality.
  $$\text{OPVQ}_{\text{score}} = 1.0 + 4.0 \times \left(1.0 - D_{\text{pooled}}\right)$$

### 2. Spatial & Temporal Distortion Modeling
- **Structural Similarity (SSIM)** on luminance frames:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- **Spatial Information (SI)** using Sobel gradient energy:
  $$\Delta \text{SI} = \frac{|\text{SI}_{\text{ref}} - \text{SI}_{\text{dist}}|}{\text{SI}_{\text{ref}} + \epsilon}$$
- **Temporal Information (TI)** motion continuity:
  $$\Delta \text{TI} = \frac{|\text{TI}_{\text{ref}} - \text{TI}_{\text{dist}}|}{\text{TI}_{\text{ref}} + \epsilon}$$
- **Minkowski Temporal Pooling ($p=2$)**:
  $$D_{\text{pooled}} = \left( \frac{1}{T} \sum_{t=1}^T D_t^2 \right)^{1/2}$$

### 3. Statistical Analysis Framework (Section 3.11 & Table 6)
To compare device performance across manufacturing years within the same lineage (e.g. 2016 vs 2017, 2017 vs 2018, 2016 vs 2018):
- **Average Difference $D(A, B)$**:
  $$D(A, B) = \frac{1}{N} \sum_{i=1}^N (x_i^A - x_i^B)$$
  Negative values indicate that older devices ($A$) exhibit lower perceptual quality than newer devices ($B$).
- **Win Count**:
  Measures consistency across evaluated video observations; counts the number of times newer devices achieved superior quality under identical conditions.
- **Statistical Significance**:
  Paired $t$-tests evaluating significance at $p < 0.01$.

---

## System Requirements & Installation

*(As specified in `OPVQ_Installation_Guide.docx`)*

- **Operating System:** Ubuntu 20.04 / 22.04 / 24.04 or WSL (Windows Subsystem for Linux Ubuntu)
- **Python:** Version 3.8 or higher
- **FFmpeg:** Compiled with `libvmaf` support

### Step 1: Update System Packages
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Build Dependencies
```bash
sudo apt install -y git pkg-config build-essential cmake nasm yasm python3 python3-pip python3-venv
```

### Step 3: Install libvmaf
```bash
sudo apt install -y libvmaf-dev libvmaf-tools
vmaf --version
```

### Step 4: Install FFmpeg with libvmaf
```bash
sudo apt install -y ffmpeg
ffmpeg -filters | grep vmaf
```

### Step 5: Configure Python Environment
```bash
python3 -m venv opvq_env
source opvq_env/bin/activate
pip install --upgrade pip
```

### Step 6: Install Required Python Packages
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Run OPVQ & VMAF Evaluation (Single Video Pair)
Runs evaluation comparing `reference.mp4` and `distorted.mp4`:
```bash
python opvq_mos.py
```
*Outputs OPVQ Score (1–5), OPVQ Index (0–100), VMAF Score, and Metric Agreement ($\Delta\text{MOS}$).*

### 2. Evaluate Custom Video Pairs
```bash
python opvq_mos.py /path/to/reference.mp4 /path/to/distorted.mp4
```

### 3. Display Section 4.2 Table 6 Benchmark (Pairwise Comparisons)
```bash
python opvq_mos.py --table6
```

### 4. Compute Table 6 Statistics from Dynamic Evaluation Scores
```bash
python opvq_mos.py --stats evaluated_video_scores.json --output_csv opvq_table6_results.csv
```

### 5. Run Synthetic Compression Benchmark
Generates an H.264 compressed test artifact and benchmarks degradation:
```bash
python test_compression.py
```

---

## Academic Note

> **Note (Section 3.4):** Scores reflect perceptual video quality on standardized scales (OPVQ $\in [1.0, 5.0]$, VMAF $\in [0, 100]$).
