# OPVQ (Objective Perceptual Video Quality)

## Overview

**OPVQ** is a full-reference perceptual video quality evaluation metric designed to model the human visual system's sensitivity to spatial and temporal video distortions.

In this study (*"Measuring Intra-Model Performance Degradation in Used Smartphones"*), OPVQ is evaluated alongside **VMAF** to assess perceptual video degradation in smartphone screen recordings. Evaluating video fidelity across multiple complementary metrics ensures that observed intra-model performance degradation trends are robust and not biased by a single perceptual model.

---

## Directory Layout

```
OPVQ/
├── opvq_mos.py                  # Core OPVQ & VMAF evaluation module + MOS mapping
├── test_compression.py          # Benchmark runner against synthetic compression artifacts
├── reference.mp4                # Pristine reference screen recording
├── distorted.mp4                # Captured / degraded screen recording
├── requirements.txt             # Python dependencies (numpy, matplotlib, pandas)
├── OPVQ_Installation_Guide.docx # Detailed system & environment setup document
└── README.md                    # Module documentation & execution guide
```

---

## Metric Architecture & Mathematical Formulation

### 1. Spatial Fidelity Evaluation
Evaluates structural distortion and spatial edge preservation on the luminance plane:
- **Structural Similarity (SSIM)**:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- **Spatial Information (SI)** using Sobel gradient energy:
  $$\text{SI} = \text{std}\left(\sqrt{G_x^2 + G_y^2}\right)$$
  $$\Delta \text{SI} = \frac{|\text{SI}_{\text{ref}} - \text{SI}_{\text{dist}}|}{\text{SI}_{\text{ref}} + \epsilon}$$

### 2. Temporal Fidelity Evaluation
Evaluates motion continuity and temporal jitter across consecutive frames:
- **Temporal Information (TI)**:
  $$\text{TI} = \text{std}\left(F_t - F_{t-1}\right)$$
  $$\Delta \text{TI} = \frac{|\text{TI}_{\text{ref}} - \text{TI}_{\text{dist}}|}{\text{TI}_{\text{ref}} + \epsilon}$$

### 3. Minkowski Temporal Pooling
Distortion across frames is aggregated using Minkowski $p=2$ temporal pooling to reflect human perception of localized worst impairments:
$$D_{\text{pooled}} = \left( \frac{1}{T} \sum_{t=1}^T D_t^2 \right)^{1/2}$$
$$\text{OPVQ Index} = 100 \times (1 - D_{\text{pooled}})$$

### 4. Mean Opinion Score (MOS) Prediction
Maps 0–100 quality indices (both OPVQ and VMAF) to the standardized ITU 5-point Mean Opinion Score scale:
$$\text{MOS}_{\text{pred}} = 1.0 + 4.0 \times \left( \frac{\text{Score}}{100.0} \right)$$
$$\text{MOS}_{\text{pred}} \in [1.0, 5.0]$$

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
# Alternatively: pip install numpy matplotlib pandas
```

---

## Usage Instructions

### 1. Run OPVQ & VMAF Evaluation (Default Benchmark)
Runs evaluation comparing `reference.mp4` and `distorted.mp4`:
```bash
python opvq_mos.py
```
*Outputs OPVQ Index, VMAF Score, Predicted MOS (1–5) for both metrics, and Metric Agreement ($\Delta\text{MOS}$).*

### 2. Evaluate Custom Video Pairs
```bash
python opvq_mos.py /path/to/reference.mp4 /path/to/distorted.mp4
```

### 3. Specify Metric Mode
```bash
# Standalone OPVQ only
python opvq_mos.py reference.mp4 distorted.mp4 --mode opvq

# VMAF only
python opvq_mos.py reference.mp4 distorted.mp4 --mode vmaf

# Dual evaluation (both metrics)
python opvq_mos.py reference.mp4 distorted.mp4 --mode both
```

### 4. Run Synthetic Compression Benchmark
Generates an H.264 compressed test artifact and benchmarks degradation:
```bash
python test_compression.py
```

---

## Academic Note

> **Note:** Always report results as **Predicted MOS** or **MOS-equivalent scores**, not raw subjective MOS.
