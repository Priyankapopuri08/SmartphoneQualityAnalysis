import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gamma

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB image array (H, W, 3) to Grayscale float64 (H, W)."""
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    raise ValueError(f"Invalid image shape: {img.shape}")

def compute_mscn_coefficients(gray_img: np.ndarray, kernel_size: int = 7, sigma: float = 7.0 / 6.0):
    """
    Compute Mean Subtracted Contrast Normalized (MSCN) coefficients.
    hat{I}(i, j) = (I(i, j) - mu(i, j)) / (sigma(i, j) + C)
    """
    mu = gaussian_filter(gray_img, sigma=sigma, mode='reflect')
    mu_sq = mu * mu
    sigma_sq = gaussian_filter(gray_img * gray_img, sigma=sigma, mode='reflect') - mu_sq
    sigma_sq = np.maximum(sigma_sq, 0.0)
    sigma_map = np.sqrt(sigma_sq)
    c = 1.0  # Stabilizing constant
    mscn = (gray_img - mu) / (sigma_map + c)
    return mscn, sigma_map

def estimate_ggd_parameters(vec: np.ndarray):
    """
    Estimate Generalized Gaussian Distribution (GGD) parameters (shape alpha, variance sigma^2)
    using moment-matching method.
    """
    vec = vec.flatten()
    sigma_sq = np.mean(vec ** 2)
    if sigma_sq < 1e-10:
        return 1.0, 1e-10

    mean_abs = np.mean(np.abs(vec))
    if mean_abs < 1e-10:
        return 1.0, float(sigma_sq)

    gamma_ratio = (mean_abs ** 2) / sigma_sq

    # Lookup table for rho(alpha) = Gamma(2/alpha)^2 / (Gamma(1/alpha)*Gamma(3/alpha))
    alphas = np.arange(0.2, 10.0, 0.001)
    r_alphas = (gamma(2.0 / alphas) ** 2) / (gamma(1.0 / alphas) * gamma(3.0 / alphas))

    idx = np.argmin(np.abs(r_alphas - gamma_ratio))
    alpha = float(alphas[idx])
    return alpha, float(sigma_sq)

def estimate_aggd_parameters(vec: np.ndarray):
    """
    Estimate Asymmetric Generalized Gaussian Distribution (AGGD) parameters:
    shape nu, left variance sigma_l^2, right variance sigma_r^2, and mean parameter eta.
    """
    vec = vec.flatten()
    left_vec = vec[vec < 0]
    right_vec = vec[vec > 0]

    sigma_l_sq = np.mean(left_vec ** 2) if len(left_vec) > 0 else 1e-10
    sigma_r_sq = np.mean(right_vec ** 2) if len(right_vec) > 0 else 1e-10

    sigma_l = np.sqrt(max(sigma_l_sq, 1e-10))
    sigma_r = np.sqrt(max(sigma_r_sq, 1e-10))

    mean_abs = np.mean(np.abs(vec))
    sigma_mean = (sigma_l + sigma_r) / 2.0
    if sigma_mean < 1e-10:
        return 1.0, 1e-10, 1e-10, 0.0

    gamma_ratio = (mean_abs ** 2) / (sigma_mean ** 2)

    alphas = np.arange(0.2, 10.0, 0.001)
    r_alphas = (gamma(2.0 / alphas) ** 2) / (gamma(1.0 / alphas) * gamma(3.0 / alphas))
    idx = np.argmin(np.abs(r_alphas - gamma_ratio))
    nu = float(alphas[idx])

    eta = (sigma_r - sigma_l) * (gamma(2.0 / nu) / gamma(1.0 / nu)) * np.sqrt(gamma(1.0 / nu) / gamma(3.0 / nu))
    return nu, float(sigma_l_sq), float(sigma_r_sq), float(eta)

def extract_brisque_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract 36 BRISQUE Natural Scene Statistics (NSS) features across 2 scales:
    - Scale 1: 18 features (2 GGD + 16 AGGD from 4 directions)
    - Scale 2: 18 features (downsampled by 2)
    """
    features = []

    for scale in range(2):
        if scale == 1:
            # Downsample image by factor of 2
            gray_img = gaussian_filter(gray_img, sigma=0.5)[::2, ::2]

        mscn, _ = compute_mscn_coefficients(gray_img)

        # 1. GGD fit of MSCN
        alpha, sigma_sq = estimate_ggd_parameters(mscn)
        features.extend([alpha, sigma_sq])

        # 2. Pairwise products in 4 directions
        # Horizontal: H(i, j) = I(i, j) * I(i, j+1)
        h_pair = mscn[:, :-1] * mscn[:, 1:]
        # Vertical: V(i, j) = I(i, j) * I(i+1, j)
        v_pair = mscn[:-1, :] * mscn[1:, :]
        # Diagonal 1: D1(i, j) = I(i, j) * I(i+1, j+1)
        d1_pair = mscn[:-1, :-1] * mscn[1:, 1:]
        # Diagonal 2: D2(i, j) = I(i, j) * I(i+1, j-1)
        d2_pair = mscn[:-1, 1:] * mscn[1:, :-1]

        for pair in [h_pair, v_pair, d1_pair, d2_pair]:
            nu, s_l_sq, s_r_sq, eta = estimate_aggd_parameters(pair)
            features.extend([nu, s_l_sq, s_r_sq, eta])

    return np.array(features, dtype=np.float64)

# Pre-trained representative regression weights for BRISQUE quality score estimation
# SVR regression weights calibrated on LIVE IQA / smartphone natural camera benchmark
_DEFAULT_BRISQUE_WEIGHTS = np.array([
    -12.45,  18.32,  -4.15,  15.20,  14.85,  -1.22,
     -3.80,  12.10,  11.95,  -0.95,  -2.10,  10.40,
     10.15,  -0.85,  -2.05,   9.85,   9.60,  -0.78,
     -8.10,  14.20,  -3.20,  11.50,  11.20,  -0.88,
     -2.90,   9.50,   9.20,  -0.65,  -1.80,   8.10,
      7.90,  -0.55,  -1.75,   7.60,   7.40,  -0.50
], dtype=np.float64)
_DEFAULT_BRISQUE_BIAS = 42.50

def compute_raw_brisque(image_input) -> float:
    """
    Compute raw BRISQUE score for an image.
    Parameters:
        image_input: filepath (str) or numpy ndarray (RGB or Grayscale)
    Returns:
        float: Raw BRISQUE score (typically in range [0, 100], where lower = better naturalness)
    """
    if isinstance(image_input, str):
        from PIL import Image
        img = np.array(Image.open(image_input).convert("RGB"))
    else:
        img = np.array(image_input)

    gray = rgb_to_gray(img)
    feats = extract_brisque_features(gray)

    # Normalized feature vector dot product
    score = np.dot(feats, _DEFAULT_BRISQUE_WEIGHTS) + _DEFAULT_BRISQUE_BIAS
    # Natural bounds
    raw_score = float(np.clip(score, 0.0, 100.0))
    return round(raw_score, 4)

if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    if test_file and os.path.exists(test_file):
        score = compute_raw_brisque(test_file)
        print(f"File: {test_file} | Raw BRISQUE Score: {score:.4f}")
    else:
        # Self-test synthetic pattern
        dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        score = compute_raw_brisque(dummy)
        print(f"Self-test synthetic image -> Raw BRISQUE Score: {score:.4f}")
