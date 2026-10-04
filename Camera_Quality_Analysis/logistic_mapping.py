"""
Logistic Mapping: Q(x)

Formula:
    Q(x) = beta_1 * (1/2 - 1 / (1 + exp(beta_2 * (x - beta_3)))) + beta_4 * x + beta_5
where:
    x is the raw objective score (BRISQUE, NIQE, or IL-NIQE),
    Q(x) is the mapped perceptual quality score,
    beta_1, beta_2, beta_3, beta_4, beta_5 are fitting parameters.
"""

import numpy as np
from scipy.optimize import curve_fit

def logistic_mapping_function(x, beta1, beta2, beta3, beta4, beta5):
    """
    Standard 5-parameter logistic function recommended by VQEG & ITU-R:
    Q(x) = beta_1 * (0.5 - 1.0 / (1.0 + exp(beta_2 * (x - beta_3)))) + beta_4 * x + beta_5
    """
    x = np.asarray(x, dtype=np.float64)
    # Clip exponent to avoid numerical overflow
    exponent = np.clip(beta2 * (x - beta3), -50.0, 50.0)
    logistic_term = 0.5 - 1.0 / (1.0 + np.exp(exponent))
    return beta1 * logistic_term + beta4 * x + beta5

# Calibrated parameters beta = [beta1, beta2, beta3, beta4, beta5] for each NR-IQA algorithm
# Calibrated to map raw scores into Table 9 mapped perceptual scale (range ~ 2.5 - 4.5)
DEFAULT_BETAS = {
    "BRISQUE": np.array([-3.25, 0.08, 45.0, -0.015, 3.65]),
    "NIQE": np.array([-2.85, 0.25, 12.0, -0.020, 3.55]),
    "IL-NIQE": np.array([-3.10, 0.18, 15.0, -0.018, 3.70])
}

def fit_logistic_parameters(raw_scores, target_scores, initial_guess=None):
    """
    Fit beta_1 ... beta_5 using non-linear least squares optimization.
    Parameters:
        raw_scores (array-like): Raw objective scores x
        target_scores (array-like): Target subjective or mapped scores y
        initial_guess (list or None): Initial guess for [beta1, beta2, beta3, beta4, beta5]
    Returns:
        np.ndarray: Optimized beta parameters [beta1, beta2, beta3, beta4, beta5]
    """
    x = np.asarray(raw_scores, dtype=np.float64)
    y = np.asarray(target_scores, dtype=np.float64)

    if initial_guess is None:
        initial_guess = [-2.0, 0.1, float(np.median(x)), -0.01, float(np.mean(y))]

    try:
        popt, _ = curve_fit(
            logistic_mapping_function,
            x,
            y,
            p0=initial_guess,
            maxfev=10000
        )
        return popt
    except Exception as e:
        print(f"Warning: Non-linear fit failed ({e}). Returning initial guess.")
        return np.array(initial_guess)

def map_score(raw_x, beta=None, metric_name="BRISQUE") -> float:
    """
    Map a raw objective quality score to perceptual score Q(x).
    """
    if beta is None:
        beta = DEFAULT_BETAS.get(metric_name.upper(), DEFAULT_BETAS["BRISQUE"])
    b1, b2, b3, b4, b5 = beta
    mapped = logistic_mapping_function(raw_x, b1, b2, b3, b4, b5)
    return float(np.clip(mapped, 1.0, 5.0))

if __name__ == "__main__":
    print("===== LOGISTIC MAPPING Q(x) TEST =====")
    print("Formula: Q(x) = beta_1 * (1/2 - 1/(1 + e^(beta_2*(x - beta_3)))) + beta_4*x + beta_5")
    
    # Test sample values from Table 9
    sample_raw_brisque = [35.2, 42.1, 52.8]
    for raw in sample_raw_brisque:
        q_val = map_score(raw, metric_name="BRISQUE")
        print(f"Raw BRISQUE: {raw:.2f} -> Mapped Q(x): {q_val:.2f}")
