"""
Correlation Analysis: SRCC, PLCC, and RMSE

Formulas:
1. Spearman Rank-Order Correlation Coefficient (SRCC):
    SRCC = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
    where d_i is the difference between ranks of objective and subjective scores.

2. Pearson Linear Correlation Coefficient (PLCC):
    PLCC = sum((x_i - bar{x}) * (y_i - bar{y})) / sqrt(sum((x_i - bar{x})^2) * sum((y_i - bar{y})^2))

3. Root Mean Squared Error (RMSE):
    RMSE = sqrt((1 / N) * sum_{i=1}^N (x_i - y_i)^2)

Higher SRCC and PLCC values and lower RMSE values indicate stronger agreement
between subjective and objective quality estimates.
"""

import numpy as np
from scipy.stats import rankdata

def compute_srcc(x, y) -> float:
    """
    Compute Spearman Rank-Order Correlation Coefficient (SRCC).
    Formula:
        SRCC = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 1.0

    rx = rankdata(x)
    ry = rankdata(y)
    d = rx - ry
    sum_d_sq = np.sum(d ** 2)

    srcc = 1.0 - (6.0 * sum_d_sq) / (n * (n ** 2 - 1.0))
    return float(np.clip(srcc, -1.0, 1.0))

def compute_plcc(x, y) -> float:
    """
    Compute Pearson Linear Correlation Coefficient (PLCC).
    Formula:
        PLCC = sum((x_i - bar{x}) * (y_i - bar{y})) / (sqrt(sum((x_i - bar{x})^2)) * sqrt(sum((y_i - bar{y})^2)))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 1.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    dev_x = x - mean_x
    dev_y = y - mean_y

    numerator = np.sum(dev_x * dev_y)
    denominator = np.sqrt(np.sum(dev_x ** 2) * np.sum(dev_y ** 2))

    if denominator < 1e-12:
        return 0.0

    plcc = numerator / denominator
    return float(np.clip(plcc, -1.0, 1.0))

def compute_rmse(x, y) -> float:
    """
    Compute Root Mean Squared Error (RMSE).
    Formula:
        RMSE = sqrt((1 / N) * sum_{i=1}^N (x_i - y_i)^2)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        return 0.0

    mse = np.mean((x - y) ** 2)
    return float(np.sqrt(mse))

def evaluate_correlation_suite(objective_scores, subjective_scores):
    """
    Calculate full evaluation suite: SRCC, PLCC, and RMSE.
    """
    srcc = compute_srcc(objective_scores, subjective_scores)
    plcc = compute_plcc(objective_scores, subjective_scores)
    rmse = compute_rmse(objective_scores, subjective_scores)

    return {
        "SRCC": round(srcc, 4),
        "PLCC": round(plcc, 4),
        "RMSE": round(rmse, 4)
    }

# Ground truth benchmarks reported in Table 10 of the paper
PAPER_TABLE_10_BENCHMARK = {
    "Oppo A37 2016": {"BRISQUE": [0.91, 0.997, 0.097], "NIQE": [0.89, 0.994, 0.113], "IL-NIQE": [0.92, 0.994, 0.093]},
    "Oppo A37 2017": {"BRISQUE": [0.94, 0.997, 0.094], "NIQE": [0.92, 0.995, 0.117], "IL-NIQE": [0.93, 0.998, 0.089]},
    "Oppo A37 2018": {"BRISQUE": [0.98, 0.998, 0.093], "NIQE": [0.96, 0.998, 0.119], "IL-NIQE": [0.96, 0.998, 0.087]},
    "Vivo Y67 2016": {"BRISQUE": [0.92, 0.987, 0.094], "NIQE": [0.92, 0.996, 0.118], "IL-NIQE": [0.94, 0.997, 0.091]},
    "Vivo Y67 2017": {"BRISQUE": [0.96, 0.994, 0.092], "NIQE": [0.95, 0.997, 0.122], "IL-NIQE": [0.95, 0.998, 0.086]},
    "Vivo Y67 2018": {"BRISQUE": [1.00, 0.997, 0.084], "NIQE": [0.97, 0.999, 0.127], "IL-NIQE": [0.98, 0.998, 0.083]},
    "Redmi 5A 2016": {"BRISQUE": [0.84, 0.967, 0.098], "NIQE": [0.82, 0.984, 0.109], "IL-NIQE": [0.85, 0.972, 0.097]},
    "Redmi 5A 2017": {"BRISQUE": [0.87, 0.975, 0.097], "NIQE": [0.84, 0.989, 0.114], "IL-NIQE": [0.86, 0.981, 0.093]},
    "Redmi 5A 2018": {"BRISQUE": [0.91, 0.981, 0.093], "NIQE": [0.87, 0.994, 0.118], "IL-NIQE": [0.89, 0.987, 0.091]},
}

if __name__ == "__main__":
    print("===== CORRELATION SUITE (SRCC, PLCC, RMSE) TEST =====")
    # Verification with synthetic paired observations
    subj = [1.4, 1.8, 2.9, 3.5, 4.2]
    obj = [1.38, 1.82, 2.95, 3.48, 4.19]
    res = evaluate_correlation_suite(obj, subj)
    print(f"Sample Subjective: {subj}")
    print(f"Sample Objective : {obj}")
    print(f"Calculated Metrics: SRCC={res['SRCC']}, PLCC={res['PLCC']}, RMSE={res['RMSE']}")
