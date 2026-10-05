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

if __name__ == "__main__":
    print("===== CORRELATION SUITE (SRCC, PLCC, RMSE) TEST =====")
    # Verification with synthetic paired observations
    subj = [1.4, 1.8, 2.9, 3.5, 4.2]
    obj = [1.38, 1.82, 2.95, 3.48, 4.19]
    res = evaluate_correlation_suite(obj, subj)
    print(f"Sample Subjective: {subj}")
    print(f"Sample Objective : {obj}")
    print(f"Calculated Metrics: SRCC={res['SRCC']}, PLCC={res['PLCC']}, RMSE={res['RMSE']}")
