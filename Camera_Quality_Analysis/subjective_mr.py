"""
Subjective Evaluation: Mean Rank (MR)

Formula:
    MR_j = (1 / N) * sum_{i=1}^N r_{ij}
where:
    r_{ij} is the rank assigned by participant i to image/device j,
    N is the total number of participants.
    Lower MR values indicate superior perceived image quality.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd


def compute_mean_rank(rank_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Mean Rank (MR) across participants.
    
    Parameters:
        rank_matrix (np.ndarray): Shape (N, J) where N is number of participants,
                                  and J is number of images/devices.
    Returns:
        np.ndarray: Array of size J containing MR_j for each device/image.
    """
    rank_matrix = np.asarray(rank_matrix, dtype=np.float64)
    if rank_matrix.ndim == 1:
        return rank_matrix
    n_participants = rank_matrix.shape[0]
    if n_participants == 0:
        raise ValueError("Participant count N cannot be zero.")
    mr = np.mean(rank_matrix, axis=0)
    return mr


def normalize_device_key(name: str) -> str:
    """Normalize device name for resilient matching (e.g. 'Oppo A37 (2016)' -> 'oppoa372016')."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def load_rankings_from_file(file_path: str) -> dict:
    """
    Load participant rankings from an external file (CSV or JSON) and compute
    the dynamic Subjective Mean Rank (MR) for each device/item.
    
    No hardcoded table data is used.
    
    Supported File Formats:
    1. Wide CSV / Excel format:
       Columns represent phone models/devices, rows represent individual participant rankings.
       Optional first column can be 'Evaluator' / 'Participant' / 'ID'.
       Example:
           Evaluator,Oppo A37 (2016),Oppo A37 (2017),Oppo A37 (2018)
           Participant_1,1,2,3
           Participant_2,1,2,3
           
    2. Long CSV format:
       Rows with columns: [Participant/Evaluator, Device/Model/Phone/Item, Rank/Score]
       Example:
           Evaluator,Device,Rank
           P1,Oppo A37 (2016),1
           P1,Oppo A37 (2017),2
           P2,Oppo A37 (2016),1
           
    3. Direct Summary CSV:
       Rows with columns: [Device/Model/Phone/Item, Mean_Rank/MR/Rank]
       Example:
           Device,Mean_Rank
           Oppo A37 (2016),1.42
           Oppo A37 (2017),1.84
           
    4. JSON format:
       Dictionary where keys are device names and values are lists of participant ranks:
       {"Oppo A37 (2016)": [1, 2, 1], "Oppo A37 (2017)": [2, 1, 2]}
       OR list of record dictionaries:
       [{"Device": "Oppo A37 (2016)", "Rank": 1}, ...]

    Returns:
        dict: {device_name: computed_mr_float}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Rankings file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            mr_results = {}
            for dev, scores in data.items():
                if isinstance(scores, list):
                    arr = np.asarray(scores, dtype=np.float64)
                    mr_results[str(dev)] = round(float(np.mean(arr)), 4)
                elif isinstance(scores, (int, float)):
                    mr_results[str(dev)] = round(float(scores), 4)
            return mr_results

        elif isinstance(data, list):
            df = pd.DataFrame(data)
            return _parse_dataframe_rankings(df)
        else:
            raise ValueError("Unsupported JSON format in rankings file.")

    else:
        # Load as CSV
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine="python")
        return _parse_dataframe_rankings(df)


def _parse_dataframe_rankings(df: pd.DataFrame) -> dict:
    """Internal helper to extract device mean ranks from various DataFrame layouts."""
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # Case 1: Long format (e.g. Evaluator, Device, Rank)
    dev_col_match = [c for c in cols if c.lower() in ("device", "phone", "model", "phone model", "item", "image")]
    rank_col_match = [c for c in cols if c.lower() in ("rank", "ranking", "score", "mr", "mean_rank", "mean rank")]

    if dev_col_match and rank_col_match:
        dev_col = dev_col_match[0]
        rank_col = rank_col_match[0]
        grouped = df.groupby(dev_col)[rank_col].mean()
        return {str(k): round(float(v), 4) for k, v in grouped.items()}

    # Case 2: Direct Summary 2-column format (Device, Mean_Rank)
    if len(cols) == 2 and any(c.lower() in ("device", "phone", "model", "name") for c in cols):
        dev_col = cols[0] if cols[0].lower() in ("device", "phone", "model", "name") else cols[1]
        val_col = cols[1] if dev_col == cols[0] else cols[0]
        res = {}
        for _, row in df.iterrows():
            dev = str(row[dev_col]).strip()
            val = float(row[val_col])
            res[dev] = round(val, 4)
        return res

    # Case 3: Wide format (Columns are devices, rows are participants)
    id_cols = {"evaluator", "participant", "id", "user", "rater", "index", "unnamed: 0"}
    device_cols = [c for c in cols if c.lower() not in id_cols]

    mr_results = {}
    for col in device_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 0:
            mr_results[str(col).strip()] = round(float(series.mean()), 4)

    if not mr_results:
        raise ValueError("Could not extract any valid device rankings from the provided file.")

    return mr_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate Subjective Mean Rank (MR) from a file")
    parser.add_argument("file", nargs="?", default=None, help="Path to CSV or JSON file containing rankings")
    args = parser.parse_args()

    if args.file:
        res = load_rankings_from_file(args.file)
        print("===== SUBJECTIVE MEAN RANK (MR) FROM FILE =====")
        print(f"File: {args.file}")
        print(f"{'Device / Item':<24} | {'Mean Rank (MR)':<15}")
        print("-" * 42)
        for dev, mr in sorted(res.items(), key=lambda x: x[1]):
            print(f"{dev:<24} | {mr:<15.4f}")
    else:
        print("===== SUBJECTIVE MEAN RANK (MR) MODULE =====")
        print("Formula: MR_j = (1 / N) * sum_{i=1}^N r_{ij}")
        print("\nUsage:")
        print("  python subjective_mr.py <path_to_rankings.csv>")
        print("\nSupported formats:")
        print("  - Wide CSV: Evaluator, Device1, Device2, Device3, ...")
        print("  - Long CSV: Evaluator, Device, Rank")
        print("  - JSON: {'Device1': [1, 2, ...], 'Device2': [2, 1, ...]}")
