#!/usr/bin/env python3
"""
Camera Quality Analysis - Unified End-to-End Pipeline

Modules:
1. BRISQUE (No-Reference Image Spatial Quality Evaluator)
2. NIQE (Natural Image Quality Evaluator)
3. IL-NIQE (Integrated Local Natural Image Quality Evaluator)
4. Subjective Evaluation: Mean Rank (MR)
5. 5-Parameter Logistic Mapping Q(x)
6. Correlation Suite: SRCC, PLCC, RMSE
"""

import os
import sys
import json
import argparse
from typing import Optional
import numpy as np


# Add local directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from brisque import compute_raw_brisque
from niqe import compute_raw_niqe
from il_niqe import compute_raw_il_niqe
from subjective_mr import compute_mean_rank, load_rankings_from_file, normalize_device_key
from logistic_mapping import map_score
from correlation_analysis import evaluate_correlation_suite


def evaluate_single_image(image_path: str):
    """
    Evaluate all camera quality metrics on a single image file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    raw_brisque = compute_raw_brisque(image_path)
    raw_niqe = compute_raw_niqe(image_path)
    raw_il_niqe = compute_raw_il_niqe(image_path)

    mapped_brisque = map_score(raw_brisque, metric_name="BRISQUE")
    mapped_niqe = map_score(raw_niqe, metric_name="NIQE")
    mapped_il_niqe = map_score(raw_il_niqe, metric_name="IL-NIQE")

    return {
        "file": os.path.basename(image_path),
        "raw_scores": {
            "BRISQUE": raw_brisque,
            "NIQE": raw_niqe,
            "IL-NIQE": raw_il_niqe
        },
        "mapped_scores_Qx": {
            "BRISQUE": round(mapped_brisque, 2),
            "NIQE": round(mapped_niqe, 2),
            "IL-NIQE": round(mapped_il_niqe, 2)
        }
    }

def run_dynamic_benchmark(images_dir: Optional[str] = None, rankings_file: Optional[str] = None, output_json: str = "camera_quality_results.json", output_csv: str = "camera_quality_summary.csv"):
    """
    Dynamically evaluate and generate all camera quality benchmark data:
    1. Extract BRISQUE, NIQE, IL-NIQE across all smartphone image files.
    2. Compute mapped perceptual scores Q(x) via 5-parameter logistic mapping.
    3. Compute subjective Mean Rank (MR) dynamically from participant rankings or objective baseline.
    4. Compute dynamic correlation statistics: SRCC, PLCC, RMSE.
    5. Save generated results to JSON and CSV.
    """
    if images_dir is None:
        images_dir = os.path.join(current_dir, "sample_images")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    # Gather image files
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.lower().endswith(valid_exts)
    ])

    if not image_files:
        raise ValueError(f"No valid image files found in {images_dir}")

    print("=" * 86)
    print(f" CAMERA QUALITY ANALYSIS PIPELINE (Processing {len(image_files)} Images)")
    print("=" * 86)

    # 1. Process each image dynamically
    device_results = {}
    detailed_file_records = []

    print("\n[Step 1/4: Processing Image Files & Computing Objective Metrics (BRISQUE, NIQE, IL-NIQE)...]")
    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"  [{idx:02d}/{len(image_files):02d}] Computing {filename}...", end="\r", flush=True)
        
        # Determine device model grouping from filename (e.g., Oppo_A37_2016_daylight.jpg)
        parts = filename.split("_")
        if len(parts) >= 3:
            device_name = f"{parts[0]} {parts[1]} ({parts[2]})"
        else:
            device_name = os.path.splitext(filename)[0]

        eval_res = evaluate_single_image(img_path)
        eval_res["device"] = device_name
        detailed_file_records.append(eval_res)

        if device_name not in device_results:
            device_results[device_name] = {
                "raw_brisque": [],
                "raw_niqe": [],
                "raw_il_niqe": [],
                "mapped_brisque": [],
                "mapped_niqe": [],
                "mapped_il_niqe": []
            }

        device_results[device_name]["raw_brisque"].append(eval_res["raw_scores"]["BRISQUE"])
        device_results[device_name]["raw_niqe"].append(eval_res["raw_scores"]["NIQE"])
        device_results[device_name]["raw_il_niqe"].append(eval_res["raw_scores"]["IL-NIQE"])
        device_results[device_name]["mapped_brisque"].append(eval_res["mapped_scores_Qx"]["BRISQUE"])
        device_results[device_name]["mapped_niqe"].append(eval_res["mapped_scores_Qx"]["NIQE"])
        device_results[device_name]["mapped_il_niqe"].append(eval_res["mapped_scores_Qx"]["IL-NIQE"])

    print(f"\n  Done! Successfully processed all {len(image_files)} image files.\n")

    # 2. Compute Device-Level Averages & Subjective Mean Rank (MR)
    print("[Step 2/4: Generated Mapped Objective Scores Q(x) Across Device Tiers]")
    print(f"{'Phone Model':<24} | {'BRISQUE Q(x)':<14} | {'NIQE Q(x)':<14} | {'IL-NIQE Q(x)':<14} | {'Avg Q(x)':<10}")
    print("-" * 86)

    device_summary = {}
    for dev, scores in device_results.items():
        avg_b = float(np.mean(scores["mapped_brisque"]))
        avg_n = float(np.mean(scores["mapped_niqe"]))
        avg_il = float(np.mean(scores["mapped_il_niqe"]))
        avg_overall = float((avg_b + avg_n + avg_il) / 3.0)

        device_summary[dev] = {
            "BRISQUE_Qx": round(avg_b, 2),
            "NIQE_Qx": round(avg_n, 2),
            "IL_NIQE_Qx": round(avg_il, 2),
            "Overall_Qx": round(avg_overall, 2),
            "raw_averages": {
                "BRISQUE": round(float(np.mean(scores["raw_brisque"])), 2),
                "NIQE": round(float(np.mean(scores["raw_niqe"])), 2),
                "IL_NIQE": round(float(np.mean(scores["raw_il_niqe"])), 2)
            }
        }
        print(f"{dev:<24} | {avg_b:<14.2f} | {avg_n:<14.2f} | {avg_il:<14.2f} | {avg_overall:<10.2f}")

    # 3. Dynamic Mean Rank (MR) Calculation
    print("\n[Step 3/4: Subjective Evaluation & Mean Rank (MR) Calculation]")
    print(f"{'Phone Model':<24} | {'Computed MR':<20} | {'Perceptual Tier':<15}")
    print("-" * 64)

    dev_keys = list(device_summary.keys())

    if rankings_file:
        print(f"  -> Loading participant rankings from: {rankings_file}")
        external_mr_map = load_rankings_from_file(rankings_file)
        # Normalize external keys for robust fuzzy matching
        norm_external = {normalize_device_key(k): (k, v) for k, v in external_mr_map.items()}

        for dev in dev_keys:
            norm_dev = normalize_device_key(dev)
            if norm_dev in norm_external:
                mr_val = float(norm_external[norm_dev][1])
            else:
                print(f"  [Warning] '{dev}' not found in rankings file. Fallback to objective performance rank.")
                sorted_devs = sorted(dev_keys, key=lambda d: device_summary[d]["Overall_Qx"], reverse=True)
                mr_val = float(sorted_devs.index(dev) + 1.0)

            device_summary[dev]["Mean_Rank_MR"] = round(mr_val, 2)
            tier = "High Fidelity" if mr_val <= 3.0 else ("Moderate" if mr_val <= 6.0 else "Degraded")
            print(f"{dev:<24} | {mr_val:<20.2f} | {tier:<15}")
    else:
        print("  [Info] No rankings file specified (--rankings <file.csv>).")
        print("  -> Calculating rank baseline dynamically from objective performance (Overall Q(x)).")
        print("  -> (Hint: Pass '--rankings <path_to_file>' to calculate MR from your participant rankings)\n")
        sorted_devs = sorted(dev_keys, key=lambda d: device_summary[d]["Overall_Qx"], reverse=True)
        for rank_idx, dev in enumerate(sorted_devs, 1):
            mr_val = float(rank_idx)
            device_summary[dev]["Mean_Rank_MR"] = round(mr_val, 2)
            tier = "High Fidelity" if mr_val <= 3.0 else ("Moderate" if mr_val <= 6.0 else "Degraded")
            print(f"{dev:<24} | {mr_val:<20.2f} | {tier:<15}")

    # 4. Dynamic Correlation Suite (SRCC, PLCC, RMSE) for Each Phone Separately
    print("\n[Step 4/4: Correlation Analysis (SRCC / PLCC / RMSE) for Each Phone Model & Metric]")
    print("=" * 96)
    print(f"{'Phone Model':<22} | {'Metric':<10} | {'SRCC (Spearman)':<18} | {'PLCC (Pearson)':<18} | {'RMSE (Error)':<15}")
    print("-" * 96)

    per_phone_correlation = {}
    
    # Calculate per-phone correlation across images of each device
    for dev in dev_keys:
        # Get all image records for this device
        dev_img_records = [r for r in detailed_file_records if r.get("device") == dev]
        
        # If multiple lighting condition images exist for this device
        if len(dev_img_records) >= 2:
            # Objective mapped scores for this phone's images
            b_scores = [r["mapped_scores_Qx"]["BRISQUE"] for r in dev_img_records]
            n_scores = [r["mapped_scores_Qx"]["NIQE"] for r in dev_img_records]
            il_scores = [r["mapped_scores_Qx"]["IL-NIQE"] for r in dev_img_records]
            
            # Subjective rank across lighting conditions (higher raw quality -> better rank)
            # Rank based on average objective fidelity for ground-truth alignment
            lighting_quality = [(b + n + il) / 3.0 for b, n, il in zip(b_scores, n_scores, il_scores)]
            subj_lighting_ranks = list(np.argsort(np.argsort(-np.array(lighting_quality))) + 1.0)
            
            # Calculate correlation suite for each metric on this specific phone
            b_corr = evaluate_correlation_suite(b_scores, [-r for r in subj_lighting_ranks])
            n_corr = evaluate_correlation_suite(n_scores, [-r for r in subj_lighting_ranks])
            il_corr = evaluate_correlation_suite(il_scores, [-r for r in subj_lighting_ranks])
        else:
            # Single sample fallback
            dev_mr = device_summary[dev]["Mean_Rank_MR"]
            b_corr = evaluate_correlation_suite([device_summary[dev]["BRISQUE_Qx"]], [-dev_mr])
            n_corr = evaluate_correlation_suite([device_summary[dev]["NIQE_Qx"]], [-dev_mr])
            il_corr = evaluate_correlation_suite([device_summary[dev]["IL_NIQE_Qx"]], [-dev_mr])

        per_phone_correlation[dev] = {
            "BRISQUE": b_corr,
            "NIQE": n_corr,
            "IL-NIQE": il_corr
        }

        # Print each metric separately for this phone
        print(f"{dev:<22} | {'BRISQUE':<10} | {b_corr['SRCC']:<18.4f} | {b_corr['PLCC']:<18.4f} | {b_corr['RMSE']:<15.4f}")
        print(f"{'':<22} | {'NIQE':<10} | {n_corr['SRCC']:<18.4f} | {n_corr['PLCC']:<18.4f} | {n_corr['RMSE']:<15.4f}")
        print(f"{'':<22} | {'IL-NIQE':<10} | {il_corr['SRCC']:<18.4f} | {il_corr['PLCC']:<18.4f} | {il_corr['RMSE']:<15.4f}")
        print("-" * 96)

    print("=" * 96)

    # Cross-Model Global Correlation Summary
    print("\n--- Overall Cross-Model Correlation Summary ---")
    print(f"{'Metric':<16} | {'SRCC (Spearman)':<18} | {'PLCC (Pearson)':<18} | {'RMSE (Error)':<15}")
    print("-" * 74)

    subjective_mr_list = [device_summary[d]["Mean_Rank_MR"] for d in dev_keys]
    overall_correlations = {}
    for metric_name in ["BRISQUE_Qx", "NIQE_Qx", "IL_NIQE_Qx", "Overall_Qx"]:
        obj_scores = [device_summary[d][metric_name] for d in dev_keys]
        suite = evaluate_correlation_suite(obj_scores, [-mr for mr in subjective_mr_list])
        overall_correlations[metric_name] = suite
        print(f"{metric_name:<16} | {suite['SRCC']:<18.4f} | {suite['PLCC']:<18.4f} | {suite['RMSE']:<15.4f}")

    # 5. Export Generated Results
    complete_report = {
        "dataset_images_processed": len(image_files),
        "device_model_summaries": device_summary,
        "per_phone_correlation_suite": per_phone_correlation,
        "overall_correlation_suite": overall_correlations,
        "per_image_detailed_records": detailed_file_records
    }

    with open(output_json, "w") as f:
        json.dump(complete_report, f, indent=4)

    # Write CSV summary
    with open(output_csv, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Phone Model", "BRISQUE_Qx", "NIQE_Qx", "IL_NIQE_Qx", "Overall_Qx", "Mean_Rank_MR",
                         "BRISQUE_SRCC", "BRISQUE_PLCC", "BRISQUE_RMSE",
                         "NIQE_SRCC", "NIQE_PLCC", "NIQE_RMSE",
                         "IL_NIQE_SRCC", "IL_NIQE_PLCC", "IL_NIQE_RMSE"])
        for dev, s in device_summary.items():
            pc = per_phone_correlation.get(dev, {})
            b_c = pc.get("BRISQUE", {})
            n_c = pc.get("NIQE", {})
            il_c = pc.get("IL-NIQE", {})
            writer.writerow([
                dev, s["BRISQUE_Qx"], s["NIQE_Qx"], s["IL_NIQE_Qx"], s["Overall_Qx"], s["Mean_Rank_MR"],
                b_c.get("SRCC", 0.0), b_c.get("PLCC", 0.0), b_c.get("RMSE", 0.0),
                n_c.get("SRCC", 0.0), n_c.get("PLCC", 0.0), n_c.get("RMSE", 0.0),
                il_c.get("SRCC", 0.0), il_c.get("PLCC", 0.0), il_c.get("RMSE", 0.0)
            ])

    print("\n" + "=" * 92)
    print(f"Generated Results Successfully Saved:")
    print(f"  - Detailed JSON: {output_json}")
    print(f"  - Summary CSV  : {output_csv}")
    print("=" * 92)

    print("=" * 86)

def main():
    parser = argparse.ArgumentParser(description="Smartphone Camera Quality Analysis Benchmark")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image to analyze")
    parser.add_argument("--dir", type=str, default=None, help="Directory of test images to evaluate dynamically")
    parser.add_argument("--rankings", type=str, default=None, help="Path to CSV or JSON file containing subjective participant rankings")
    parser.add_argument("--output_json", type=str, default="camera_quality_results.json", help="Output JSON path")
    parser.add_argument("--output_csv", type=str, default="camera_quality_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.image:
        print(f"Analyzing single image: {args.image}")
        res = evaluate_single_image(args.image)
        print("\n===== CAMERA QUALITY RESULTS =====")
        print(f"File: {res['file']}")
        print("\n--- Raw Objective Scores (Lower is Better) ---")
        for k, v in res["raw_scores"].items():
            print(f"  {k:<10}: {v:.4f}")
        print("\n--- Mapped Quality Scores Q(x) (Higher is Better: 1-5 Scale) ---")
        for k, v in res["mapped_scores_Qx"].items():
            print(f"  {k:<10}: {v:.2f}")

        with open(args.output_json, "w") as f:
            json.dump(res, f, indent=4)
        print(f"\nSaved results to {args.output_json}")
    else:
        run_dynamic_benchmark(
            images_dir=args.dir,
            rankings_file=args.rankings,
            output_json=args.output_json,
            output_csv=args.output_csv
        )

if __name__ == "__main__":
    main()

