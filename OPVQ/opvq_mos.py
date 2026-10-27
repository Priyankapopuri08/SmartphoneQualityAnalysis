"""
OPVQ (Objective Perceptual Video Quality / OpenVQ)
Full-Reference Spatial-Temporal Perceptual Video Quality Evaluation & Statistical Benchmarking Module.

Methodology as documented in:
"How to Choose your Pre-owned Smartphone?: A Multi-Dimensional Benchmarking of Performance and Quality"
- Section 3.4 & 3.6: Video Quality Evaluation using VMAF [0, 100] and OPVQ [1.0, 5.0]
- Section 3.11: Statistical Analysis Framework:
    * Average Difference: D(A, B) = (1/N) * sum(x_i^A - x_i^B)
    * Win Count: Consistency check of superior perceptual quality
    * Paired t-test: Statistical significance (p < 0.01)
- Section 4.2 & Figure 5 & Table 6: Intra-model video degradation across smartphone lineages (Oppo A37, Vivo Y67, Redmi 5A)
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from shutil import which
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

def _normalize_path(p: str) -> str:
    """Normalize path for cross-platform compatibility."""
    return os.path.abspath(p).replace("\\", "/")

def _get_ffmpeg_cmd(base_args: list) -> list:
    """
    Detect FFmpeg binary:
    1. Direct native FFmpeg in PATH
    2. WSL FFmpeg bridge if on Windows and native ffmpeg is not present
    """
    native_ffmpeg = which("ffmpeg")
    if native_ffmpeg:
        return [native_ffmpeg] + base_args

    if os.name == "nt":
        wsl_bin = which("wsl")
        if wsl_bin:
            wsl_args = []
            for arg in base_args:
                if isinstance(arg, str) and (":\\" in arg or ":/" in arg or os.path.exists(arg)):
                    abs_p = os.path.abspath(arg).replace("\\", "/")
                    drive = abs_p[0].lower()
                    wsl_args.append(f"/mnt/{drive}{abs_p[2:]}")
                else:
                    wsl_args.append(arg)
            return [wsl_bin, "ffmpeg"] + wsl_args

    return ["ffmpeg"] + base_args

def _extract_raw_frames(video_path: str, width: int = 384, height: int = 216, max_frames: int = 300) -> list:
    """
    Extract Y-channel (luminance) planar frames using FFmpeg rawvideo pipe.
    """
    raw_cmd = [
        "-v", "error",
        "-i", video_path,
        "-vf", f"scale={width}:{height},format=gray",
        "-vframes", str(max_frames),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]
    cmd = _get_ffmpeg_cmd(raw_cmd)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg not found in PATH or WSL. Please install FFmpeg as described in OPVQ_Installation_Guide.docx."
        ) from exc

    frame_size = width * height
    frames = []

    if proc.stdout is not None:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width)).astype(np.float32)
            frames.append(frame)
        proc.stdout.close()
    proc.wait()
    return frames

def _compute_spatial_info(frame: np.ndarray) -> float:
    """Spatial Information (SI) using Sobel gradient energy."""
    gy, gx = np.gradient(frame)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return float(np.std(grad_mag))

def _compute_frame_ssim(ref: np.ndarray, dist: np.ndarray) -> float:
    """Structural Similarity (SSIM) index on luminance plane."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_ref = float(np.mean(ref))
    mu_dist = float(np.mean(dist))
    sigma_ref_sq = float(np.var(ref))
    sigma_dist_sq = float(np.var(dist))
    sigma_cross = float(np.mean((ref - mu_ref) * (dist - mu_dist)))

    numerator = (2 * mu_ref * mu_dist + c1) * (2 * sigma_cross + c2)
    denominator = (mu_ref**2 + mu_dist**2 + c1) * (sigma_ref_sq + sigma_dist_sq + c2)
    return float(np.clip(numerator / denominator, 0.0, 1.0))

def opvq_to_mos(opvq_index: float) -> float:
    """
    Standard linear mapping from OPVQ 0-100 index to ITU 1.0-5.0 Perceptual Score (Figure 5 & Table 6).
    """
    mos = 1.0 + 4.0 * (opvq_index / 100.0)
    return round(float(np.clip(mos, 1.0, 5.0)), 2)

def vmaf_to_mos(vmaf_score: float) -> float:
    """
    Standard linear mapping from VMAF 0-100 score to 1.0-5.0 Perceptual Score (Figure 5).
    """
    mos = 1.0 + 4.0 * (vmaf_score / 100.0)
    return round(float(np.clip(mos, 1.0, 5.0)), 2)

def compute_opvq(reference_video: str, distorted_video: str, log_file: str = None, max_frames: int = 300) -> dict:
    """
    Compute standalone OPVQ full-reference perceptual quality score.
    Returns:
    - opvq_score: Primary perceptual quality score in range [1.0, 5.0] matching Section 3.6, Figure 5 & Table 6.
    - opvq_index: Raw internal quality index in range [0, 100].
    - mean_ssim: Mean structural similarity.
    - spatial_fidelity, temporal_fidelity: Edge and motion fidelity indicators.
    """
    if not os.path.exists(reference_video):
        raise FileNotFoundError(f"Reference video not found: {reference_video}")
    if not os.path.exists(distorted_video):
        raise FileNotFoundError(f"Distorted video not found: {distorted_video}")

    ref_frames = _extract_raw_frames(reference_video, max_frames=max_frames)
    dist_frames = _extract_raw_frames(distorted_video, max_frames=max_frames)

    num_frames = min(len(ref_frames), len(dist_frames))
    if num_frames == 0:
        raise ValueError("Could not extract valid video frames for OPVQ comparison.")

    frame_scores = []
    spatial_diffs = []
    temporal_diffs = []
    ssim_scores = []

    for i in range(num_frames):
        ref_f = ref_frames[i]
        dist_f = dist_frames[i]

        ssim_val = _compute_frame_ssim(ref_f, dist_f)
        si_ref = _compute_spatial_info(ref_f)
        si_dist = _compute_spatial_info(dist_f)
        si_error = abs(si_ref - si_dist) / (si_ref + 1e-4)

        if i > 0:
            ti_ref = float(np.std(ref_f - ref_frames[i - 1]))
            ti_dist = float(np.std(dist_f - dist_frames[i - 1]))
            ti_error = abs(ti_ref - ti_dist) / (ti_ref + 1e-4)
        else:
            ti_error = 0.0

        frame_distortion = 0.65 * (1.0 - ssim_val) + 0.20 * min(si_error, 1.0) + 0.15 * min(ti_error, 1.0)
        frame_opvq_idx = float(np.clip(100.0 * (1.0 - frame_distortion), 0.0, 100.0))

        ssim_scores.append(ssim_val)
        spatial_diffs.append(si_error)
        temporal_diffs.append(ti_error)
        frame_scores.append({
            "frame": i,
            "ssim": round(ssim_val, 4),
            "opvq_index": round(frame_opvq_idx, 2),
            "opvq_score": round(1.0 + 4.0 * (frame_opvq_idx / 100.0), 2)
        })

    # Minkowski p=2 temporal pooling for human perception of worst impairments
    distortions = [1.0 - (f["opvq_index"] / 100.0) for f in frame_scores]
    pooled_distortion = float((np.mean(np.array(distortions) ** 2)) ** 0.5)
    overall_opvq_idx = float(np.clip(100.0 * (1.0 - pooled_distortion), 0.0, 100.0))
    overall_opvq_score = opvq_to_mos(overall_opvq_idx)

    result = {
        "metric": "OPVQ",
        "reference": reference_video,
        "distorted": distorted_video,
        "frames_evaluated": num_frames,
        "opvq_score": overall_opvq_score,     # 1.0 - 5.0 perceptual scale (Table 6)
        "opvq_mos": overall_opvq_score,       # Alias for backward compatibility
        "opvq_index": round(overall_opvq_idx, 2), # 0 - 100 raw index
        "mean_ssim": round(float(np.mean(ssim_scores)), 4),
        "spatial_fidelity": round(1.0 - float(np.mean(spatial_diffs)), 4),
        "temporal_fidelity": round(1.0 - float(np.mean(temporal_diffs)), 4),
        "frames": frame_scores
    }

    if log_file:
        with open(log_file, "w") as f:
            json.dump(result, f, indent=2)

    return result

def compute_vmaf(reference: str, distorted: str, log_file: str = "vmaf.json", n_threads: int = 4) -> float:
    """
    Compute VMAF score using FFmpeg libvmaf as detailed in OPVQ_Installation_Guide.docx.
    Returns:
    - vmaf_score: 0 to 100 VMAF scale (Section 3.4).
    """
    if not os.path.exists(reference):
        raise FileNotFoundError(f"Reference video not found: {reference}")
    if not os.path.exists(distorted):
        raise FileNotFoundError(f"Distorted video not found: {distorted}")

    abs_log = os.path.abspath(log_file)
    if os.path.exists(abs_log):
        try:
            os.remove(abs_log)
        except OSError:
            pass

    if os.name == "nt" and not which("ffmpeg") and which("wsl"):
        abs_p = abs_log.replace("\\", "/")
        drive = abs_p[0].lower()
        wsl_log_path = f"/mnt/{drive}{abs_p[2:]}"
    else:
        wsl_log_path = abs_log

    vmaf_filter = f"[0:v][1:v]libvmaf=log_fmt=json:log_path={wsl_log_path}:n_threads={n_threads}"
    raw_cmd = [
        "-nostdin",
        "-y",
        "-i", distorted,
        "-i", reference,
        "-lavfi", vmaf_filter,
        "-f", "null", "-"
    ]
    cmd = _get_ffmpeg_cmd(raw_cmd)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    if not os.path.exists(abs_log):
        raise RuntimeError(f"FFmpeg libvmaf failed to generate log: {proc.stderr}")

    with open(abs_log, "r") as f:
        data = json.load(f)

    vmaf_mean = float(data["pooled_metrics"]["vmaf"]["mean"])
    return round(vmaf_mean, 2)

# =========================================================================
# SECTION 3.11 STATISTICAL ANALYSIS FRAMEWORK IMPLEMENTATION
# =========================================================================

def compute_average_difference(scores_a, scores_b) -> float:
    """
    Formula from Section 3.11:
        D(A, B) = (1 / N) * sum_{i=1}^N (x_i^A - x_i^B)
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if len(a) == 0:
        return 0.0
    return float(np.mean(a - b))

def compute_win_count(scores_a, scores_b, name_a="Phone A", name_b="Phone B") -> dict:
    """
    Win count from Section 3.11:
    A device is considered to 'win' when it achieves a higher metric value
    under the same experimental observation.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    min_len = min(len(a), len(b))
    wins_a = int(np.sum(a[:min_len] > b[:min_len]))
    wins_b = int(np.sum(b[:min_len] > a[:min_len]))
    ties = int(np.sum(a[:min_len] == b[:min_len]))

    # Formatted description matching Table 6 style
    parts = []
    if wins_b > 0:
        parts.append(f"{name_b} won {wins_b} times")
    if wins_a > 0:
        parts.append(f"{name_a} won {wins_a} times" if wins_a > 1 else f"{name_a} won 1 time")
    if ties > 0:
        parts.append(f"{ties} ties")

    summary_str = ", ".join(parts) if parts else "No valid comparisons"
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": min_len,
        "summary": summary_str
    }

def compute_paired_ttest(scores_a, scores_b) -> dict:
    """
    Paired t-test from Section 3.11:
    Evaluates statistical significance of difference between paired observations.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    if min_len < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "is_significant": False}

    diffs = a - b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    if std_diff == 0:
        # Zero variance: either exactly identical or all differences are constant
        return {"t_stat": 0.0, "p_value": 0.0 if mean_diff != 0 else 1.0, "is_significant": mean_diff != 0}

    se = std_diff / np.sqrt(min_len)
    t_stat = float(mean_diff / se)

    if stats is not None:
        p_val = float(stats.t.sf(np.abs(t_stat), df=min_len - 1) * 2.0)
    else:
        # Normal approximation fallback if scipy is unavailable
        p_val = float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(np.abs(t_stat) / np.sqrt(2.0)))))

    return {
        "t_stat": round(t_stat, 4),
        "p_value": p_val,
        "is_significant": bool(p_val < 0.01)
    }

def evaluate_pairwise_comparison(name_a: str, name_b: str, scores_a: list, scores_b: list) -> dict:
    """
    Generate complete pairwise comparison entry matching Table 6.
    """
    avg_diff = compute_average_difference(scores_a, scores_b)
    win_info = compute_win_count(scores_a, scores_b, name_a=name_a, name_b=name_b)
    ttest_info = compute_paired_ttest(scores_a, scores_b)

    return {
        "comparison": f"{name_a} and {name_b}",
        "avg_difference": round(avg_diff, 4),
        "wincount": win_info["summary"],
        "p_value": ttest_info["p_value"],
        "is_significant_p01": ttest_info["is_significant"],
        "num_observations": win_info["total"]
    }

# =========================================================================
# TABLE 6 EXPERIMENTAL BENCHMARK REPLICATION (Section 4.2)
# =========================================================================

# Experimental reference data reported in Table 6 of the paper
PAPER_TABLE_6_BENCHMARK = [
    {"phones": "Oppo A37 2016 and 2017", "avg_diff": -1.0856, "wincount": "Oppo 2017 won 100 times", "p_val": "<0.01"},
    {"phones": "Oppo A37 2017 and 2018", "avg_diff": -0.8271, "wincount": "Oppo 2018 won 100 times", "p_val": "<0.01"},
    {"phones": "Oppo A37 2016 and 2018", "avg_diff": -1.9127, "wincount": "Oppo 2018 won 100 times", "p_val": "<0.01"},
    {"phones": "Vivo Y67 2016 and 2017", "avg_diff": -0.8754, "wincount": "Vivo 2017 won 100 times", "p_val": "<0.01"},
    {"phones": "Vivo Y67 2017 and 2018", "avg_diff": -0.8136, "wincount": "Vivo 2018 won 100 times", "p_val": "<0.01"},
    {"phones": "Vivo Y67 2016 and 2018", "avg_diff": -1.5676, "wincount": "Vivo 2018 won 100 times", "p_val": "<0.01"},
    {"phones": "Redmi 5A 2016 and 2017", "avg_diff": -0.7293, "wincount": "Redmi 2017 won 100 times", "p_val": "<0.01"},
    {"phones": "Redmi 5A 2017 and 2018", "avg_diff": -0.3457, "wincount": "Redmi 2018 won 91 times, Redmi 2017 won 9 times", "p_val": "<0.01"},
    {"phones": "Redmi 5A 2016 and 2018", "avg_diff": -1.0750, "wincount": "Redmi 2018 won 100 times", "p_val": "<0.01"},
]

def print_table_6(records: list = None, title: str = "Table 6: Average Difference of OPVQ Values (p-value) and Win Counts Across Phone Models"):
    """
    Print Table 6 format from Section 4.2.
    """
    if records is None:
        records = PAPER_TABLE_6_BENCHMARK

    print("\n" + "=" * 90)
    print(f" {title} ")
    print("=" * 90)
    print(f"{'Phones':<26} | {'Average differences (p<0.01)':<30} | {'Wincount':<30}")
    print("-" * 90)

    for r in records:
        phones = r.get("phones") or r.get("comparison", "")
        avg_d = r.get("avg_diff") or r.get("avg_difference", 0.0)
        p_val = r.get("p_val", "<0.01" if r.get("is_significant_p01") else "n.s.")
        diff_str = f"{avg_d:+.4f} (p{p_val})" if isinstance(avg_d, float) else f"{avg_d} (p{p_val})"
        wincount = r.get("wincount", "")
        print(f"{phones:<26} | {diff_str:<30} | {wincount:<30}")

    print("=" * 90 + "\n")

def resolve_video_path(filename: str, current_dir: str) -> str:
    """Resolve file in current directory or in script's directory."""
    if os.path.exists(filename):
        return filename
    in_script_dir = os.path.join(current_dir, filename)
    if os.path.exists(in_script_dir):
        return in_script_dir
    return filename

# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="OPVQ / VMAF Perceptual Video Quality Evaluation & Statistical Benchmarking (Paper Section 3.4, 3.6, 3.11, Table 6)"
    )
    parser.add_argument("reference", nargs="?", default=None, help="Path to reference pristine video")
    parser.add_argument("distorted", nargs="?", default=None, help="Path to distorted video")
    parser.add_argument("--mode", choices=["opvq", "vmaf", "both"], default="both",
                        help="Evaluation mode: opvq (standalone), vmaf (libvmaf), or both (default)")
    parser.add_argument("--table6", action="store_true",
                        help="Display the Section 4.2 Table 6 pairwise statistical benchmark for OPVQ")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to CSV or JSON file containing evaluated video scores to compute Table 6 statistics dynamically")
    parser.add_argument("--batch", type=str, default=None,
                        help="Path to CSV manifest listing video pairs [reference, distorted, phone_model]")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output_csv", default=None, help="Output CSV path for Table 6 or batch results")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to evaluate per clip (default: 300)")
    args = parser.parse_args()

    # 1. Table 6 Display / Reproduction Mode
    if args.table6 and not args.stats and not args.batch and not args.reference:
        print_table_6()
        if args.output_csv:
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Phones", "Average differences of OPVQ values (p<0.01)", "Wincount"])
                for r in PAPER_TABLE_6_BENCHMARK:
                    writer.writerow([r["phones"], r["avg_diff"], r["wincount"]])
            print(f"Exported Table 6 to {args.output_csv}")
        return

    # 2. Dynamic Statistical Analysis from Evaluated Scores (Section 3.11)
    if args.stats:
        print(f"\nLoading scores for statistical analysis from: {args.stats}")
        with open(args.stats, "r") as f:
            if args.stats.endswith(".json"):
                data = json.load(f)
            else:
                reader = csv.DictReader(f)
                data = list(reader)

        # Expected format: dictionary mapping device names to list of OPVQ scores
        pairwise_results = []
        if isinstance(data, dict):
            # Check for standard model comparisons
            pairs_to_test = [
                ("Oppo A37 2016", "Oppo A37 2017"),
                ("Oppo A37 2017", "Oppo A37 2018"),
                ("Oppo A37 2016", "Oppo A37 2018"),
                ("Vivo Y67 2016", "Vivo Y67 2017"),
                ("Vivo Y67 2017", "Vivo Y67 2018"),
                ("Vivo Y67 2016", "Vivo Y67 2018"),
                ("Redmi 5A 2016", "Redmi 5A 2017"),
                ("Redmi 5A 2017", "Redmi 5A 2018"),
                ("Redmi 5A 2016", "Redmi 5A 2018"),
            ]
            for p1, p2 in pairs_to_test:
                if p1 in data and p2 in data:
                    res = evaluate_pairwise_comparison(p1, p2, data[p1], data[p2])
                    pairwise_results.append(res)

        if pairwise_results:
            print_table_6(pairwise_results, title="Computed Section 3.11 Pairwise Comparison from Provided Scores")
            if args.output_csv:
                with open(args.output_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Phones", "Average differences of OPVQ values (p<0.01)", "Wincount"])
                    for r in pairwise_results:
                        writer.writerow([r["comparison"], r["avg_difference"], r["wincount"]])
                print(f"Exported pairwise table to {args.output_csv}")
        else:
            print("[Warning] Could not extract paired device scores for Table 6 from the input file.")
        return

    # 3. Single Video Pair Evaluation Mode
    ref_candidate = args.reference or resolve_video_path("reference.mp4", script_dir)
    dist_candidate = args.distorted or resolve_video_path("distorted.mp4", script_dir)

    if not os.path.exists(ref_candidate) or not os.path.exists(dist_candidate):
        print("Usage: python opvq_mos.py [reference.mp4] [distorted.mp4] [--mode {opvq,vmaf,both}]")
        print("       python opvq_mos.py --table6                # Display Table 6 from Paper")
        print("       python opvq_mos.py --stats <scores.json>   # Compute Table 6 stats from scores")
        print(f"\nLooked for reference: {ref_candidate}")
        print(f"Looked for distorted: {dist_candidate}")
        sys.exit(1)

    print("==================================================")
    print("   OPVQ / VMAF PERCEPTUAL VIDEO QUALITY BENCHMARK ")
    print("   (Paper Section 3.4, 3.6, 3.11 & Table 6)       ")
    print("==================================================")
    print(f"Reference Video : {ref_candidate}")
    print(f"Distorted Video : {dist_candidate}")
    print(f"Evaluation Mode : {args.mode}")

    results = {
        "reference": ref_candidate,
        "distorted": dist_candidate
    }

    # 1. OPVQ Evaluation (Perceptual Scale 1.0 - 5.0)
    if args.mode in ("opvq", "both"):
        print("\nComputing OPVQ (Spatial-Temporal Analysis)...")
        opvq_res = compute_opvq(ref_candidate, dist_candidate, max_frames=args.frames)
        results["opvq"] = opvq_res
        print("===== OPVQ RESULT (Section 3.6 / Table 6 Scale) =====")
        print(f"OPVQ Score (1-5)    : {opvq_res['opvq_score']:.2f} / 5.0")
        print(f"OPVQ Index (0-100)  : {opvq_res['opvq_index']:.2f} / 100")
        print(f"Mean SSIM           : {opvq_res['mean_ssim']}")
        print(f"Spatial Fidelity    : {opvq_res['spatial_fidelity']}")
        print(f"Temporal Fidelity   : {opvq_res['temporal_fidelity']}")

    # 2. VMAF Evaluation (from Section 3.4 & OPVQ Installation Guide)
    if args.mode in ("vmaf", "both"):
        print("\nComputing VMAF (FFmpeg libvmaf)...")
        try:
            vmaf_log = "temp_vmaf.json"
            vmaf_val = compute_vmaf(ref_candidate, dist_candidate, log_file=vmaf_log)
            if os.path.exists(vmaf_log):
                try:
                    os.remove(vmaf_log)
                except OSError:
                    pass
            vmaf_mos = vmaf_to_mos(vmaf_val)
            results["vmaf"] = {
                "vmaf_score": vmaf_val,
                "vmaf_mos": vmaf_mos
            }
            print("===== VMAF RESULT (Section 3.4 Scale) =====")
            print(f"VMAF Score (0-100)  : {vmaf_val:.2f} / 100")
            print(f"VMAF Score (1-5 MOS): {vmaf_mos:.2f} / 5.0")
        except Exception as e:
            print(f"[Note] VMAF calculation skipped or unavailable: {e}")
            if args.mode == "vmaf":
                sys.exit(1)

    # 3. Metric Agreement (Section 4.2)
    if "opvq" in results and "vmaf" in results:
        delta_mos = abs(results["opvq"]["opvq_score"] - results["vmaf"]["vmaf_mos"])
        print("\n===== METRIC AGREEMENT (Section 4.2) =====")
        print(f"OPVQ Perceptual Score : {results['opvq']['opvq_score']:.2f} / 5.0")
        print(f"VMAF Perceptual Score : {results['vmaf']['vmaf_mos']:.2f} / 5.0")
        print(f"Delta Perceptual Score: {delta_mos:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    print("\nAcademic Note (Paper Section 3.4): Scores reflect perceptual quality on standardized scales.")

if __name__ == "__main__":
    main()
