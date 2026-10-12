"""
OPVQ (Objective Perceptual Video Quality / OpenVQ)
Full-Reference Spatial-Temporal Perceptual Video Quality Evaluation Module.

Evaluates:
1. Spatial Degradation (Structural similarity SSIM, edge energy / Spatial Information SI)
2. Temporal Degradation (Motion continuity / Temporal Information TI)
3. Minkowski-pooled OPVQ Quality Index (0–100) and Predicted OPVQ MOS (1.0–5.0)
4. Full-Reference VMAF (Video Multi-Method Assessment Fusion) & Predicted VMAF MOS (1.0–5.0)
   as described in OPVQ_Installation_Guide.docx
"""

import os
import sys
import json
import argparse
import subprocess
from shutil import which
import numpy as np

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
            # Convert paths in arguments to WSL /mnt format
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

def compute_opvq(reference_video: str, distorted_video: str, log_file: str = None, max_frames: int = 300) -> dict:
    """
    Compute standalone OPVQ full-reference perceptual quality score.
    Models human visual system sensitivity to spatial and temporal distortions.
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
        frame_opvq = float(np.clip(100.0 * (1.0 - frame_distortion), 0.0, 100.0))

        ssim_scores.append(ssim_val)
        spatial_diffs.append(si_error)
        temporal_diffs.append(ti_error)
        frame_scores.append({
            "frame": i,
            "ssim": round(ssim_val, 4),
            "opvq_index": round(frame_opvq, 2)
        })

    # Minkowski p=2 temporal pooling for human perception of worst impairments
    distortions = [1.0 - (f["opvq_index"] / 100.0) for f in frame_scores]
    pooled_distortion = float((np.mean(np.array(distortions) ** 2)) ** 0.5)
    overall_opvq = float(np.clip(100.0 * (1.0 - pooled_distortion), 0.0, 100.0))
    opvq_mos = opvq_to_mos(overall_opvq)

    result = {
        "metric": "OPVQ",
        "reference": reference_video,
        "distorted": distorted_video,
        "frames_evaluated": num_frames,
        "opvq_score": round(overall_opvq, 2),
        "opvq_mos": opvq_mos,
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

    # If running WSL bridge, ensure log file path is mapped
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

def opvq_to_mos(opvq_score: float) -> float:
    """
    Standard linear MOS prediction from OPVQ index (Range 1.0 – 5.0).
    """
    mos = 1.0 + 4.0 * (opvq_score / 100.0)
    return round(float(np.clip(mos, 1.0, 5.0)), 2)

def vmaf_to_mos(vmaf_score: float) -> float:
    """
    Standard linear MOS prediction from VMAF score (Range 1.0 – 5.0).
    """
    mos = 1.0 + 4.0 * (vmaf_score / 100.0)
    return round(float(np.clip(mos, 1.0, 5.0)), 2)

def resolve_video_path(filename: str, current_dir: str) -> str:
    """Resolve file in current directory or in script's directory."""
    if os.path.exists(filename):
        return filename
    in_script_dir = os.path.join(current_dir, filename)
    if os.path.exists(in_script_dir):
        return in_script_dir
    return filename

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="OPVQ / VMAF Perceptual Video Quality Evaluation")
    parser.add_argument("reference", nargs="?", default=None, help="Path to reference pristine video")
    parser.add_argument("distorted", nargs="?", default=None, help="Path to distorted video")
    parser.add_argument("--mode", choices=["opvq", "vmaf", "both"], default="both",
                        help="Evaluation mode: opvq (standalone), vmaf (libvmaf), or both (default)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to evaluate (default: 300)")
    args = parser.parse_args()

    # Default to reference.mp4 and distorted.mp4 in cwd or script directory
    ref_candidate = args.reference or resolve_video_path("reference.mp4", script_dir)
    dist_candidate = args.distorted or resolve_video_path("distorted.mp4", script_dir)

    if not os.path.exists(ref_candidate) or not os.path.exists(dist_candidate):
        print("Usage: python opvq_mos.py [reference.mp4] [distorted.mp4] [--mode {opvq,vmaf,both}]")
        print(f"Looked for reference: {ref_candidate}")
        print(f"Looked for distorted: {dist_candidate}")
        print("Please provide valid video file paths.")
        sys.exit(1)

    print("==================================================")
    print("   OPVQ / VMAF PERCEPTUAL VIDEO QUALITY BENCHMARK ")
    print("==================================================")
    print(f"Reference Video : {ref_candidate}")
    print(f"Distorted Video : {dist_candidate}")
    print(f"Evaluation Mode : {args.mode}")

    results = {
        "reference": ref_candidate,
        "distorted": dist_candidate
    }

    # 1. OPVQ Evaluation
    if args.mode in ("opvq", "both"):
        print("\nComputing OPVQ (Spatial-Temporal Analysis)...")
        opvq_res = compute_opvq(ref_candidate, dist_candidate, max_frames=args.frames)
        results["opvq"] = opvq_res
        print("===== OPVQ RESULT =====")
        print(f"OPVQ Index (0-100)  : {opvq_res['opvq_score']:.2f} / 100")
        print(f"Predicted MOS (1-5) : {opvq_res['opvq_mos']:.2f} / 5.0")
        print(f"Mean SSIM           : {opvq_res['mean_ssim']}")
        print(f"Spatial Fidelity    : {opvq_res['spatial_fidelity']}")
        print(f"Temporal Fidelity   : {opvq_res['temporal_fidelity']}")

    # 2. VMAF Evaluation (from OPVQ Installation Guide)
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
            print("===== VMAF RESULT (OPVQ Guide) =====")
            print(f"VMAF Score (0-100)  : {vmaf_val:.2f} / 100")
            print(f"Predicted MOS (1-5) : {vmaf_mos:.2f} / 5.0")
        except Exception as e:
            print(f"[Note] VMAF calculation skipped or unavailable: {e}")
            if args.mode == "vmaf":
                sys.exit(1)

    # 3. Correlation / Metric Agreement
    if "opvq" in results and "vmaf" in results:
        delta_mos = abs(results["opvq"]["opvq_mos"] - results["vmaf"]["vmaf_mos"])
        print("\n===== METRIC AGREEMENT =====")
        print(f"OPVQ Predicted MOS  : {results['opvq']['opvq_mos']:.2f} / 5.0")
        print(f"VMAF Predicted MOS  : {results['vmaf']['vmaf_mos']:.2f} / 5.0")
        print(f"Delta MOS           : {delta_mos:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    print("\nAcademic Note: Always report results as Predicted MOS or MOS-equivalent scores, not raw subjective MOS.")

if __name__ == "__main__":
    main()