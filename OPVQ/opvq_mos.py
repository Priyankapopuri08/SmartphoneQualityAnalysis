"""
OPVQ Video Quality Analysis
Full-Reference Perceptual Video Quality Evaluation using OpenVQ.

Usage:
  python opvq_mos.py
  python opvq_mos.py reference.mp4 distorted.mp4
  python opvq_mos.py --src reference.mp4 --pvs distorted.mp4
"""

import os
import sys
import argparse

# Add OPVQ folder to sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import openvq
from openvq import compute_opvq


def evaluate_video_pair(reference_video: str, test_video: str, max_frames: int = 300) -> dict:
    """
    Run full-reference OPVQ evaluation on a video pair using the OpenVQ library.
    """
    if not os.path.exists(reference_video):
        raise FileNotFoundError(f"Reference video not found: {reference_video}")
    if not os.path.exists(test_video):
        raise FileNotFoundError(f"Test/distorted video not found: {test_video}")

    return compute_opvq(
        reference_video=reference_video,
        distorted_video=test_video,
        max_frames=max_frames
    )


def main():
    parser = argparse.ArgumentParser(description="OPVQ Perceptual Video Quality Evaluation")
    parser.add_argument("subcmd_or_ref", nargs="?", default=None, help="Reference video or subcommand")
    parser.add_argument("distorted", nargs="?", default=None, help="Distorted video")
    parser.add_argument("--src", type=str, default=None, help="Path to original/reference video")
    parser.add_argument("--pvs", type=str, default=None, help="Path to processed/phone video")
    parser.add_argument("--ref", type=str, default=None, help="Path to reference video")
    parser.add_argument("--dist", type=str, default=None, help="Path to test video")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to evaluate (default: 300)")
    args = parser.parse_args()

    ref_val = args.src or args.ref or (args.subcmd_or_ref if args.subcmd_or_ref != "opvq" else None)
    dist_val = args.pvs or args.dist or args.distorted

    # Fallback to local default files if not specified
    if not ref_val and not dist_val:
        default_ref = os.path.join(_current_dir, "reference.mp4")
        default_dist = os.path.join(_current_dir, "distorted.mp4")
        if os.path.exists(default_ref) and os.path.exists(default_dist):
            ref_val = default_ref
            dist_val = default_dist

    if not ref_val or not dist_val:
        print("Usage:")
        print("  python opvq_mos.py [reference.mp4] [distorted.mp4]")
        print("  python opvq_mos.py --src reference.mp4 --pvs distorted.mp4")
        sys.exit(1)

    res = evaluate_video_pair(ref_val, dist_val, max_frames=args.frames)

    print("===== OPVQ RESULT =====")
    print(f"OPVQ Score (Scale 1.0 - 5.0) : {res['opvq_score']:.2f} / 5.0")
    print(f"Mean SSIM                    : {res['mean_ssim']:.4f}")
    if "mean_psnr" in res:
        print(f"Mean PSNR (dB)               : {res['mean_psnr']:.2f}")
    print(f"Frames Evaluated             : {res['frames_evaluated']}")


if __name__ == "__main__":
    main()
