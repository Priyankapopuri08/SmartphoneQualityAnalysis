"""
Branch 03: IL-NIQE Standalone Execution
Computes Integrated Local Natural Image Quality Evaluator score on target image.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from il_niqe import compute_raw_il_niqe
from logistic_mapping import map_score

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../sample_images/low_light_2018.jpg")
    
    if os.path.exists(target):
        raw_score = compute_raw_il_niqe(target)
        mapped_score = map_score(raw_score, metric_name="IL-NIQE")
        print("=" * 45)
        print(" IL-NIQE IMAGE QUALITY EVALUATOR")
        print("=" * 45)
        print(f"Target Image      : {os.path.basename(target)}")
        print(f"Raw IL-NIQE Score : {raw_score:.4f} (Lower = Better)")
        print(f"Mapped Score Q(x) : {mapped_score:.2f} / 5.0 (Higher = Better)")
        print("=" * 45)
    else:
        print(f"Target not found: {target}. Running dummy check.")
        import numpy as np
        dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        raw = compute_raw_il_niqe(dummy)
        print(f"Raw IL-NIQE on synthetic test: {raw:.4f}")

if __name__ == "__main__":
    main()
