"""
Branch 01: BRISQUE Standalone Execution
Computes Blind/Referenceless Image Spatial Quality Evaluator score on target image or directory.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brisque import compute_raw_brisque
from logistic_mapping import map_score

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../sample_images/Oppo_A37_2018_daylight_1.jpg")
    
    if os.path.exists(target):
        raw_score = compute_raw_brisque(target)
        mapped_score = map_score(raw_score, metric_name="BRISQUE")
        print("=" * 45)
        print(" BRISQUE IMAGE QUALITY EVALUATOR")
        print("=" * 45)
        print(f"Target Image      : {os.path.basename(target)}")
        print(f"Raw BRISQUE Score : {raw_score:.4f} (Lower = Better)")
        print(f"Mapped Score Q(x) : {mapped_score:.2f} / 5.0 (Higher = Better)")
        print("=" * 45)
    else:
        print(f"Target not found: {target}. Running dummy check.")
        import numpy as np
        dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        raw = compute_raw_brisque(dummy)
        print(f"Raw BRISQUE on synthetic test: {raw:.4f}")

if __name__ == "__main__":
    main()
