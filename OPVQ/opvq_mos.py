import subprocess
import json
import os

def compute_vmaf(reference, distorted, log_file="vmaf.json"):
    """
    Compute VMAF score using FFmpeg libvmaf
    """
    cmd = [
        "ffmpeg",
        "-i", reference,
        "-i", distorted,
        "-lavfi",
        f"[0:v][1:v]libvmaf=log_fmt=json:log_path={log_file}",
        "-f", "null", "-"
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    with open(log_file, "r") as f:
        data = json.load(f)

    vmaf_mean = data["pooled_metrics"]["vmaf"]["mean"]
    return vmaf_mean


def vmaf_to_mos(vmaf):
    """
    Linear MOS prediction from VMAF
    MOS range: 1–5
    """
    mos = 1 + 4 * (vmaf / 100)
    return round(mos, 2)


if __name__ == "__main__":
    reference_video = "reference.mp4"
    distorted_video = "distorted.mp4"

    if not os.path.exists(reference_video) or not os.path.exists(distorted_video):
        raise FileNotFoundError("Reference or distorted video not found")

    vmaf_score = compute_vmaf(reference_video, distorted_video)
    mos_score = vmaf_to_mos(vmaf_score)

    print("===== OPVQ RESULT =====")
    print(f"VMAF Score       : {vmaf_score:.2f}")
    print(f"Predicted MOS    : {mos_score} / 5")