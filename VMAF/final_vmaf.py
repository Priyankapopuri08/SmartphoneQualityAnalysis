import subprocess
import json
import csv
from shutil import which

reference_video = "reference.mp4"
distorted_video = "distorted.mp4"
output_json = "vmaf_result.json"
output_csv = "vmaf_frame_scores.csv"
vmaf_model = "vmaf_v0.6.1"

# ---------------- FIND TOOLS ----------------
ffmpeg = which("ffmpeg")
ffprobe = which("ffprobe")

if not ffmpeg or not ffprobe:
    raise RuntimeError("ffmpeg/ffprobe not found in PATH")

print("Using FFmpeg:", ffmpeg)

# ---------------- GET REFERENCE RESOLUTION ----------------
probe_cmd = [
    ffprobe, "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height",
    "-of", "json",
    reference_video
]

probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
info = json.loads(probe.stdout)

width = info["streams"][0]["width"]
height = info["streams"][0]["height"]

print(f"Reference resolution: {width}x{height}")

# ---------------- RUN VMAF ----------------
cmd = [
    ffmpeg,
    "-i", distorted_video,
    "-i", reference_video,
    "-lavfi",
    (
        f"[0:v]scale={width}:{height}[dist];"
        f"[1:v]scale={width}:{height}[ref];"
        f"[dist][ref]libvmaf="
        f"model=version={vmaf_model}:"
        f"log_fmt=json:log_path={output_json}"
    ),
    "-f", "null", "-"
    "-f", "null", "-"
]

print("Running VMAF...")
subprocess.run(cmd, check=True)
print("VMAF completed")

# ---------------- PARSE RESULTS ----------------
with open(output_json) as f:
    data = json.load(f)

overall = data["pooled_metrics"]["vmaf"]["mean"]
print(f"\nOverall VMAF score: {overall:.2f}")

# ---------------- SAVE FRAME SCORES ----------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "vmaf"])
    for i, frame in enumerate(data["frames"]):
        writer.writerow([i, frame["metrics"]["vmaf"]])

print(f"Frame-level scores saved to {output_csv}")