import os
import subprocess
from opvq_mos import compute_vmaf, vmaf_to_mos

# Generate compressed video with realistic compression distortion
ref = "reference.mp4"
comp_output = "distorted_compressed.mp4"

cmd = [
    "ffmpeg", "-y", "-i", ref,
    "-c:v", "libx264", "-crf", "35", "-preset", "veryfast",
    comp_output
]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# Evaluate VMAF and MOS
vmaf_score = compute_vmaf(ref, comp_output, log_file="comp_vmaf.json")
mos_score = vmaf_to_mos(vmaf_score)

print("===== COMPRESSED VIDEO BENCHMARK RESULT =====")
print(f"Reference Video  : {ref}")
print(f"Degraded Video   : {comp_output}")
print(f"VMAF Score       : {vmaf_score:.2f} / 100")
print(f"Predicted MOS    : {mos_score} / 5.0")
