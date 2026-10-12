import os
import sys
import subprocess
from opvq_mos import compute_opvq, compute_vmaf, opvq_to_mos, vmaf_to_mos, _get_ffmpeg_cmd, resolve_video_path

script_dir = os.path.dirname(os.path.abspath(__file__))
ref = resolve_video_path("reference.mp4", script_dir)
comp_output = os.path.join(script_dir, "distorted_compressed.mp4")
log_output = os.path.join(script_dir, "comp_opvq.json")

# Generate compressed video with realistic compression distortion if ref exists
if os.path.exists(ref):
    raw_cmd = [
        "-y", "-i", ref,
        "-c:v", "libx264", "-crf", "35", "-preset", "veryfast",
        comp_output
    ]
    cmd = _get_ffmpeg_cmd(raw_cmd)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Evaluate standalone OPVQ and MOS
    res = compute_opvq(ref, comp_output, log_file=log_output)

    print("===== COMPRESSED VIDEO OPVQ BENCHMARK RESULT =====")
    print(f"Reference Video     : {ref}")
    print(f"Degraded Video      : {comp_output}")
    print(f"OPVQ Index (0-100)  : {res['opvq_score']:.2f}")
    print(f"Predicted MOS (1-5) : {res['opvq_mos']:.2f} / 5.0")
    print(f"Mean SSIM           : {res['mean_ssim']}")
    print(f"Spatial Fidelity    : {res['spatial_fidelity']}")
    print(f"Temporal Fidelity   : {res['temporal_fidelity']}")

    # Also compute VMAF comparison if possible
    try:
        vmaf_log = os.path.join(script_dir, "comp_vmaf.json")
        vmaf_score = compute_vmaf(ref, comp_output, log_file=vmaf_log)
        vmaf_mos = vmaf_to_mos(vmaf_score)
        print("===== COMPRESSED VIDEO VMAF BENCHMARK RESULT =====")
        print(f"VMAF Score (0-100)  : {vmaf_score:.2f}")
        print(f"Predicted MOS (1-5) : {vmaf_mos:.2f} / 5.0")
        print(f"Delta MOS           : {abs(res['opvq_mos'] - vmaf_mos):.2f}")
    except Exception as e:
        print(f"[Note] VMAF comparison skipped: {e}")
else:
    print("Please place a reference.mp4 in this folder to run the benchmark.")
