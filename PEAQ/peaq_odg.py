import numpy as np
import soundfile as sf
from scipy.signal import stft, resample

def peaq_basic_odg(ref_file, deg_file):
    """
    Compute a basic PEAQ-like ODG score between reference and degraded audio.

    Parameters:
        ref_file (str): Path to reference WAV file
        deg_file (str): Path to degraded WAV file

    Returns:
        float: ODG score (-4.0 = very annoying, 0.0 = imperceptible)
    """
    # Load audio
    ref, fs1 = sf.read(ref_file)
    deg, fs2 = sf.read(deg_file)

    # Convert to mono if needed
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if deg.ndim > 1:
        deg = deg.mean(axis=1)

    # Resample degraded if sample rates differ
    if fs1 != fs2:
        print(f"Resampling degraded audio from {fs2} Hz → {fs1} Hz")
        num_samples = int(len(deg) * fs1 / fs2)
        deg = resample(deg, num_samples)
        fs2 = fs1

    # Trim to shortest length
    min_len = min(len(ref), len(deg))
    if len(ref) != len(deg):
        print(f"Trimming audio to {min_len} samples to match length")
        ref = ref[:min_len]
        deg = deg[:min_len]

    # Short-time Fourier transform
    _, _, Z_ref = stft(ref, fs1, nperseg=2048)
    _, _, Z_deg = stft(deg, fs1, nperseg=2048)

    # Magnitude spectra
    mag_ref = np.abs(Z_ref)
    mag_deg = np.abs(Z_deg)

    # Log-spectral distance
    eps = 1e-10
    lsd = np.mean((np.log10(mag_ref + eps) - np.log10(mag_deg + eps)) ** 2)


    # Empirical mapping to ODG (-4 to 0)
    odg = -1.5 * np.sqrt(lsd)
    odg = np.clip(odg, -4.0, 0.0)

    return odg

if __name__ == "__main__":
    import sys

    # Command-line arguments support
    if len(sys.argv) == 3:
        ref = sys.argv[1]
        deg = sys.argv[2]
    else:
        # Default filenames
        ref = "reference.wav"
        deg = "degraded.wav"

    try:
        score = peaq_basic_odg(ref, deg)
        print(f"PEAQ ODG Score: {score:.2f}")
    except Exception as e:
        print("Error computing ODG score:", e)