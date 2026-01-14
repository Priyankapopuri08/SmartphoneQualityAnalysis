import numpy as np
from scipy.io import wavfile
import soundfile as sf
import librosa
from pesq import pesq

def convert_to_16k(input_path, output_path):
    """Convert any WAV file to 16 kHz mono."""
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    if sr != 16000:
        print(f"Resampling {input_path} from {sr} Hz to 16000 Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(output_path, audio, 16000)
    return output_path

# --- 1. Load Reference and Degraded Audio Files ---
try:
    # Convert both files to 16 kHz mono before PESQ
    ref_path = convert_to_16k("sample3.wav", "sample3_16k.wav")
    deg_path = convert_to_16k("sample30.wav", "sample30_16k.wav")

    # Load converted files
    rate, ref = wavfile.read(ref_path)
    rate, deg = wavfile.read(deg_path)

except FileNotFoundError:
    print("Audio files not found. Please place wav files in directory, or update file paths.")
    exit()

# --- 2. Calculate PESQ Scores ---
print("Sample rate detected:", rate)

# Wideband PESQ (requires 16 kHz)
if rate == 16000:
    print("Entering wideband PESQ block...")
    try:
        pesq_score_wb = pesq(rate, ref, deg, 'wb')
        print(f"PESQ Wideband (wb) score: {pesq_score_wb}")
    except Exception as e:
        print(f"Error in wideband PESQ calculation: {e}")

# Narrowband PESQ (supports 8 kHz or 16 kHz)
if rate in (8000, 16000):
    print("Entering narrowband PESQ block...")
    try:
        pesq_score_nb = pesq(rate, ref, deg, 'nb')
        print(f"PESQ Narrowband (nb) score: {pesq_score_nb}")
    except Exception as e:
        print(f"Error in narrowband PESQ calculation: {e}")