import os
import csv
from pathlib import Path

import numpy as np
import essentia.standard as es
import librosa


# -------------------------------------------------------------------
# Konfigurace
# -------------------------------------------------------------------

MUSIC_DIR = Path("music")
OUTPUT_CSV = Path("analysis.csv")

AUDIO_EXT = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".aiff", ".alac"
}

SAMPLE_RATE = 44100


# -------------------------------------------------------------------
# Pomocné DSP funkce
# -------------------------------------------------------------------

def spectral_features(y, sr):
    """Výpočet RMS, centroid, flux, HF ratio, bass ratio, attack sharpness."""
    
    # STFT
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    flux_mean = float(np.mean(flux))
    flux_std = float(np.std(flux))

    # HPSS
    harmonic, percussive = librosa.effects.hpss(y)
    percussive_ratio = float(
        np.sum(percussive**2) / (np.sum(y**2) + 1e-9)
    )

    # Onset density
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration_sec = len(y) / sr
    onset_rate = float(len(onset_frames) / (duration_sec + 1e-9))

    # High-frequency ratio (>= 6 kHz)
    hf_mask = freqs >= 6000
    hf_energy = float(np.sum(S[hf_mask]))
    total_energy = float(np.sum(S))
    hf_ratio = float(hf_energy / (total_energy + 1e-9))

    # Bass ratio (< 200 Hz)
    bass_mask = freqs < 200
    bass_energy = float(np.sum(S[bass_mask]))
    bass_ratio = float(bass_energy / (total_energy + 1e-9))

    # Attack sharpness (derivace onset_strength)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) > 1:
        attack_sharpness = float(np.max(np.diff(onset_env)))
    else:
        attack_sharpness = 0.0

    return {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "flux_mean": flux_mean,
        "flux_std": flux_std,
        "percussive_ratio": percussive_ratio,
        "onset_rate": onset_rate,
        "hf_ratio": hf_ratio,
        "bass_ratio": bass_ratio,
        "attack_sharpness": attack_sharpness,
    }


# -------------------------------------------------------------------
# Analýza jednoho tracku
# -------------------------------------------------------------------

def analyze_track(path: Path):
    try:
        # Essentia loader → resample + mono
        audio = es.MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE)()

        # Key
        key_extractor = es.KeyExtractor()
        key, mode, key_strength = key_extractor(audio)

        # BPM
        rhythm = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, bpm_conf, _, _ = rhythm(audio)

        # Librosa features
        y = np.array(audio, dtype=np.float32)
        feats = spectral_features(y, SAMPLE_RATE)

        return {
            "filename": path.name,
            "key": key,
            "mode": mode,
            "key_strength": round(float(key_strength), 3),
            "bpm": round(float(bpm), 2),
            "bpm_confidence": round(float(bpm_conf), 3),
            **feats
        }

    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return None


# -------------------------------------------------------------------
# Hlavní část: sken + export
# -------------------------------------------------------------------

def main():
    if not MUSIC_DIR.exists():
        print(f"[ERROR] Folder '{MUSIC_DIR}' not found.")
        return

    tracks = []
    for root, _, files in os.walk(MUSIC_DIR):
        for f in files:
            if Path(f).suffix.lower() in AUDIO_EXT:
                tracks.append(Path(root) / f)

    if not tracks:
        print("[INFO] No audio files found.")
        return

    print(f"[INFO] Found {len(tracks)} tracks. Analyzing...")

    results = []
    for track in tracks:
        res = analyze_track(track)
        if res:
            results.append(res)

    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] Saved to {OUTPUT_CSV}")
    else:
        print("[WARN] No analyzable tracks found.")


if __name__ == "__main__":
    main()
