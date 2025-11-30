#!/usr/bin/env python3
"""
Analyze tracks from DJ playlist history CSV
============================================
Allows selection of specific range of tracks from a playlist CSV
Preserves playlist metadata (#, Title, Artist, Album) in output
Uses file paths from CSV instead of scanning music folder
"""

import csv
from pathlib import Path

import numpy as np
import essentia.standard as es
import librosa


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SAMPLE_RATE = 44100


# -------------------------------------------------------------------
# DSP Feature Extraction
# -------------------------------------------------------------------

def spectral_features(y, sr):
    """Compute RMS, centroid, flux, HF ratio, bass ratio, attack sharpness."""
    
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

    # Attack sharpness (derivative of onset_strength)
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
# Track Analysis
# -------------------------------------------------------------------

def analyze_track(file_path: str, track_number: int, title: str, artist: str, album: str):
    """
    Analyze a single track and return results with playlist metadata.
    
    Args:
        file_path: Path to audio file (from CSV)
        track_number: Track position in playlist
        title: Track title
        artist: Artist name
        album: Album name
    
    Returns:
        Dictionary with playlist metadata + MIR features, or None on error
    """
    # Handle special paths (history cache, etc)
    if "#history#" in file_path or not Path(file_path).exists():
        print(f"[SKIP] Track #{track_number}: File not accessible - {title}")
        return None
    
    try:
        # Essentia loader → resample + mono
        audio = es.MonoLoader(filename=file_path, sampleRate=SAMPLE_RATE)()

        # Key detection
        key_extractor = es.KeyExtractor()
        key, mode, key_strength = key_extractor(audio)

        # BPM detection
        rhythm = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, bpm_conf, _, _ = rhythm(audio)

        # Librosa features
        y = np.array(audio, dtype=np.float32)
        feats = spectral_features(y, SAMPLE_RATE)

        return {
            "#": track_number,
            "Title": title,
            "Artist": artist,
            "Album": album,
            "key": key,
            "mode": mode,
            "key_strength": round(float(key_strength), 3),
            "bpm": round(float(bpm), 2),
            "bpm_confidence": round(float(bpm_conf), 3),
            **feats
        }

    except Exception as e:
        print(f"[ERROR] Track #{track_number} ({title}): {e}")
        return None


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DJ PLAYLIST ANALYSIS")
    print("=" * 70)
    
    # 1. Select CSV file
    print("\nAvailable CSV files:")
    csv_files = list(Path(".").glob("*.csv"))
    
    if not csv_files:
        print("[ERROR] No CSV files found in current directory.")
        return
    
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"  {idx}. {csv_file.name}")
    
    while True:
        try:
            csv_choice = int(input(f"\nSelect CSV file (1-{len(csv_files)}): "))
            if 1 <= csv_choice <= len(csv_files):
                selected_csv = csv_files[csv_choice - 1]
                break
            print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\n✓ Selected: {selected_csv.name}")
    
    # 2. Read CSV and count tracks
    with open(selected_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_tracks = list(reader)
    
    total_tracks = len(all_tracks)
    print(f"✓ Found {total_tracks} tracks in playlist")
    
    # 3. Select track range
    print(f"\n{'─' * 70}")
    print("SELECT TRACK RANGE")
    print(f"{'─' * 70}")
    
    while True:
        try:
            start = int(input(f"Start track # (1-{total_tracks}): "))
            if 1 <= start <= total_tracks:
                break
            print(f"Please enter a number between 1 and {total_tracks}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    while True:
        try:
            end = int(input(f"End track # ({start}-{total_tracks}): "))
            if start <= end <= total_tracks:
                break
            print(f"Please enter a number between {start} and {total_tracks}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # 4. Filter selected tracks
    selected_tracks = [t for t in all_tracks if start <= int(t['#']) <= end]
    
    print(f"\n✓ Analyzing tracks #{start} to #{end} ({len(selected_tracks)} tracks)")
    print(f"{'─' * 70}\n")
    
    # 5. Analyze tracks
    results = []
    for track_data in selected_tracks:
        track_num = int(track_data['#'])
        title = track_data['Title']
        artist = track_data['Artist']
        album = track_data['Album']
        file_path = track_data['File name']
        
        print(f"[{track_num}/{end}] Analyzing: {artist} - {title}")
        
        result = analyze_track(file_path, track_num, title, artist, album)
        if result:
            results.append(result)
    
    # 6. Save results
    if results:
        output_name = f"analysis_{selected_csv.stem}_tracks_{start}-{end}.csv"
        output_path = Path(output_name)
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'=' * 70}")
        print(f"✓ Analysis complete!")
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Successfully analyzed: {len(results)}/{len(selected_tracks)} tracks")
        print(f"{'=' * 70}")
    else:
        print("\n[WARN] No tracks were successfully analyzed.")


if __name__ == "__main__":
    main()
