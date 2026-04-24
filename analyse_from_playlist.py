#!/usr/bin/env python3
"""
DJ Playlist Analysis - Parallel Processing Version
===================================================
Uses ProcessPoolExecutor for parallel track analysis on multi-core CPUs
"""

import csv
import multiprocessing
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import essentia.standard as es
import librosa


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SAMPLE_RATE = 44100
NUM_WORKERS = min(4, multiprocessing.cpu_count())  # Limit to 4 workers to avoid memory issues


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
        return {"error": "skip", "track_num": track_number, "title": title}

    try:
        # Essentia loader → resample + mono
        audio = es.MonoLoader(filename=str(file_path), sampleRate=SAMPLE_RATE)()

        # Key detection
        key_extractor = es.KeyExtractor()
        key, mode, key_strength = key_extractor(audio)

        # BPM detection with retry on buffer error
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rhythm = es.RhythmExtractor2013(method="multifeature")
                bpm, beats, bpm_conf, _, _ = rhythm(audio)
                break
            except RuntimeError as e:
                if "buffer is full" in str(e) and attempt < max_retries - 1:
                    continue  # Retry
                raise

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
        return {"error": str(e), "track_num": track_number, "title": title}


# -------------------------------------------------------------------
# Helper Functions for Incremental Analysis
# -------------------------------------------------------------------

def find_existing_analysis(csv_path: str | Path) -> Path | None:
    """Find existing analysis file for a given playlist CSV."""
    analysed_dir = Path("analysed_playlists")
    if not analysed_dir.exists():
        return None

    # Extract stem from path or string
    csv_name = Path(csv_path).stem
    pattern = f"analysis_{csv_name}_tracks_*.csv"
    existing = list(analysed_dir.glob(pattern))

    if existing:
        # Return the most recently modified one
        return max(existing, key=lambda p: p.stat().st_mtime)
    return None


def load_existing_analysis(path: Path | None) -> pd.DataFrame | None:
    """Load existing analysis data if file exists."""
    if path is None:
        return None
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def get_analyzed_tracks(existing_df: pd.DataFrame) -> set[tuple]:
    """Get set of (title, artist) tuples from existing analysis."""
    tracks = set()
    for _, row in existing_df.iterrows():
        title = str(row.get('Title', '')).strip().lower()
        artist = str(row.get('Artist', '')).strip().lower()
        if title and artist:
            tracks.add((title, artist))
    return tracks


def merge_and_save_results(csv_path: str | Path, existing_df: pd.DataFrame, new_results: list[dict], start: int, end: int):
    """Merge existing and new results, save to file."""
    new_df = pd.DataFrame(new_results)
    csv_stem = Path(csv_path).stem

    if existing_df is not None and not existing_df.empty:
        # Combine and remove duplicates (keep first occurrence)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Title', 'Artist'], keep='first')

        # Sort by track number
        combined['#'] = pd.to_numeric(combined['#'], errors='coerce').fillna(0).astype(int)
        combined = combined.sort_values('#').reset_index(drop=True)

        max_track = combined['#'].max()
        output_name = f"analysis_{csv_stem}_tracks_1-{max_track}.csv"
    else:
        combined = new_df
        output_name = f"analysis_{csv_stem}_tracks_{start}-{end}.csv"

    output_path = Path("analysed_playlists") / output_name

    # Remove old individual files if merging
    if existing_df is not None and not existing_df.empty:
        old_files = list(Path("analysed_playlists").glob(f"analysis_{csv_stem}_tracks_*.csv"))
        for old_file in old_files:
            if old_file != output_path:
                old_file.unlink()

    combined.to_csv(output_path, index=False)
    return output_path, len(combined)


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DJ PLAYLIST ANALYSIS")
    print("=" * 70)

    # 1. Select CSV file
    print("\nAvailable CSV files:")
    csv_files = list(Path("playlists_to_analyse").glob("*.csv"))

    if not csv_files:
        print("[ERROR] No CSV files found in playlists_to_analyse directory.")
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

    # 2. Check for existing analysis
    existing_analysis_path = find_existing_analysis(selected_csv.name)
    existing_df = load_existing_analysis(existing_analysis_path) if existing_analysis_path else None

    if existing_df is not None and not existing_df.empty:
        print(f"\n✓ Found existing analysis: {existing_analysis_path.name}")
        print(f"  Tracks already analyzed: {len(existing_df)}")
    else:
        existing_df = None
        print("\n[INFO] No existing analysis found - will analyze from scratch.")

    # 3. Read CSV and count tracks
    with open(selected_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_tracks = list(reader)

    total_tracks = len(all_tracks)
    print(f"✓ Found {total_tracks} tracks in source playlist")

    # 4. Get already analyzed tracks
    analyzed_tracks = get_analyzed_tracks(existing_df) if existing_df is not None else set()

    # 5. Select track range
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
        except KeyboardInterrupt:
            print("\n\nAnalysis cancelled.")
            return

    while True:
        try:
            end = int(input(f"End track # ({start}-{total_tracks}): "))
            if start <= end <= total_tracks:
                break
            print(f"Please enter a number between {start} and {total_tracks}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nAnalysis cancelled.")
            return

    # 6. Filter selected tracks and identify new ones
    selected_tracks = [t for t in all_tracks if start <= int(t['#']) <= end]

    new_tracks = []
    skipped_count = 0
    for t in selected_tracks:
        title = t.get('Title', '').strip().lower()
        artist = t.get('Artist', '').strip().lower()
        if title and artist and (title, artist) in analyzed_tracks:
            skipped_count += 1
        else:
            new_tracks.append(t)

    if skipped_count > 0:
        print(f"\n✓ Skipping {skipped_count} already-analyzed tracks")

    print(f"✓ Analyzing {len(new_tracks)} new tracks from #{start} to #{end}")
    print(f"{'─' * 70}\n")

    # 7. Analyze only new tracks (parallel)
    print(f"✓ Using {NUM_WORKERS} parallel workers\n")

    results = []
    task_args = [
        (track_data['File name'], int(track_data['#']), track_data['Title'],
         track_data['Artist'], track_data['Album'])
        for track_data in new_tracks
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_track = {executor.submit(analyze_track, *args): args[1] for args in task_args}

        for future in as_completed(future_to_track):
            track_num = future_to_track[future]
            try:
                result = future.result()
                if result and result.get('error') != 'skip':
                    results.append(result)
                    progress = len(results) / len(task_args) * 100
                    print(f"[{track_num}/{end}] {progress:.0f}% complete")
            except Exception as e:
                print(f"[ERROR] Track #{track_num}: {e}")

    # Filter out error entries
    results = [r for r in results if not r.get('error', '').startswith('In RhythmExtractor')]

    # 8. Save and merge results
    if results:
        output_path, total_count = merge_and_save_results(
            selected_csv.name, existing_df, results, start, end
        )

        print(f"\n{'=' * 70}")
        print(f"✓ Analysis complete!")
        print(f"✓ New tracks analyzed: {len(results)}")
        print(f"✓ Total tracks in merged file: {total_count}")
        print(f"✓ Results saved to: {output_path}")
        print(f"{'=' * 70}")
    elif existing_df is not None and not existing_df.empty:
        print(f"\n{'=' * 70}")
        print(f"✓ All tracks in selected range were already analyzed.")
        print(f"✓ Existing analysis: {existing_analysis_path.name} ({len(existing_df)} tracks)")
        print(f"{'=' * 70}")
    else:
        print("\n[WARN] No tracks were successfully analyzed.")


if __name__ == "__main__":
    main()
