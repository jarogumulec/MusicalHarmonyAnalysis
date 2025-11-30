
# Musical Harmony Analysis – README

This project performs MIR-based (Music Information Retrieval) analysis of audio tracks (MP3, M4A/ALAC, FLAC, WAV…) without relying on any machine‑learning datasets or Spotify Premium.  
Focused on DJ‑relevant signal features: rhythmic density, percussive character, tonal center, spectral balance, and transient sharpness.

---

## 🎵 **Analyzed Audio Parameters (Explained Technically)**

Below are all low‑complexity, fully deterministic DSP metrics extracted from audio in this project.  
Each metric is based on established MIR methods (Essentia, Librosa) and corresponds to DJ‑relevant perceptual qualities.

---

## **1. Key (Tónina) & Mode (Major/Minor)**  
**Method:** Essentia `KeyExtractor`  
- Detects global tonal center (e.g., F#, D, A♭).  
- Major/minor classification.  
- `key_strength` indicates model confidence.

**Usage for DJs:** harmonic mixing, Camelot mapping, color-coded grouping.

---

## **2. BPM (Tempo)**  
**Method:** Essentia `RhythmExtractor2013`  
- Robust beat detection using multi-feature estimation.  
- `bpm_confidence` for reliability.

**Usage:** verifying tempo, detecting problematic files (rips, edits).

---

## **3. RMS Energy**  
**What it represents:** raw physical energy of the waveform.

**Why it matters:** correlates with perceived punch, but does not equal loudness.  
Useful as a neutral, gain‑independent baseline.

Parameters:
- `rms_mean`  
- `rms_std`

---

## **4. Percussive Ratio (HPSS)**  
**Method:** Harmonic–Percussive Source Separation  
Measures how much of the signal consists of *transients / drum hits* vs *pads / harmonic content*.

- High → techno, electro, energetic bangers  
- Low → deep house, ambient, groove‑light tracks

Parameter:
- `percussive_ratio`

---

## **5. Onset Density (Rhythmic Density)**  
**Meaning:** counts how many transient events per second occur.

This reliably captures:
- **1/8 hats** (high onset density → energetic feel)  
- **1/4 hats** (lower density → drop in groove)

Parameter:
- `onset_rate`

---

## **6. Spectral Centroid (“Brightness”)**  
**Meaning:** where the “center of mass” of spectral energy lies.  
Bright = sizzle, fizz, highs.  
Dark = warm, muted, deep.

Parameters:
- `centroid_mean`  
- `centroid_std`

---

## **7. Spectral Flux (“Transient Activity”)**  
How much the spectrum changes frame‑to‑frame.  
Tracks with strong percussion have high flux.

Parameters:
- `flux_mean`
- `flux_std`

---

## **8. High-Frequency Ratio (>= 6 kHz)**  
A proxy for the **hi-hat / shimmer layer**.

- High → crisp top end, energetic hats  
- Low → muffled, soft, pad‑driven

Parameter:
- `hf_ratio`

---

## **9. Bass Ratio (< 200 Hz)**  
Measures low‑end dominance.

- Useful to identify tracks with strong sub/bass  
- Helps characterize balance (sub‑heavy vs mid‑heavy tracks)

Parameter:
- `bass_ratio`

---

## **10. Attack Sharpness**  
Derived from the slope of the onset envelope.  
Captures how “hard” the transient attacks are.

- High → electro/techno punchy kicks  
- Low → deep house smooth edges

Parameter:
- `attack_sharpness`

---

# 📈 Output Format  
All results are saved to:

```
analysis.csv
```

Each row = one track, columns = above features.

---

# 📂 Directory Structure

```
MusicalHarmonyAnalysis/
 ├── music/           # put audio files here
 ├── analyse_music.py
 ├── analysis.csv     # output
 ├── .venv/           # uv environment
 └── run.sh           # bootstrap / execution
```

---

# 🗂️ TODO / Optional Extensions

### **1. Camelot Key Mapping**  
Convert keys (e.g., F# minor → 11A) for harmonic mixing.

### **2. DJ Energy Index (combined metric)**  
Composite index using:
- percussive_ratio  
- onset_rate  
- hf_ratio  
- attack_sharpness  

### **3. Transition Compatibility Scoring**  
Quantify how two tracks blend based on:
- key distance  
- rhythmic density match  
- spectral balance similarity

### **4. Heatmaps & Visualization Tools**  
- brightness vs. bass ratio  
- groove density maps  
- clustering based on audio features

### **5. Spotify Audio Feature Fallback**  
If a track exists on Spotify:
- fetch ML‑derived features (danceability, valence…)  
- merge with local DSP metrics  
(No Spotify Premium required for these endpoints.)

---

# 🧪 Notes on Audio Processing

- All audio is decoded on-the-fly (MP3, M4A/ALAC, FLAC, WAV, AAC).  
- Sampling rate is automatically resampled to 44.1 kHz.  
- Bit depth does not matter (16/24/32 float → normalized internally).  
- Stereo mix → mono for stable analysis.

---

# ✔ Summary

This project extracts **real, deterministic, DSP-based audio descriptors** that directly correlate with DJ‑perceived musical properties:
- tonal center  
- rhythmic density  
- transient sharpness  
- spectral balance  
- percussive dominance

These form a robust baseline for further energy models, compatibility scoring, and integration with Spotify data if desired.

