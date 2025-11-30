#!/usr/bin/env python3
"""
DJ Playlist Analysis Visualization
===================================
Multi-panel visualization of MIR features from playlist analysis CSV
Tracks sorted by playlist order (#)
Focuses on energy, rhythm, and spectral characteristics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'

# -------------------------------------------------------------------
# Select CSV file
# -------------------------------------------------------------------

print("=" * 70)
print("DJ PLAYLIST VISUALIZATION")
print("=" * 70)

# Find analysis CSV files (from playlist)
csv_files = list(Path(".").glob("analysis_*.csv"))

if not csv_files:
    print("\n[ERROR] No analysis CSV files found.")
    print("Please run analyse_from_playlist.py first.")
    exit(1)

print("\nAvailable analysis files:")
for idx, csv_file in enumerate(csv_files, 1):
    print(f"  {idx}. {csv_file.name}")

while True:
    try:
        csv_choice = int(input(f"\nSelect file to visualize (1-{len(csv_files)}): "))
        if 1 <= csv_choice <= len(csv_files):
            csv_path = csv_files[csv_choice - 1]
            break
        print(f"Please enter a number between 1 and {len(csv_files)}")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"\n✓ Selected: {csv_path.name}")
print(f"{'─' * 70}\n")

# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------

df = pd.read_csv(csv_path)

# Sort by track number (playlist order)
df = df.sort_values('#').reset_index(drop=True)

# Create display labels (track # + Artist - Title)
df['display'] = df.apply(lambda row: f"#{row['#']} {row['Artist']} - {row['Title']}", axis=1)

# Truncate long labels
df['display'] = df['display'].apply(lambda x: x[:80] + "..." if len(x) > 80 else x)

num_tracks = len(df)
print(f"✓ Loaded {num_tracks} tracks")
print(f"✓ Generating visualization...\n")

# -------------------------------------------------------------------
# Create visualization
# -------------------------------------------------------------------

# Extract playlist name from CSV filename
playlist_name = csv_path.stem.replace('analysis_', '').replace('_tracks_', ' tracks ')

# Calculate figure height based on number of tracks
fig_height = max(12, num_tracks * 0.35 + 3)
fig = plt.figure(figsize=(22, fig_height))
fig.suptitle(playlist_name, fontsize=16, fontweight='bold', x=0.02, y=0.995, ha='left')

# Track indices for shared y-axis
track_indices = np.arange(num_tracks)

# Single consistent color per metric
COLOR_BPM = '#1f77b4'
COLOR_RMS = '#ff7f0e'
COLOR_PERCUSSIVE = '#d62728'
COLOR_ONSET = '#2ca02c'
COLOR_ATTACK = '#9467bd'
COLOR_HF = '#17becf'
COLOR_BASS = '#bcbd22'
COLOR_CENTROID = '#e377c2'
COLOR_FLUX = '#8c564b'

# Define grid layout using GridSpec for better control
from matplotlib.gridspec import GridSpec

# Create GridSpec with 2 rows: top for headers, bottom for charts
gs = GridSpec(2, 13, figure=fig, height_ratios=[0.08, 1], hspace=0.02, wspace=0.4)

# Row 1: Headers spanning multiple columns
# Energy Profile header (spans columns 1-2)
ax_energy_header = fig.add_subplot(gs[0, 1:3])
ax_energy_header.text(0.5, 0.5, 'Energy Profile', ha='center', va='center', 
                      fontsize=12, fontweight='bold', color='#d84315')
ax_energy_header.axis('off')

# Rhythmic Intensity header (spans columns 3-4)
ax_rhythm_header = fig.add_subplot(gs[0, 3:5])
ax_rhythm_header.text(0.5, 0.5, 'Rhythmic Intensity', ha='center', va='center',
                      fontsize=12, fontweight='bold', color='#558b2f')
ax_rhythm_header.axis('off')

# Spectral Balance header (spans columns 5-6)
ax_spectral_header = fig.add_subplot(gs[0, 5:7])
ax_spectral_header.text(0.5, 0.5, 'Spectral Balance', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='#1565c0')
ax_spectral_header.axis('off')

# Row 2: All charts
# 1. Key + Track labels (leftmost) - with track numbers
ax1 = fig.add_subplot(gs[1, 0])
# Create key display with color coding
key_colors = {'major': '#2E7D32', 'minor': '#C62828'}  # green for major, red for minor
for idx, (key, mode) in enumerate(zip(df['key'], df['mode'])):
    color = key_colors.get(mode, '#000000')
    ax1.text(0.5, track_indices[idx], key, ha='center', va='center', 
             fontsize=8, fontweight='bold', color=color)
ax1.set_xlim(0, 1)
ax1.set_xlabel('Key', fontweight='bold', fontsize=9)
ax1.set_title('Key\n(maj/min)', fontsize=10, pad=8, fontweight='bold')
ax1.set_yticks(track_indices)
ax1.set_yticklabels(df['display'], fontsize=7)
ax1.invert_yaxis()
ax1.set_xticks([])
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# 2. RMS Energy
ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
ax2.barh(track_indices, df['rms_mean'], color=COLOR_RMS, alpha=0.85, height=0.8)
ax2.set_xlabel('RMS', fontweight='bold', fontsize=9)
ax2.set_title('RMS Energy', fontsize=10, pad=8)
ax2.tick_params(labelleft=False)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 3. Percussive Ratio
ax3 = fig.add_subplot(gs[1, 2], sharey=ax1)
ax3.barh(track_indices, df['percussive_ratio'], color=COLOR_PERCUSSIVE, alpha=0.85, height=0.8)
ax3.set_xlabel('Ratio', fontweight='bold', fontsize=9)
ax3.set_title('Percussive Ratio', fontsize=10, pad=8)
ax3.tick_params(labelleft=False)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# 4. Onset Rate
ax4 = fig.add_subplot(gs[1, 3], sharey=ax1)
ax4.barh(track_indices, df['onset_rate'], color=COLOR_ONSET, alpha=0.85, height=0.8)
ax4.set_xlabel('Events/sec', fontweight='bold', fontsize=9)
ax4.set_title('Onset Rate', fontsize=10, pad=8)
ax4.tick_params(labelleft=False)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# 5. Attack Sharpness
ax5 = fig.add_subplot(gs[1, 4], sharey=ax1)
ax5.barh(track_indices, df['attack_sharpness'], color=COLOR_ATTACK, alpha=0.85, height=0.8)
ax5.set_xlabel('Sharpness', fontweight='bold', fontsize=9)
ax5.set_title('Attack Sharpness', fontsize=10, pad=8)
ax5.tick_params(labelleft=False)
ax5.grid(axis='x', alpha=0.3, linestyle='--')

# 6. High Frequency Ratio
ax6 = fig.add_subplot(gs[1, 5], sharey=ax1)
ax6.barh(track_indices, df['hf_ratio'], color=COLOR_HF, alpha=0.85, height=0.8)
ax6.set_xlabel('Ratio', fontweight='bold', fontsize=9)
ax6.set_title('HF Ratio (≥6kHz)', fontsize=10, pad=8)
ax6.tick_params(labelleft=False)
ax6.grid(axis='x', alpha=0.3, linestyle='--')

# 7. Bass Ratio
ax7 = fig.add_subplot(gs[1, 6], sharey=ax1)
ax7.barh(track_indices, df['bass_ratio'], color=COLOR_BASS, alpha=0.85, height=0.8)
ax7.set_xlabel('Ratio', fontweight='bold', fontsize=9)
ax7.set_title('Bass Ratio (<200Hz)', fontsize=10, pad=8)
ax7.tick_params(labelleft=False)
ax7.grid(axis='x', alpha=0.3, linestyle='--')

# 8. Brightness (Spectral Centroid)
ax8 = fig.add_subplot(gs[1, 7], sharey=ax1)
ax8.barh(track_indices, df['centroid_mean'], color=COLOR_CENTROID, alpha=0.8, height=0.8)
ax8.set_xlabel('Hz', fontweight='bold', fontsize=9)
ax8.set_title('Brightness', fontsize=10, pad=8)
ax8.tick_params(labelleft=False)
ax8.grid(axis='x', alpha=0.3, linestyle='--')

# 9. Spectral Flux
ax9 = fig.add_subplot(gs[1, 8], sharey=ax1)
ax9.barh(track_indices, df['flux_mean'], color=COLOR_FLUX, alpha=0.8, height=0.8)
ax9.set_xlabel('Flux', fontweight='bold', fontsize=9)
ax9.set_title('Spectral Flux', fontsize=10, pad=8)
ax9.tick_params(labelleft=False)
ax9.grid(axis='x', alpha=0.3, linestyle='--')

# 10. RMS Std Dev
ax10 = fig.add_subplot(gs[1, 9], sharey=ax1)
ax10.barh(track_indices, df['rms_std'], color='#7f7f7f', alpha=0.8, height=0.8)
ax10.set_xlabel('Std Dev', fontweight='bold', fontsize=9)
ax10.set_title('Energy Variation', fontsize=10, pad=8)
ax10.tick_params(labelleft=False)
ax10.grid(axis='x', alpha=0.3, linestyle='--')

# No need for tight_layout with GridSpec - already handled

# Save figure
output_path = csv_path.parent / f"{csv_path.stem}_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

print(f"{'=' * 70}\n")

# Show plot
plt.show()
