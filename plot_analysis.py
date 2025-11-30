#!/usr/bin/env python3
"""
DJ Music Analysis Visualization
================================
Multi-panel visualization of MIR features from analysis.csv
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

# Load data
csv_path = Path(__file__).parent / "analysis.csv"
df = pd.read_csv(csv_path)

# Extract track names (remove extension for cleaner display)
df['track'] = df['filename'].apply(lambda x: Path(x).stem)
num_tracks = len(df)

# Calculate figure height based on number of tracks (min 10, grows with tracks)
fig_height = max(10, num_tracks * 0.4 + 4)
fig = plt.figure(figsize=(20, fig_height))
fig.suptitle('DJ Music Analysis – MIR Features Overview', fontsize=18, fontweight='bold', y=0.998)

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

# 1. BPM Distribution (leftmost, with track labels)
ax1 = plt.subplot(2, 4, 1)
ax1.barh(track_indices, df['bpm'], color=COLOR_BPM, alpha=0.8, height=0.8)
ax1.set_xlabel('BPM', fontweight='bold', fontsize=10)
ax1.set_title('Tempo', fontsize=13, pad=10, fontweight='bold')
ax1.set_yticks(track_indices)
ax1.set_yticklabels(df['track'], fontsize=8)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# 2. Energy Profile (RMS + Percussive Ratio)
ax2 = plt.subplot(2, 4, 2, sharey=ax1)
width = 0.38
ax2.barh(track_indices - width/2, df['rms_mean'], width, label='RMS Energy', color=COLOR_RMS, alpha=0.85)
ax2.barh(track_indices + width/2, df['percussive_ratio'], width, label='Percussive Ratio', color=COLOR_PERCUSSIVE, alpha=0.85)
ax2.set_xlabel('Value', fontweight='bold', fontsize=10)
ax2.set_title('Energy Profile', fontsize=13, pad=10, fontweight='bold')
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.tick_params(labelleft=False)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 3. Rhythmic Intensity (Onset Rate + Attack Sharpness) - with proper legend
ax3 = plt.subplot(2, 4, 3, sharey=ax1)
width = 0.38
ax3.barh(track_indices - width/2, df['onset_rate'], width, label='Onset Rate', color=COLOR_ONSET, alpha=0.85)
ax3.barh(track_indices + width/2, df['attack_sharpness'], width, label='Attack Sharpness', color=COLOR_ATTACK, alpha=0.85)
ax3.set_xlabel('Value', fontweight='bold', fontsize=10)
ax3.set_title('Rhythmic Intensity', fontsize=13, pad=10, fontweight='bold')
ax3.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax3.tick_params(labelleft=False)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# 4. Spectral Balance (HF vs Bass Ratio)
ax4 = plt.subplot(2, 4, 4, sharey=ax1)
width = 0.38
ax4.barh(track_indices - width/2, df['hf_ratio'], width, label='High Freq (≥6kHz)', color=COLOR_HF, alpha=0.85)
ax4.barh(track_indices + width/2, df['bass_ratio'], width, label='Bass (<200Hz)', color=COLOR_BASS, alpha=0.85)
ax4.set_xlabel('Ratio', fontweight='bold', fontsize=10)
ax4.set_title('Spectral Balance', fontsize=13, pad=10, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax4.tick_params(labelleft=False)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# 5. Brightness (Spectral Centroid) - solid color
ax5 = plt.subplot(2, 4, 5, sharey=ax1)
ax5.barh(track_indices, df['centroid_mean'], color=COLOR_CENTROID, alpha=0.8, height=0.8)
ax5.set_xlabel('Spectral Centroid (Hz)', fontweight='bold', fontsize=10)
ax5.set_title('Brightness', fontsize=13, pad=10, fontweight='bold')
ax5.set_yticks(track_indices)
ax5.set_yticklabels(df['track'], fontsize=8)
ax5.grid(axis='x', alpha=0.3, linestyle='--')

# 6. Spectral Flux (Dynamic Change) - solid color
ax6 = plt.subplot(2, 4, 6, sharey=ax1)
ax6.barh(track_indices, df['flux_mean'], color=COLOR_FLUX, alpha=0.8, height=0.8)
ax6.set_xlabel('Spectral Flux', fontweight='bold', fontsize=10)
ax6.set_title('Spectral Dynamics', fontsize=13, pad=10, fontweight='bold')
ax6.tick_params(labelleft=False)
ax6.grid(axis='x', alpha=0.3, linestyle='--')

# 7. RMS Std (Dynamics variation)
ax7 = plt.subplot(2, 4, 7, sharey=ax1)
ax7.barh(track_indices, df['rms_std'], color='#7f7f7f', alpha=0.8, height=0.8)
ax7.set_xlabel('RMS Std Dev', fontweight='bold', fontsize=10)
ax7.set_title('Energy Variation', fontsize=13, pad=10, fontweight='bold')
ax7.tick_params(labelleft=False)
ax7.grid(axis='x', alpha=0.3, linestyle='--')

# 8. Centroid Std (Timbral variation)
ax8 = plt.subplot(2, 4, 8, sharey=ax1)
ax8.barh(track_indices, df['centroid_std'], color='#404040', alpha=0.8, height=0.8)
ax8.set_xlabel('Centroid Std Dev (Hz)', fontweight='bold', fontsize=10)
ax8.set_title('Timbral Variation', fontsize=13, pad=10, fontweight='bold')
ax8.tick_params(labelleft=False)
ax8.grid(axis='x', alpha=0.3, linestyle='--')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
output_path = Path(__file__).parent / "analysis_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Show plot
plt.show()
