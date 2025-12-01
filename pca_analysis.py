#!/usr/bin/env python3
"""
PCA Analysis of Music Features
===============================
Merge all analysis CSVs and visualize tracks in PCA space
Interactive plots with hover tooltips showing track information
Shows both PCA of cases (tracks) and variables (features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Features to use for PCA (excluding metadata)
FEATURE_COLUMNS = [
    'rms_mean',
    'rms_std',
    'centroid_mean',
    'centroid_std',
    'flux_mean',
    'flux_std',
    'percussive_ratio',
    'onset_rate',
    'hf_ratio',
    'bass_ratio',
    'attack_sharpness'
]

# Feature display names for plots
FEATURE_NAMES = {
    'rms_mean': 'RMS Energy',
    'rms_std': 'Energy Variation',
    'centroid_mean': 'Brightness',
    'centroid_std': 'Timbral Variation',
    'flux_mean': 'Spectral Flux',
    'flux_std': 'Flux Variation',
    'percussive_ratio': 'Percussive Ratio',
    'onset_rate': 'Onset Rate',
    'hf_ratio': 'HF Ratio',
    'bass_ratio': 'Bass Ratio',
    'attack_sharpness': 'Attack Sharpness'
}

# -------------------------------------------------------------------
# Load and merge data
# -------------------------------------------------------------------

print("=" * 70)
print("PCA ANALYSIS - MUSIC FEATURES")
print("=" * 70)

# Find all analysis CSV files
csv_files = list(Path(".").glob("analysis*.csv"))

if not csv_files:
    print("\n[ERROR] No analysis CSV files found.")
    print("Please run analyse_music.py or analyse_from_playlist.py first.")
    exit(1)

print(f"\n✓ Found {len(csv_files)} analysis file(s)")

# If multiple CSVs, let user choose
if len(csv_files) > 1:
    print("\nAvailable analysis files:")
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"  {idx}. {csv_file.name}")
    print(f"  {len(csv_files) + 1}. Merge all files")
    
    while True:
        try:
            choice = int(input(f"\nSelect file for PCA (1-{len(csv_files) + 1}): "))
            if 1 <= choice <= len(csv_files) + 1:
                break
            print(f"Please enter a number between 1 and {len(csv_files) + 1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if choice <= len(csv_files):
        # Single file selected
        selected_csv = csv_files[choice - 1]
        print(f"\n✓ Selected: {selected_csv.name}")
        df_all = pd.read_csv(selected_csv)
        df_all['source_file'] = selected_csv.stem
    else:
        # Merge all files
        print(f"\n✓ Merging all {len(csv_files)} files...")
        dfs = []
        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file)
            df_temp['source_file'] = csv_file.stem
            dfs.append(df_temp)
        df_all = pd.concat(dfs, ignore_index=True)
else:
    # Only one CSV file
    print(f"\n✓ Using: {csv_files[0].name}")
    df_all = pd.read_csv(csv_files[0])
    df_all['source_file'] = csv_files[0].stem

# Create display labels
if 'Title' in df_all.columns and 'Artist' in df_all.columns:
    # Playlist-based analysis
    df_all['label'] = df_all.apply(
        lambda row: f"{row['Artist']} - {row['Title']}" 
        if pd.notna(row['Artist']) and pd.notna(row['Title'])
        else row.get('filename', 'Unknown'), axis=1
    )
    df_all['hover_info'] = df_all.apply(
        lambda row: f"<b>{row['Artist']} - {row['Title']}</b><br>"
                   f"Album: {row.get('Album', 'N/A')}<br>"
                   f"Key: {row['key']} {row['mode']}<br>"
                   f"BPM: {row['bpm']:.1f}"
        if pd.notna(row.get('Artist')) else row.get('filename', 'Unknown'),
        axis=1
    )
else:
    # Folder-based analysis
    df_all['label'] = df_all['filename'].apply(lambda x: Path(x).stem)
    df_all['hover_info'] = df_all.apply(
        lambda row: f"<b>{Path(row['filename']).stem}</b><br>"
                   f"Key: {row['key']} {row['mode']}<br>"
                   f"BPM: {row['bpm']:.1f}",
        axis=1
    )

# Remove duplicates (if same track analyzed multiple times)
df_all = df_all.drop_duplicates(subset='label', keep='first')

print(f"✓ Loaded {len(df_all)} unique tracks")

# -------------------------------------------------------------------
# Prepare data for PCA
# -------------------------------------------------------------------

# Extract feature matrix
X = df_all[FEATURE_COLUMNS].values

# Check for missing values
if np.isnan(X).any():
    print("⚠ Warning: Found missing values, filling with column means...")
    X = pd.DataFrame(X, columns=FEATURE_COLUMNS).fillna(pd.DataFrame(X, columns=FEATURE_COLUMNS).mean()).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Prepared feature matrix: {X.shape[0]} tracks × {X.shape[1]} features")

# -------------------------------------------------------------------
# Perform PCA
# -------------------------------------------------------------------

# Compute PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get explained variance
explained_var = pca.explained_variance_ratio_ * 100
cumulative_var = np.cumsum(explained_var)

print(f"\n✓ PCA completed")
print(f"  PC1 explains: {explained_var[0]:.1f}% variance")
print(f"  PC2 explains: {explained_var[1]:.1f}% variance")
print(f"  PC1+PC2 total: {cumulative_var[1]:.1f}% variance")

# Get loadings (variable contributions)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# -------------------------------------------------------------------
# Create interactive plots
# -------------------------------------------------------------------

print("\n✓ Generating interactive visualizations...")

# Create subplot figure
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('PCA - Tracks (Cases)', 'PCA - Features (Variables)'),
    horizontal_spacing=0.10
)

# Color coding by key mode
color_map = {'major': '#2E7D32', 'minor': '#C62828'}
colors = df_all['mode'].map(color_map).fillna('#666666')

# Plot 1: PCA of tracks (cases)
fig.add_trace(
    go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        marker=dict(
            size=9,
            color=colors,
            opacity=0.75,
            line=dict(width=0.8, color='white')
        ),
        text=df_all['label'],
        hovertext=df_all['hover_info'],
        hovertemplate='<span style="font-size:13px">%{hovertext}</span><extra></extra>',
        customdata=df_all['label'],
        name='Tracks',
        showlegend=False
    ),
    row=1, col=1
)

# Plot 2: PCA of variables (loadings)
# Scale loadings for visibility
loading_scale = 3.0
for i, feature in enumerate(FEATURE_COLUMNS):
    display_name = FEATURE_NAMES[feature]
    fig.add_trace(
        go.Scatter(
            x=[0, loadings[i, 0] * loading_scale],
            y=[0, loadings[i, 1] * loading_scale],
            mode='lines+markers+text',
            line=dict(color='#1f77b4', width=1.8),
            marker=dict(size=[0, 7]),
            text=['', display_name],
            textposition='top center',
            textfont=dict(size=11),
            hovertemplate=f'<span style="font-size:13px"><b>{display_name}</b><br>'
                         f'PC1: {loadings[i, 0]:.3f}<br>'
                         f'PC2: {loadings[i, 1]:.3f}</span><extra></extra>',
            name=display_name,
            showlegend=False
        ),
        row=1, col=2
    )

# Update axes
fig.update_xaxes(
    title_text=f'PC1 ({explained_var[0]:.1f}%)',
    title_font=dict(size=13),
    row=1, col=1,
    zeroline=True, zerolinewidth=1, zerolinecolor='lightgray',
    tickfont=dict(size=11)
)
fig.update_yaxes(
    title_text=f'PC2 ({explained_var[1]:.1f}%)',
    title_font=dict(size=13),
    row=1, col=1,
    zeroline=True, zerolinewidth=1, zerolinecolor='lightgray',
    tickfont=dict(size=11)
)

fig.update_xaxes(
    title_text=f'PC1 ({explained_var[0]:.1f}%)',
    title_font=dict(size=13),
    row=1, col=2,
    zeroline=True, zerolinewidth=2, zerolinecolor='black',
    tickfont=dict(size=11)
)
fig.update_yaxes(
    title_text=f'PC2 ({explained_var[1]:.1f}%)',
    title_font=dict(size=13),
    row=1, col=2,
    zeroline=True, zerolinewidth=2, zerolinecolor='black',
    tickfont=dict(size=11)
)

# Update layout
fig.update_layout(
    title=dict(
        text=f'PCA Analysis - Music Features ({len(df_all)} tracks) - Hover or click for track info',
        x=0.5,
        xanchor='center',
        font=dict(size=16, color='#333')
    ),
    width=1400,
    height=650,
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=11),
    clickmode='event+select'
)

# Add legend for color coding
fig.add_annotation(
    text="🟢 Major  🔴 Minor",
    xref="paper", yref="paper",
    x=0.48, y=-0.08,
    showarrow=False,
    font=dict(size=12)
)

# Save interactive HTML
output_html = Path("pca_analysis.html")
fig.write_html(output_html)
print(f"✓ Interactive plot saved: {output_html}")

# -------------------------------------------------------------------
# Create static matplotlib plot as backup
# -------------------------------------------------------------------

fig_static, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Tracks
for mode, color in color_map.items():
    mask = df_all['mode'] == mode
    ax1.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=color,
        label=mode.capitalize(),
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}% variance)', fontweight='bold', fontsize=11)
ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}% variance)', fontweight='bold', fontsize=11)
ax1.set_title('PCA - Tracks (Cases)', fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.8, alpha=0.5)

# Plot 2: Variables (biplot)
for i, feature in enumerate(FEATURE_COLUMNS):
    ax2.arrow(
        0, 0,
        loadings[i, 0] * loading_scale,
        loadings[i, 1] * loading_scale,
        head_width=0.15,
        head_length=0.15,
        fc='#1f77b4',
        ec='#1f77b4',
        alpha=0.7,
        linewidth=1.5
    )
    ax2.text(
        loadings[i, 0] * loading_scale * 1.1,
        loadings[i, 1] * loading_scale * 1.1,
        feature,
        fontsize=9,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
    )

ax2.set_xlabel(f'PC1 ({explained_var[0]:.1f}% variance)', fontweight='bold', fontsize=11)
ax2.set_ylabel(f'PC2 ({explained_var[1]:.1f}% variance)', fontweight='bold', fontsize=11)
ax2.set_title('PCA - Features (Variables)', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_aspect('equal')

plt.suptitle(f'PCA Analysis - Music Features ({len(df_all)} tracks)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save static plot
output_png = Path("pca_analysis.png")
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✓ Static plot saved: {output_png}")

# -------------------------------------------------------------------
# Print feature importance
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (PC1 & PC2)")
print("=" * 70)

# Calculate feature importance as absolute loading magnitude
importance_pc1 = np.abs(loadings[:, 0])
importance_pc2 = np.abs(loadings[:, 1])

print("\nPC1 - Top contributing features:")
pc1_ranking = sorted(zip(FEATURE_COLUMNS, importance_pc1), key=lambda x: x[1], reverse=True)
for i, (feat, val) in enumerate(pc1_ranking[:5], 1):
    print(f"  {i}. {FEATURE_NAMES[feat]:25s} {val:.3f}")

print("\nPC2 - Top contributing features:")
pc2_ranking = sorted(zip(FEATURE_COLUMNS, importance_pc2), key=lambda x: x[1], reverse=True)
for i, (feat, val) in enumerate(pc2_ranking[:5], 1):
    print(f"  {i}. {FEATURE_NAMES[feat]:25s} {val:.3f}")

print("\n" + "=" * 70)
print("✓ PCA analysis complete!")
print(f"  Open {output_html} in browser for interactive exploration")
print("=" * 70 + "\n")

# Show static plot
plt.show()
