#!/usr/bin/env python3
"""
Advanced Music Clustering Analysis - t-SNE with Camelot Wheel
==============================================================
t-SNE clustering with circular key encoding for harmonic continuity
Camelot wheel color coding + shape markers (circle=major, square=minor)
Interactive sortable table with color-coded parameter intensities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Features to use for clustering
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
    'attack_sharpness',
    'bpm'
]

# Feature display names for table
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
    'attack_sharpness': 'Attack Sharpness',
    'bpm': 'BPM'
}

# Key to numeric mapping (chromatic circle)
KEY_MAPPING = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# Camelot wheel mapping (for harmonic mixing)
CAMELOT_WHEEL = {
    'Ab': {'major': '4B', 'minor': '1A'},
    'B': {'major': '1B', 'minor': '10A'},
    'Db': {'major': '3B', 'minor': '12A'},
    'Eb': {'major': '5B', 'minor': '2A'},
    'F#': {'major': '2B', 'minor': '11A'},
    'Gb': {'major': '2B', 'minor': '11A'},
    'A': {'major': '11B', 'minor': '8A'},
    'Bb': {'major': '6B', 'minor': '3A'},
    'C': {'major': '8B', 'minor': '5A'},
    'C#': {'major': '12B', 'minor': '9A'},
    'D': {'major': '10B', 'minor': '7A'},
    'E': {'major': '12B', 'minor': '9A'},
    'F': {'major': '7B', 'minor': '4A'},
    'G': {'major': '9B', 'minor': '6A'}
}

# Camelot colors (position-based with circular continuity)
CAMELOT_COLORS = [
    '#FF0000', '#FF4500', '#FF8C00', '#FFD700',  # 1-4: Red to Yellow
    '#9ACD32', '#32CD32', '#00FA9A', '#00CED1',  # 5-8: Green to Cyan
    '#1E90FF', '#4169E1', '#8A2BE2', '#FF1493'   # 9-12: Blue to Magenta
]

# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------

print("=" * 70)
print("CLUSTERING ANALYSIS - t-SNE with Camelot Wheel")
print("=" * 70)

# Find all analysis CSV files
csv_files = list(Path(".").glob("analysis*.csv"))

if not csv_files:
    print("\n[ERROR] No analysis CSV files found.")
    print("Please run analyse_music.py or analyse_from_playlist.py first.")
    exit(1)

print(f"\nFound {len(csv_files)} analysis file(s)")

# If multiple CSVs, let user choose
if len(csv_files) > 1:
    print("\nAvailable analysis files:")
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"  {idx}. {csv_file.name}")
    print(f"  {len(csv_files) + 1}. Merge all files")
    
    while True:
        try:
            choice = int(input(f"\nSelect file for clustering (1-{len(csv_files) + 1}): "))
            if 1 <= choice <= len(csv_files) + 1:
                break
            print(f"Please enter a number between 1 and {len(csv_files) + 1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    if choice <= len(csv_files):
        selected_csv = csv_files[choice - 1]
        print(f"\nSelected: {selected_csv.name}")
        df_all = pd.read_csv(selected_csv)
        df_all['source_file'] = selected_csv.stem
    else:
        print(f"\nMerging all {len(csv_files)} files...")
        dfs = []
        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file)
            df_temp['source_file'] = csv_file.stem
            dfs.append(df_temp)
        df_all = pd.concat(dfs, ignore_index=True)
else:
    print(f"\nUsing: {csv_files[0].name}")
    df_all = pd.read_csv(csv_files[0])
    df_all['source_file'] = csv_files[0].stem

# Try to find and merge with original Engine DJ CSV for file paths
# Look for corresponding original CSV (without 'analysis_' prefix and '_tracks_...' suffix)
source_csv_pattern = df_all['source_file'].iloc[0] if 'source_file' in df_all.columns else csv_files[0].stem
# Extract original filename (e.g., 'analysis_A_IMPORT_tracks_1-979' -> 'A_IMPORT')
if source_csv_pattern.startswith('analysis_'):
    original_name = source_csv_pattern.replace('analysis_', '').split('_tracks_')[0]
    original_csv_path = Path(original_name + '.csv')
    
    if original_csv_path.exists():
        print(f"Loading file paths from: {original_csv_path.name}")
        df_original = pd.read_csv(original_csv_path)
        
        # Merge on Title, Artist, Album to get file paths
        if 'File name' in df_original.columns:
            merge_cols = []
            if 'Title' in df_all.columns and 'Title' in df_original.columns:
                merge_cols.append('Title')
            if 'Artist' in df_all.columns and 'Artist' in df_original.columns:
                merge_cols.append('Artist')
            if 'Album' in df_all.columns and 'Album' in df_original.columns:
                merge_cols.append('Album')
            
            if merge_cols:
                df_all = df_all.merge(
                    df_original[merge_cols + ['File name']],
                    on=merge_cols,
                    how='left'
                )
                print(f"Merged file paths for {df_all['File name'].notna().sum()} tracks")

# Create display labels
if 'Title' in df_all.columns and 'Artist' in df_all.columns:
    df_all['label'] = df_all.apply(
        lambda row: f"{row['Artist']} - {row['Title']}" 
        if pd.notna(row['Artist']) and pd.notna(row['Title'])
        else row.get('filename', 'Unknown'), axis=1
    )
else:
    df_all['label'] = df_all['filename'].apply(lambda x: Path(x).stem)

# Remove duplicates
df_all = df_all.drop_duplicates(subset='label', keep='first')

print(f"Loaded {len(df_all)} unique tracks")

# -------------------------------------------------------------------
# Prepare feature matrix with circular key encoding
# -------------------------------------------------------------------

# Extract numerical features
X_numerical = df_all[FEATURE_COLUMNS].values

# Encode key using circular representation (sin/cos to handle 12->1 continuity)
df_all['key_numeric'] = df_all['key'].map(KEY_MAPPING).fillna(0)
key_radians = 2 * np.pi * df_all['key_numeric'] / 12
key_sin = np.sin(key_radians).values.reshape(-1, 1)
key_cos = np.cos(key_radians).values.reshape(-1, 1)

# Encode mode (0 = minor, 1 = major)
mode_numeric = (df_all['mode'] == 'major').astype(int).values.reshape(-1, 1)

# Combine features (key as sin/cos for circular continuity)
X_combined = np.hstack([X_numerical, key_sin, key_cos, mode_numeric])

# Check for missing values
if np.isnan(X_combined).any():
    print("Warning: Found missing values, filling with column means...")
    X_combined = pd.DataFrame(X_combined).fillna(pd.DataFrame(X_combined).mean()).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

print(f"Prepared feature matrix: {X_combined.shape[0]} tracks x {X_combined.shape[1]} features")
print(f"  - {len(FEATURE_COLUMNS)} numerical features (includes mean + std)")
print(f"  - 3 encoded features (key_sin, key_cos, mode) for circular continuity")

# -------------------------------------------------------------------
# Compute t-SNE
# -------------------------------------------------------------------

print("\nComputing t-SNE embedding (this may take a minute)...")
tsne_model = TSNE(
    n_components=2,
    perplexity=min(30, len(df_all) - 1),
    learning_rate='auto',
    init='pca',
    random_state=42,
    max_iter=1000
)
X_tsne = tsne_model.fit_transform(X_scaled)
print(f"  t-SNE completed")

# Add t-SNE coordinates to dataframe
df_all['tsne_x'] = X_tsne[:, 0]
df_all['tsne_y'] = X_tsne[:, 1]

# Generate Camelot notation for each track
df_all['camelot'] = df_all.apply(
    lambda row: CAMELOT_WHEEL.get(row['key'], {}).get(row['mode'], 'N/A'),
    axis=1
)

# Extract Camelot number (1-12) for color mapping
def get_camelot_number(camelot_str):
    if camelot_str == 'N/A':
        return 0
    return int(camelot_str[:-1])  # Remove 'A' or 'B' suffix

df_all['camelot_number'] = df_all['camelot'].apply(get_camelot_number)

# Assign colors based on Camelot number
df_all['color'] = df_all['camelot_number'].apply(
    lambda x: CAMELOT_COLORS[x - 1] if 1 <= x <= 12 else '#808080'
)

# -------------------------------------------------------------------
# Create interactive plot with Camelot colors and shape markers
# -------------------------------------------------------------------

print("\nGenerating interactive visualization...")

# Separate major and minor for different marker shapes
df_major = df_all[df_all['mode'] == 'major'].copy()
df_minor = df_all[df_all['mode'] == 'minor'].copy()

X_tsne_major = X_tsne[df_all['mode'] == 'major']
X_tsne_minor = X_tsne[df_all['mode'] == 'minor']

# Create figure
fig = go.Figure()

# Add major tracks (circles)
if len(df_major) > 0:
    fig.add_trace(
        go.Scatter(
            x=X_tsne_major[:, 0],
            y=X_tsne_major[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=df_major['color'],
                opacity=0.8,
                line=dict(width=1, color='white'),
                symbol='circle'
            ),
            text=df_major['camelot'],
            hovertext=df_major.apply(
                lambda row: f"<b>{row['label']}</b><br>"
                           f"Key: {row['camelot']} ({row['key']} {row['mode']})<br>"
                           f"BPM: {row['bpm']:.1f}<br>"
                           f"RMS: {row['rms_mean']:.3f} | Brightness: {row['centroid_mean']:.0f}<br>"
                           f"t-SNE: ({row['tsne_x']:.2f}, {row['tsne_y']:.2f})",
                axis=1
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            name='Major',
            showlegend=True
        )
    )

# Add minor tracks (squares)
if len(df_minor) > 0:
    fig.add_trace(
        go.Scatter(
            x=X_tsne_minor[:, 0],
            y=X_tsne_minor[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=df_minor['color'],
                opacity=0.8,
                line=dict(width=1, color='white'),
                symbol='square'
            ),
            text=df_minor['camelot'],
            hovertext=df_minor.apply(
                lambda row: f"<b>{row['label']}</b><br>"
                           f"Key: {row['camelot']} ({row['key']} {row['mode']})<br>"
                           f"BPM: {row['bpm']:.1f}<br>"
                           f"RMS: {row['rms_mean']:.3f} | Brightness: {row['centroid_mean']:.0f}<br>"
                           f"t-SNE: ({row['tsne_x']:.2f}, {row['tsne_y']:.2f})",
                axis=1
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            name='Minor',
            showlegend=True
        )
    )

# Update layout
fig.update_layout(
    title=dict(
        text=f't-SNE Clustering - {len(df_all)} tracks - Color: Camelot Wheel | Shape: Circle=Major, Square=Minor',
        x=0.5,
        xanchor='center',
        font=dict(size=14, color='#333')
    ),
    xaxis=dict(
        title='t-SNE 1',
        title_font=dict(size=13),
        zeroline=True, zerolinewidth=1, zerolinecolor='lightgray',
        tickfont=dict(size=11),
        showgrid=True, gridwidth=1, gridcolor='lightgray'
    ),
    yaxis=dict(
        title='t-SNE 2',
        title_font=dict(size=13),
        zeroline=True, zerolinewidth=1, zerolinecolor='lightgray',
        tickfont=dict(size=11),
        showgrid=True, gridwidth=1, gridcolor='lightgray'
    ),
    width=1200,
    height=650,
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=11),
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )
)

# -------------------------------------------------------------------
# Create interactive sortable table with color-coded parameters
# -------------------------------------------------------------------

print("Building interactive table with sortable columns...")

# Table columns (exclude std and tsne, keep only mean values + BPM)
TABLE_COLUMNS = [
    'rms_mean',
    'centroid_mean',
    'flux_mean',
    'percussive_ratio',
    'onset_rate',
    'hf_ratio',
    'bass_ratio',
    'attack_sharpness',
    'bpm'
]

# Normalize features to 0-1 range for color coding
scaler_table = MinMaxScaler()
feature_normalized = scaler_table.fit_transform(df_all[TABLE_COLUMNS])

# Create color-coded cells for each feature
def value_to_color(value):
    """Convert normalized value (0-1) to RGB color (Blue-White-Red, red=high)"""
    if value < 0.5:
        # Blue to White (0 to 0.5)
        intensity = int(255 * value * 2)
        return f'rgb({intensity},{intensity},255)'
    else:
        # White to Red (0.5 to 1)
        intensity = int(255 * (1 - (value - 0.5) * 2))
        return f'rgb(255,{intensity},{intensity})'

# Build HTML table with styling
table_html = '<div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">'
table_html += '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">'
table_html += '<h3 style="margin: 0; color: #333;">Track Parameters - Sortable & Filterable</h3>'
table_html += '<div>'
table_html += '<button onclick="exportToM3U()" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px; margin-right: 10px;">📥 Export to M3U Playlist</button>'
table_html += '<button onclick="exportToEngineCSV()" style="padding: 10px 20px; background: #27ae60; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px;">📥 Export to CSV</button>'
table_html += '</div>'
table_html += '</div>'
table_html += '<div style="margin-bottom: 10px; color: #555; font-size: 13px;"><span id="visibleCount">{len(df_all)}</span> / {len(df_all)} tracks visible</div>'
table_html += '<input type="text" id="searchBox" placeholder="Filter tracks by name, key, BPM..." style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px;">'
table_html += '<div style="overflow-x: auto;">'
table_html += '<table id="trackTable" style="width: 100%; border-collapse: collapse; font-size: 11px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'

# Table header with filter row
table_html += '<thead>'
table_html += '<tr style="background: #2c3e50; color: white;">'
table_html += '<th rowspan="2" onclick="sortTable(0)" style="cursor: pointer; padding: 12px; text-align: left; border: 1px solid #34495e; font-weight: bold; vertical-align: middle;"># ▲▼</th>'
table_html += '<th rowspan="2" onclick="sortTable(1)" style="cursor: pointer; padding: 12px; text-align: left; border: 1px solid #34495e; font-weight: bold; min-width: 200px; vertical-align: middle;">Track ▲▼</th>'
table_html += '<th rowspan="2" onclick="sortTable(2)" style="cursor: pointer; padding: 12px; text-align: center; border: 1px solid #34495e; font-weight: bold; vertical-align: middle;">Key ▲▼</th>'
for i, col in enumerate(TABLE_COLUMNS):
    table_html += f'<th onclick="sortTable({3+i})" style="cursor: pointer; padding: 8px 4px; text-align: right; border: 1px solid #34495e; font-weight: bold; font-size: 10px;" title="{FEATURE_NAMES[col]}">{FEATURE_NAMES[col]} ▲▼</th>'
table_html += '</tr>'
# Filter row
table_html += '<tr style="background: #34495e;">'
for i, col in enumerate(TABLE_COLUMNS):
    table_html += f'<th style="padding: 4px; border: 1px solid #34495e;"><input type="text" class="filter-input" data-col="{3+i}" placeholder="min-max" style="width: 80%; padding: 3px; font-size: 10px; text-align: center;"></th>'
table_html += '</tr></thead>'

# Table body
table_html += '<tbody>'
for idx, (_, row) in enumerate(df_all.iterrows()):
    row_bg = '#f9f9f9' if idx % 2 == 0 else 'white'
    # Store original Engine DJ data as data attributes
    artist = row.get('Artist', '')
    title = row.get('Title', '')
    album = row.get('Album', '')
    filepath = row.get('File name', '')  # Use 'File name' from Engine DJ CSV
    table_html += f'<tr class="data-row" style="border-bottom: 1px solid #eee; background: {row_bg};" data-artist="{artist}" data-title="{title}" data-album="{album}" data-filepath="{filepath}" data-key="{row["key"]}" data-mode="{row["mode"]}" data-bpm="{row["bpm"]:.1f}">'
    table_html += f'<td style="padding: 10px; border: 1px solid #ddd;">{idx+1}</td>'
    table_html += f'<td style="padding: 10px; border: 1px solid #ddd; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{row["label"]}">{row["label"]}</td>'
    
    # Camelot with color
    camelot_color = row['color']
    table_html += f'<td style="padding: 10px; border: 1px solid #ddd; text-align: center; background: {camelot_color}; color: white; font-weight: bold; font-size: 12px;">{row["camelot"]}</td>'
    
    # Color-coded feature values (no std, no tsne)
    for i, col in enumerate(TABLE_COLUMNS):
        val = row[col]
        normalized_val = feature_normalized[idx, i]
        bg_color = value_to_color(normalized_val)
        text_color = '#000' if 0.3 < normalized_val < 0.7 else '#fff'
        table_html += f'<td class="data-cell" data-col="{3+i}" data-value="{val:.3f}" style="padding: 10px; border: 1px solid #ddd; text-align: right; background: {bg_color}; color: {text_color}; font-weight: bold; font-family: monospace;">{val:.3f}</td>'
    
    table_html += '</tr>'

table_html += '</tbody></table></div></div>'

# Add JavaScript for sorting and filtering
table_js = '''
<script>
let sortState = {};

function sortTable(columnIndex) {
  var table = document.getElementById("trackTable");
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  
  // Toggle sort direction
  if (!sortState[columnIndex]) {
    sortState = {};  // Reset other columns
    sortState[columnIndex] = 'asc';
  } else if (sortState[columnIndex] === 'asc') {
    sortState[columnIndex] = 'desc';
  } else {
    sortState[columnIndex] = 'asc';
  }
  
  var dir = sortState[columnIndex];
  var multiplier = dir === 'asc' ? 1 : -1;
  
  // Sort rows array
  rows.sort(function(rowA, rowB) {
    var cellA = rowA.cells[columnIndex];
    var cellB = rowB.cells[columnIndex];
    
    var aContent = cellA.textContent.trim();
    var bContent = cellB.textContent.trim();
    
    // Try numeric comparison
    var aNum = parseFloat(aContent);
    var bNum = parseFloat(bContent);
    
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return (aNum - bNum) * multiplier;
    }
    
    // String comparison
    return aContent.localeCompare(bContent) * multiplier;
  });
  
  // Rebuild tbody with sorted rows (more efficient than swapping)
  var fragment = document.createDocumentFragment();
  rows.forEach(function(row) {
    fragment.appendChild(row);
  });
  tbody.innerHTML = '';
  tbody.appendChild(fragment);
}

document.getElementById('searchBox').addEventListener('keyup', function() {
  applyFilters();
});

// Range filter for numeric columns
var filterInputs = document.querySelectorAll('.filter-input');
filterInputs.forEach(function(input) {
  input.addEventListener('keyup', function() {
    applyFilters();
  });
});

function applyFilters() {
  var searchInput = document.getElementById("searchBox");
  var searchFilter = searchInput.value.toUpperCase();
  var table = document.getElementById("trackTable");
  var tbody = table.tBodies[0];
  var rows = tbody.getElementsByTagName("tr");
  
  // Get all range filters
  var rangeFilters = {};
  filterInputs.forEach(function(input) {
    var colIndex = parseInt(input.getAttribute('data-col'));
    var value = input.value.trim();
    if (value) {
      var parts = value.split('-');
      if (parts.length === 2) {
        rangeFilters[colIndex] = {
          min: parseFloat(parts[0]) || -Infinity,
          max: parseFloat(parts[1]) || Infinity
        };
      } else if (parts.length === 1 && !isNaN(parseFloat(parts[0]))) {
        var val = parseFloat(parts[0]);
        rangeFilters[colIndex] = { min: val, max: val };
      }
    }
  });
  
  // Apply filters to rows
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    var show = true;
    
    // Text search filter
    if (searchFilter) {
      var cells = row.getElementsByTagName("td");
      var found = false;
      for (var j = 0; j < cells.length; j++) {
        var txtValue = cells[j].textContent || cells[j].innerText;
        if (txtValue.toUpperCase().indexOf(searchFilter) > -1) {
          found = true;
          break;
        }
      }
      if (!found) show = false;
    }
    
    // Range filters
    if (show) {
      for (var colIndex in rangeFilters) {
        var cell = row.cells[colIndex];
        if (cell) {
          var cellValue = parseFloat(cell.getAttribute('data-value'));
          if (!isNaN(cellValue)) {
            if (cellValue < rangeFilters[colIndex].min || cellValue > rangeFilters[colIndex].max) {
              show = false;
              break;
            }
          }
        }
      }
    }
    
    row.style.display = show ? "" : "none";
  }
  
  // Update visible count
  updateVisibleCount();
}

function updateVisibleCount() {
  var table = document.getElementById("trackTable");
  var tbody = table.tBodies[0];
  var rows = tbody.getElementsByTagName("tr");
  var visibleCount = 0;
  
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].style.display !== "none") {
      visibleCount++;
    }
  }
  
  document.getElementById("visibleCount").textContent = visibleCount;
}

function exportToM3U() {
  var table = document.getElementById("trackTable");
  var tbody = table.tBodies[0];
  var rows = tbody.getElementsByTagName("tr");
  
  // Build M3U playlist with extended info
  var m3u = "#EXTM3U\\n";
  var trackCount = 0;
  
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    if (row.style.display !== "none") {
      var artist = row.getAttribute("data-artist");
      var title = row.getAttribute("data-title");
      var filepath = row.getAttribute("data-filepath");
      
      // Add extended info: #EXTINF:duration,artist - title
      // Duration is unknown, use -1
      m3u += "#EXTINF:-1," + artist + " - " + title + "\\n";
      m3u += filepath + "\\n";
      trackCount++;
    }
  }
  
  // Create download
  var blob = new Blob([m3u], { type: "audio/x-mpegurl;charset=utf-8;" });
  var link = document.createElement("a");
  var url = URL.createObjectURL(blob);
  
  var timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  link.setAttribute("href", url);
  link.setAttribute("download", "filtered_playlist_" + timestamp + ".m3u");
  link.style.visibility = "hidden";
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  alert("Exported " + trackCount + " filtered tracks to M3U playlist!\\n\\nTo import:\\n• iTunes/Music: File → Library → Import Playlist\\n• Engine DJ: Will read from iTunes library automatically");
}

function exportToEngineCSV() {
  var table = document.getElementById("trackTable");
  var tbody = table.tBodies[0];
  var rows = tbody.getElementsByTagName("tr");
  
  // Build Engine DJ CSV format
  var csv = "#,Title,Artist,Album,Key\\n";
  var trackNum = 1;
  
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    if (row.style.display !== "none") {
      var artist = row.getAttribute("data-artist");
      var title = row.getAttribute("data-title");
      var album = row.getAttribute("data-album");
      var key = row.getAttribute("data-key");
      var mode = row.getAttribute("data-mode");
      
      // Escape quotes and commas for CSV
      var escapeCSV = function(str) {
        if (!str) return "";
        str = String(str);
        if (str.indexOf(",") > -1 || str.indexOf('"') > -1 || str.indexOf("\\n") > -1) {
          return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
      };
      
      csv += trackNum + ",";
      csv += escapeCSV(title) + ",";
      csv += escapeCSV(artist) + ",";
      csv += escapeCSV(album) + ",";
      csv += key + " " + mode + "\\n";
      trackNum++;
    }
  }
  
  // Create download
  var blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  var link = document.createElement("a");
  var url = URL.createObjectURL(blob);
  
  var timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  link.setAttribute("href", url);
  link.setAttribute("download", "filtered_playlist_" + timestamp + ".csv");
  link.style.visibility = "hidden";
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  alert("Exported " + (trackNum - 1) + " filtered tracks to CSV format!");
}
</script>
'''

# Combine plot and table into single HTML
print("Combining plot and table into HTML...")
output_html = Path("clustering_analysis.html")

# Write plot to temporary string
plot_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

# Combine everything
full_html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Clustering Analysis - t-SNE with Camelot Wheel</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #ffffff;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        .info {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        #trackTable .data-row:hover {{
            background-color: #e8f4f8 !important;
        }}
    </style>
</head>
<body>
    <h1>t-SNE Clustering Analysis - Music Features</h1>
    <div class="info">
        <strong>{len(df_all)} tracks</strong> | 
        Circular key encoding (sin/cos for 12→1 continuity) | 
        Camelot wheel color coding | 
        Shapes: ●=Major ■=Minor | 
        Heatmap: 🔵 Low → ⚪ Mid → 🔴 High
    </div>
    {plot_html}
    {table_html}
    {table_js}
    <div style="margin-top: 30px; padding: 15px; background: #ecf0f1; border-radius: 5px; color: #555;">
        <strong>Usage:</strong><br>
        • Hover over points for track info | Click column headers to sort<br>
        • Search box for text filter | Column filters accept range (e.g., "120-130" or "0.5-0.8")<br>
        • <strong>Export filtered results:</strong><br>
        &nbsp;&nbsp;- <strong>M3U Playlist (recommended):</strong> Import directly to iTunes/Music.app (File → Library → Import Playlist). Engine DJ will see it in your iTunes library.<br>
        &nbsp;&nbsp;- <strong>CSV:</strong> Alternative format with track metadata
    </div>
</body>
</html>
'''

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Interactive HTML saved: {output_html}")

# -------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTERING SUMMARY")
print("=" * 70)

# Calculate spread metrics
tsne_spread = np.std(X_tsne, axis=0)

print(f"\nt-SNE spread:")
print(f"  Dimension 1 std: {tsne_spread[0]:.2f}")
print(f"  Dimension 2 std: {tsne_spread[1]:.2f}")

print(f"\nKey distribution (top 5):")
key_counts = df_all['key'].value_counts().head(5)
for key, count in key_counts.items():
    print(f"  {key}: {count} tracks ({count/len(df_all)*100:.1f}%)")

print(f"\nMode distribution:")
mode_counts = df_all['mode'].value_counts()
for mode, count in mode_counts.items():
    print(f"  {mode.capitalize()}: {count} tracks ({count/len(df_all)*100:.1f}%)")

print(f"\nCamelot distribution (top 5):")
camelot_counts = df_all['camelot'].value_counts().head(5)
for camelot, count in camelot_counts.items():
    print(f"  {camelot}: {count} tracks ({count/len(df_all)*100:.1f}%)")

print("\n" + "=" * 70)
print("Clustering analysis complete!")
print(f"  Open {output_html} in browser for interactive exploration")
print(f"  - t-SNE plot colored by Camelot wheel (harmonic mixing)")
print(f"  - Sortable table with color-coded intensity bars")
print(f"  - Filter tracks by typing in search box")
print(f"  - Click column headers to sort by any parameter")
print("=" * 70 + "\n")
