#!/bin/bash

# --- absolute path to project root ---
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

VENV_PATH="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"

# --- required packages ---
REQUIRED_PACKAGES=(
    "numpy"
    "pandas"
    "matplotlib"
    "seaborn"
    "essentia"
    "soundfile"
    "librosa"
    "pyloudnorm"
    "ffmpeg-python"
    "spotipy"
    "scipy"
    "scikit-learn"
    "plotly"
    "umap-learn"
)

# --- create venv if missing ---
if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Creating uv virtual environment..."
    uv venv "$VENV_PATH"
fi

# --- activate env ---
source "$VENV_PATH/bin/activate"

# --- install packages if missing ---
echo "[INFO] Checking required packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! uv pip show "$pkg" > /dev/null 2>&1; then
        echo "[INFO] Installing missing package: $pkg"
        uv pip install "$pkg"
    fi
done

# --- run main script ---
if [ -f "$PROJECT_DIR/main.py" ]; then
    echo "[INFO] Running main.py..."
    python "$PROJECT_DIR/main.py" "$@"
else
    echo "[WARN] main.py not found in project root."
fi
