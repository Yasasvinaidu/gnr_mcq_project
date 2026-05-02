#!/bin/bash
# ============================================================
# setup.bash  –  GNR Project: Deep Learning MCQ Solver
# ============================================================
# This script runs WITH internet access.
# It will:
#   1. Clone the project repository
#   2. Create conda environment  gnr_project_env  (Python 3.11)
#   3. Install all Python dependencies
#   4. Download Qwen2-VL-7B-Instruct model weights
#
# The grader will then run:
#   conda activate gnr_project_env
#   python inference.py --test_dir <path>
# ============================================================

set -e   # Exit immediately on any error

# ── CONFIG ──────────────────────────────────────────────────
# TODO: Replace with YOUR public GitHub repository URL before submitting
REPO_URL="https://github.com/Yasasvinaidu/gnr_mcq_project.git"
REPO_DIR="gnr_mcq_project"

CONDA_ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

# Model will be downloaded here (accessible without internet at inference)
MODEL_DIR="$HOME/models/Qwen2-VL-7B-Instruct"
MODEL_HF_ID="Qwen/Qwen2-VL-7B-Instruct"
# ────────────────────────────────────────────────────────────

echo "============================================"
echo "  GNR Project Setup Starting"
echo "============================================"

# ── Step 1: Clone repository ─────────────────────────────────
echo ""
echo "[1/4] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "  Repository already exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
    cd ..
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
echo "  ✓ Repository ready"

# ── Step 2: Create conda environment ─────────────────────────
echo ""
echo "[2/4] Creating conda environment: $CONDA_ENV_NAME (Python $PYTHON_VERSION)..."

# Source conda so we can use it in this script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Remove env if it already exists (clean install)
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "  Existing environment found, removing for clean install..."
    conda env remove -n "$CONDA_ENV_NAME" -y
fi

conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
echo "  ✓ Conda environment created"

# ── Step 3: Install dependencies ─────────────────────────────
echo ""
echo "[3/4] Installing Python packages..."

conda activate "$CONDA_ENV_NAME"

# PyTorch with CUDA 12.1 (compatible with CUDA 12.6 on L40s)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# HuggingFace ecosystem
pip install \
    "transformers>=4.45.0" \
    "accelerate>=0.26.0" \
    "bitsandbytes>=0.42.0" \
    "huggingface_hub>=0.20.0" \
    --quiet

# Qwen2-VL utilities and other dependencies
pip install \
    "qwen-vl-utils>=0.0.8" \
    "Pillow>=10.0.0" \
    "pandas>=2.0.0" \
    --quiet

echo "  ✓ All packages installed"

# ── Step 4: Download model weights ───────────────────────────
echo ""
echo "[4/4] Downloading model: $MODEL_HF_ID"
echo "      Saving to: $MODEL_DIR"

mkdir -p "$MODEL_DIR"

python - <<PYEOF
import os
from huggingface_hub import snapshot_download

model_dir = os.path.expanduser("$MODEL_DIR")
hf_id     = "$MODEL_HF_ID"

print(f"  Downloading {hf_id} ...")
snapshot_download(
    repo_id=hf_id,
    local_dir=model_dir,
    # Skip non-PyTorch formats to save space & time
    ignore_patterns=[
        "*.msgpack",
        "*.h5",
        "flax_model*",
        "tf_model*",
        "rust_model*",
        "onnx/*",
    ],
)
print(f"  ✓ Model saved at: {model_dir}")
PYEOF

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "  Model path: $MODEL_DIR"
echo ""
echo "  Next commands (run by grader):"
echo "    conda activate $CONDA_ENV_NAME"
echo "    python $REPO_DIR/inference.py --test_dir <path_to_test>"
echo "============================================"
