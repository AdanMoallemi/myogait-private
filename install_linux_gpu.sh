#!/usr/bin/env bash
set -e

echo "====================================================================="
echo "  MyoGait DCM Pipeline - Automated GPU Setup for Linux/Mac"
echo "====================================================================="

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] 'conda' command was not found!"
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

echo "[1/4] Creating/updating Conda environment 'dcm-gait' (Python 3.11)..."
conda create -y -n dcm-gait python=3.11 || true

echo "[2/4] Activating 'dcm-gait'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dcm-gait

echo "[3/4] Installing PyTorch with CUDA support (if Linux) or default..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    pip install torch torchvision
fi

echo "[4/4] Installing MyoGait, Dashboard, and Backends..."
pip install -e ".[all]"
pip install streamlit openpyxl

echo "Verifying environment..."
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "====================================================================="
echo "  Setup Complete! Launch with ./run_dashboard.sh"
echo "====================================================================="
