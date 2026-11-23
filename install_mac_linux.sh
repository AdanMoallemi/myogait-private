#!/usr/bin/env bash
# One-step setup of the DCM gait pipeline on macOS or Linux.
# (Windows: double-click install_windows_gpu.bat instead.)
#
#   ./install_mac_linux.sh            # install / refresh
#   SKIP_TORCH=1 ./install_mac_linux.sh   # keep the already-installed PyTorch
#
# Same pinned versions as the Windows installer; everything stays in this folder.
# On Linux with an NVIDIA GPU the PyPI PyTorch build uses CUDA. On macOS the
# pipeline runs on CPU: choose the 'mediapipe' pose model in the dashboard.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PYTHON_VERSION=3.12
TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
UV_VERSION=0.13.9
SAPIENS_URL="https://github.com/facebookresearch/sapiens2/archive/7e5bae88456ac418ff0e58e74106c9fe192055d4.zip"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python"
export MYOGAIT_MODELS_DIR="$ROOT/models"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
mkdir -p "$ROOT/.cache/bin" "$MYOGAIT_MODELS_DIR"

echo "[1/5] Package manager (uv)..."
UV="$ROOT/.cache/bin/uv"
if [ ! -x "$UV" ]; then
  if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
  else
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$ROOT/.cache/bin" UV_NO_MODIFY_PATH=1 sh
  fi
fi
echo "      $UV"

echo "[2/5] Python $PYTHON_VERSION environment (.venv)..."
PY="$ROOT/.venv/bin/python"
if [ -x "$PY" ] && ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  rm -rf "$ROOT/.venv"
fi
[ -x "$PY" ] || "$UV" venv --python "$PYTHON_VERSION" --seed "$ROOT/.venv"

echo "[3/5] PyTorch $TORCH_VERSION..."
if [ "${SKIP_TORCH:-0}" != "1" ]; then
  "$UV" pip install --python "$PY" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
else
  echo "      skipped (SKIP_TORCH=1)"
fi

echo "[4/5] myogait and the pinned packages (requirements/dcm-lock.txt)..."
"$UV" pip install --python "$PY" --no-deps -r "$ROOT/requirements/dcm-lock.txt"
"$UV" pip install --python "$PY" --no-deps -e "$ROOT"
"$UV" pip install --python "$PY" --no-deps "$SAPIENS_URL" \
  || echo "      WARNING: Meta's Sapiens 2 package could not be installed; Sapiens 2 models will not load."
"$PY" -c "import myogait, streamlit, dcm_pipeline; print('      myogait', myogait.__version__)"

echo "[5/5] GPU check..."
if "$PY" "$ROOT/scripts/check_gpu.py"; then
  echo "      NVIDIA GPU ready. Optional: pre-download Sapiens 2 Quick with"
  echo "      .venv/bin/myogait setup-sapiens2 --size 0.4b --no-install"
else
  echo "      No usable NVIDIA GPU: choose the 'mediapipe' pose model in the dashboard."
fi
echo
echo "Setup complete. Start the dashboard with: ./run_dashboard.sh"
