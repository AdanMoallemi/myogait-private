#!/usr/bin/env bash
# Start the MyoGait DCM dashboard (macOS / Linux). Opens in the browser.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "The software is not installed yet. Run ./install_mac_linux.sh first."
  exit 1
fi
export MYOGAIT_MODELS_DIR="$PWD/models"
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
exec .venv/bin/python -m streamlit run dcm_dashboard.py "$@"
