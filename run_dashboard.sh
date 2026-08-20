#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dcm-gait
streamlit run dcm_dashboard.py
