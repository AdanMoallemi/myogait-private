@echo off
echo Starting MyoGait Clinical Dashboard...
call conda activate dcm-gait
streamlit run dcm_dashboard.py
pause
