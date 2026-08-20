@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo   MyoGait DCM Pipeline - Automated GPU Setup for Windows PC
echo   (NVIDIA RTX 6000 / RTX 3000 / RTX 4000 / Quadro / CUDA Workstations)
echo =====================================================================
echo.

:: 1. Check for Conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'conda' command was not found!
    echo Please install Anaconda or Miniconda, and run this script from the
    echo 'Anaconda Prompt' or ensure conda is added to your system PATH.
    echo.
    pause
    exit /b 1
)

echo [1/5] Creating/updating Conda environment 'dcm-gait' (Python 3.11)...
call conda create -y -n dcm-gait python=3.11
if %errorlevel% neq 0 (
    echo [WARNING] Environment creation returned a non-zero code. Trying to proceed...
)

echo.
echo [2/5] Activating 'dcm-gait' environment...
call conda activate dcm-gait

echo.
echo [3/5] Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/5] Installing MyoGait, Dashboard, and Model Backends...
pip install -e ".[all]"
pip install streamlit onnxruntime-gpu openpyxl

echo.
echo [5/5] Checking GPU detection and hardware specs...
python -c "import torch; print('--------------------------------------------------'); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU Mode)'); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0); print('--------------------------------------------------')"

echo.
echo =====================================================================
echo   Setup Complete!
echo   To launch the dashboard at any time, run 'run_dashboard.bat'
echo =====================================================================
echo.
pause
