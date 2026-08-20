@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo   MyoGait DCM Pipeline - Automated GPU Setup for Windows PC
echo   (NVIDIA RTX 6000 / RTX 3000 / RTX 4000 / Quadro / CUDA Workstations)
echo =====================================================================
echo.

:: 1. Locate Conda (either in PATH or standard install paths)
set "CONDA_CMD="

where conda >nul 2>nul
if %errorlevel% equ 0 (
    set "CONDA_CMD=conda"
) else (
    for %%P in (
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
        "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
        "C:\ProgramData\miniconda3\condabin\conda.bat"
        "C:\ProgramData\anaconda3\condabin\conda.bat"
        "C:\Miniconda3\condabin\conda.bat"
        "C:\Anaconda3\condabin\conda.bat"
        "%USERPROFILE%\AppData\Local\Programs\Miniconda3\condabin\conda.bat"
        "%USERPROFILE%\AppData\Local\Programs\Anaconda3\condabin\conda.bat"
    ) do (
        if not defined CONDA_CMD (
            if exist %%P (
                set "CONDA_CMD=%%~P"
            )
        )
    )
)

if defined CONDA_CMD (
    echo [FOUND] Conda detected at: !CONDA_CMD!
    echo.
    echo [1/5] Creating/updating Conda environment 'dcm-gait' (Python 3.11)...
    call "!CONDA_CMD!" create -y -n dcm-gait python=3.11
    
    echo.
    echo [2/5] Activating 'dcm-gait' environment...
    call "!CONDA_CMD!" activate dcm-gait
) else (
    echo [NOTICE] Conda was not found in PATH or standard install folders.
    echo Checking for standard Python...
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [1/5] Creating standard Python virtual environment 'venv_dcm'...
        python -m venv venv_dcm
        echo.
        echo [2/5] Activating virtual environment...
        call venv_dcm\Scripts\activate.bat
    ) else (
        echo.
        echo =====================================================================
        echo [ERROR] Neither Conda nor Python was found on this system!
        echo.
        echo Quick Solution (takes 2 minutes):
        echo   1. Download Miniconda for Windows:
        echo      https://docs.anaconda.com/miniconda/
        echo   2. Run the installer.
        echo   3. Once installed, open 'Anaconda Prompt' from your Start Menu,
        echo      cd to this folder, and run: install_windows_gpu.bat
        echo =====================================================================
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Installing PyTorch with CUDA 12.4 GPU acceleration...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/5] Installing MyoGait, Dashboard, and Model Dependencies...
pip install -e ".[all]"
pip install streamlit onnxruntime-gpu openpyxl

echo.
echo [5/5] Checking GPU detection and hardware specs...
python -c "import torch; print('--------------------------------------------------'); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU Mode)'); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0); print('--------------------------------------------------')"

echo.
echo =====================================================================
echo   Setup Complete!
echo   To launch the dashboard at any time, double-click 'run_dashboard.bat'
echo =====================================================================
echo.
pause
