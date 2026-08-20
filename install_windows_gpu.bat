@echo off
setlocal

echo =====================================================================
echo   MyoGait DCM Pipeline - Automated GPU Setup for Windows PC
echo   (NVIDIA RTX 6000 / RTX 3000 / RTX 4000 / Quadro / CUDA Workstations)
echo =====================================================================
echo.

set "CONDA_CMD="
where conda >nul 2>nul
if %errorlevel% equ 0 set "CONDA_CMD=conda"

if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_CMD=C:\ProgramData\miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_CMD=C:\ProgramData\anaconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "C:\Miniconda3\condabin\conda.bat" set "CONDA_CMD=C:\Miniconda3\condabin\conda.bat"
if not defined CONDA_CMD if exist "C:\Anaconda3\condabin\conda.bat" set "CONDA_CMD=C:\Anaconda3\condabin\conda.bat"

if defined CONDA_CMD goto USE_CONDA
goto CHECK_PYTHON

:USE_CONDA
echo [FOUND] Conda detected at: %CONDA_CMD%
echo.
echo [1/5] Creating or updating Conda environment 'dcm-gait' (Python 3.11)...
call "%CONDA_CMD%" tos accept --all 2>nul
call "%CONDA_CMD%" create -y -n dcm-gait -c conda-forge python=3.11
if %errorlevel% neq 0 (
    echo [RETRY] Retrying with default channels...
    call "%CONDA_CMD%" create -y -n dcm-gait python=3.11
)
echo.
echo [2/5] Activating 'dcm-gait' environment...
call "%CONDA_CMD%" activate dcm-gait
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate dcm-gait environment!
    pause
    exit /b 1
)
goto INSTALL_PACKAGES

:CHECK_PYTHON
echo [NOTICE] Conda not found in PATH or standard folders. Checking for Python...
where python >nul 2>nul
if %errorlevel% neq 0 goto NO_PYTHON
echo [1/5] Creating virtual environment 'venv_dcm'...
python -m venv venv_dcm
echo.
echo [2/5] Activating virtual environment...
call venv_dcm\Scripts\activate.bat
goto INSTALL_PACKAGES

:NO_PYTHON
echo.
echo =====================================================================
echo [ERROR] Neither Conda nor Python was found on this system!
echo.
echo Please install Miniconda for Windows:
echo https://docs.anaconda.com/miniconda/
echo =====================================================================
echo.
pause
exit /b 1

:INSTALL_PACKAGES
echo.
echo [3/5] Installing PyTorch with CUDA 12.4 GPU acceleration...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.
echo [4/5] Installing MyoGait, Dashboard, and Dependencies...
pip install -e ".[all]"
pip install streamlit onnxruntime-gpu openpyxl
echo.
echo [5/5] Checking GPU detection and hardware specs...
python -c "import torch; print('--------------------------------------------------'); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU Mode)'); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0); print('--------------------------------------------------')"
echo.
echo =====================================================================
echo   Setup Complete! Launch dashboard with 'run_dashboard.bat'
echo =====================================================================
echo.
pause
