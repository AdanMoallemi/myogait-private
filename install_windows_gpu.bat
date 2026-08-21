@echo off
setlocal

echo =====================================================================
echo   MyoGait DCM Pipeline - Automated GPU Setup for Windows PC
echo   (NVIDIA RTX 6000 / RTX 3000 / RTX 4000 / Quadro / CUDA Workstations)
echo =====================================================================
echo.

:: 1. Force all temporary files onto this drive root (e.g. D:\temp_myogait)
set "TEMP=%~d0\temp_myogait"
set "TMP=%~d0\temp_myogait"
if not exist "%TEMP%" mkdir "%TEMP%" 2>nul

:: 2. Place environment in a spaceless folder on this drive root (e.g. D:\conda_envs\dcm-gait)
set "ENV_DIR=%~d0\conda_envs\dcm-gait"

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
echo [1/5] Creating environment directly on this drive at %ENV_DIR%...
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>nul
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>nul
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>nul
call "%CONDA_CMD%" config --set terms_of_service_consent yes 2>nul
call "%CONDA_CMD%" create -y --prefix "%ENV_DIR%" --override-channels -c conda-forge python=3.11
if %errorlevel% neq 0 (
    echo [RETRY] Retrying with default channels...
    call "%CONDA_CMD%" create -y --prefix "%ENV_DIR%" python=3.11
)
echo.
echo [2/5] Activating environment...
call "%CONDA_CMD%" activate "%ENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment!
    pause
    exit /b 1
)
goto INSTALL_PACKAGES

:CHECK_PYTHON
echo [NOTICE] Conda not found in PATH or standard folders. Checking for Python...
where python >nul 2>nul
if %errorlevel% neq 0 goto NO_PYTHON
echo [1/5] Creating virtual environment at %ENV_DIR%...
python -m venv "%ENV_DIR%"
echo.
echo [2/5] Activating virtual environment...
call "%ENV_DIR%\Scripts\activate.bat"
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
echo [3/5] Installing MyoGait and model dependencies...
pip install -e ".[all]" --no-cache-dir
pip install streamlit onnxruntime-gpu openpyxl --no-cache-dir
pip install git+https://github.com/facebookresearch/sapiens2.git --no-deps --ignore-requires-python



echo.
echo [4/5] Installing PyTorch with CUDA 12.4 GPU acceleration...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-cache-dir

python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] CUDA 12.4 was not recognized by your current NVIDIA driver.
    echo [FALLBACK] Trying CUDA 11.8 compatibility build...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache-dir
)

echo.
echo [5/5] Checking GPU detection and hardware specs...
python -c "import torch; print('--------------------------------------------------'); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU Mode)'); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0); print('--------------------------------------------------')"
echo.
if exist "%TEMP%" rd /s /q "%TEMP%" 2>nul
echo =====================================================================
echo   Setup Complete! Launch dashboard with 'run_dashboard.bat'
echo =====================================================================
echo.
pause
