@echo off
setlocal

echo Starting MyoGait Clinical Dashboard...

set "ENV_DIR=%~d0\conda_envs\dcm-gait"
if not exist "%ENV_DIR%" set "ENV_DIR=%~dp0env_dcm"

:: Set HuggingFace and model cache on this drive to prevent C: space exhaustion
set "MYOGAIT_MODELS_DIR=%~dp0models"
set "HF_HOME=%~dp0.cache\huggingface"
set "TORCH_HOME=%~dp0.cache\torch"
if not exist "%MYOGAIT_MODELS_DIR%" mkdir "%MYOGAIT_MODELS_DIR%" 2>nul
if not exist "%HF_HOME%" mkdir "%HF_HOME%" 2>nul

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

if exist "%ENV_DIR%\Scripts\activate.bat" (
    call "%ENV_DIR%\Scripts\activate.bat"
) else if defined CONDA_CMD (
    call "%CONDA_CMD%" activate "%ENV_DIR%"
)

:: Bind to 127.0.0.1 (localhost only) to bypass Windows Firewall prompts without requiring admin privileges
streamlit run dcm_dashboard.py --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
