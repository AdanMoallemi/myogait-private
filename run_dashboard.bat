@echo off
setlocal enabledelayedexpansion

echo Starting MyoGait Clinical Dashboard...

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
    call "!CONDA_CMD!" activate dcm-gait
) else if exist "venv_dcm\Scripts\activate.bat" (
    call venv_dcm\Scripts\activate.bat
)

streamlit run dcm_dashboard.py
pause
