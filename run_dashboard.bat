@echo off
rem Start the MyoGait DCM dashboard. It opens in the web browser.
rem Keep this window open while you use the dashboard; close it to stop.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo The software is not installed yet. Double-click install_windows_gpu.bat first.
    pause
    exit /b 1
)

rem Keep temporary files, model weights and caches on this drive.
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "MYOGAIT_MODELS_DIR=%~dp0models"
set "HF_HOME=%~dp0.cache\huggingface"
set "TORCH_HOME=%~dp0.cache\torch"
set "PYTHONUTF8=1"
if not exist "%TEMP%" mkdir "%TEMP%"

rem Skip Streamlit's one-time e-mail question.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    > "%USERPROFILE%\.streamlit\credentials.toml" (
        echo [general]
        echo email = ""
    )
)

echo Starting the MyoGait DCM dashboard... it will open in your browser.
echo Keep this window open while you work. Close it to stop the dashboard.
"%PY%" -m streamlit run dcm_dashboard.py
echo.
pause
