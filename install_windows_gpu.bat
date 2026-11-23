@echo off
rem One-step setup of the DCM gait pipeline. Double-click this file.
rem Safe to run again at any time (it reuses what is already installed).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows_gpu.ps1" %*
echo.
pause
