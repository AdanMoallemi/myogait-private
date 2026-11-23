@echo off
rem Download the latest version from GitHub and refresh the installed packages.
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed, so this folder cannot update itself.
    echo Either install Git ^(winget install --id Git.Git -e^) and clone the repository,
    echo or download the ZIP again from GitHub and run install_windows_gpu.bat.
    pause
    exit /b 1
)
if not exist "%~dp0.git" (
    echo This folder was downloaded as a ZIP, not cloned with Git, so it cannot update itself.
    echo Download the ZIP again from GitHub and run install_windows_gpu.bat in the new folder.
    pause
    exit /b 1
)

echo Downloading the latest version...
git pull --ff-only
if errorlevel 1 (
    echo.
    echo The update could not be applied ^(local changes or no connection^). Nothing was changed.
    pause
    exit /b 1
)

echo.
echo Refreshing the installed packages...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows_gpu.ps1" -NoShortcut
echo.
pause
