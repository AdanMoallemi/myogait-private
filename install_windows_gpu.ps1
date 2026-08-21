# MyoGait Automated GPU Installer for Windows (PowerShell)
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  MyoGait DCM Pipeline - Automated GPU Setup for Windows PC" -ForegroundColor Cyan
Write-Host "  (NVIDIA RTX 6000 / RTX 3000 / RTX 4000 / Quadro / CUDA Workstations)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$driveRoot = (Get-Item $PSScriptRoot).PSDrive.Root
$tmpDir = Join-Path $driveRoot "temp_myogait"
if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }
$env:TEMP = $tmpDir
$env:TMP = $tmpDir

# Spaceless environment directory on drive root (e.g. D:\conda_envs\dcm-gait)
$envDir = Join-Path $driveRoot "conda_envs\dcm-gait"

$condaCmd = $null
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $condaCmd = "conda"
} else {
    $candidatePaths = @(
        "$env:USERPROFILE\miniconda3\condabin\conda.bat",
        "$env:USERPROFILE\anaconda3\condabin\conda.bat",
        "$env:LOCALAPPDATA\miniconda3\condabin\conda.bat",
        "$env:LOCALAPPDATA\anaconda3\condabin\conda.bat",
        "C:\ProgramData\miniconda3\condabin\conda.bat",
        "C:\ProgramData\anaconda3\condabin\conda.bat",
        "C:\Miniconda3\condabin\conda.bat",
        "C:\Anaconda3\condabin\conda.bat",
        "$env:USERPROFILE\AppData\Local\Programs\Miniconda3\condabin\conda.bat",
        "$env:USERPROFILE\AppData\Local\Programs\Anaconda3\condabin\conda.bat"
    )
    foreach ($p in $candidatePaths) {
        if (Test-Path $p) {
            $condaCmd = $p
            break
        }
    }
}

if ($condaCmd) {
    Write-Host "[FOUND] Conda detected: $condaCmd" -ForegroundColor Green
    Write-Host ""
    Write-Host "[1/5] Creating environment on drive at $envDir..." -ForegroundColor Yellow
    & $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
    & $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null
    & $condaCmd tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null
    & $condaCmd config --set terms_of_service_consent yes 2>$null
    & $condaCmd create -y --prefix "$envDir" --override-channels -c conda-forge python=3.11

    Write-Host ""
    Write-Host "[2/5] Initializing conda environment..." -ForegroundColor Yellow
    & $condaCmd "shell.powershell" "hook" | Out-String | Invoke-Expression
    conda activate "$envDir"
} else {
    Write-Host "[NOTICE] Conda not found. Checking for Python..." -ForegroundColor Yellow
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "[1/5] Creating virtual environment at $envDir..." -ForegroundColor Yellow
        python -m venv "$envDir"
        Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
        $actScript = Join-Path $envDir "Scripts\Activate.ps1"
        & $actScript
    } else {
        Write-Host ""
        Write-Host "[ERROR] Neither Conda nor Python was found on this system!" -ForegroundColor Red
        Write-Host "Please install Miniconda for Windows: https://docs.anaconda.com/miniconda/" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[3/5] Installing MyoGait and model dependencies..." -ForegroundColor Yellow
pip install -e ".[all]" --no-cache-dir
pip install streamlit onnxruntime-gpu openpyxl --no-cache-dir
pip install git+https://github.com/facebookresearch/sapiens2.git --no-deps


Write-Host ""
Write-Host "[4/5] Installing PyTorch with CUDA 12.4 GPU acceleration on this drive..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-cache-dir

$isCudaOk = python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[NOTICE] CUDA 12.4 was not recognized by your current NVIDIA driver." -ForegroundColor Yellow
    Write-Host "[FALLBACK] Trying CUDA 11.8 compatibility build..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache-dir
}

Write-Host ""
Write-Host "[5/5] Checking GPU detection and hardware specs..." -ForegroundColor Yellow
python -c "import torch; print('--------------------------------------------------'); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU Mode)'); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0); print('--------------------------------------------------')"

if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "  Setup Complete! To launch the dashboard, run: .\run_dashboard.ps1" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Read-Host "Press Enter to finish"
