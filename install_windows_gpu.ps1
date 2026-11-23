<#
.SYNOPSIS
    One-step setup of the DCM gait pipeline on Windows (NVIDIA GPU, or CPU).

.DESCRIPTION
    Double-click install_windows_gpu.bat, which runs this script. Nothing has to
    be installed beforehand: the script downloads its own Python, picks the PyTorch
    build that matches the NVIDIA driver, installs the exact package versions of
    requirements\dcm-lock.txt, pre-downloads the Sapiens 2 model and puts a
    "MyoGait DCM Dashboard" shortcut on the desktop.

    Everything (Python, packages, caches, model weights) stays inside this folder,
    on this drive. Re-running is safe and fast; it is also how dependencies are
    refreshed after an update (update_windows.bat does it for you).

.PARAMETER Cuda
    auto (default), cu128, cu126, cu118 or cpu: force a PyTorch build.
.PARAMETER SkipModels
    Do not pre-download the Sapiens 2 model (it then downloads on first use).
.PARAMETER NoShortcut
    Do not create the desktop shortcut.
#>
param(
    [ValidateSet("auto", "cu128", "cu126", "cu118", "cpu")]
    [string]$Cuda = "auto",
    [switch]$SkipModels,
    [switch]$NoShortcut
)

# Native tools write progress to stderr; never treat that as a failure.
# Every native call is checked through its exit code instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Pinned versions --------------------------------------------------------
$PythonVersion      = "3.12"
$TorchVersion       = "2.7.1"
$TorchvisionVersion = "0.22.1"
$UvVersion          = "0.13.9"
$SapiensUrl         = "https://github.com/facebookresearch/sapiens2/archive/7e5bae88456ac418ff0e58e74106c9fe192055d4.zip"

# Paths: everything stays inside this folder -----------------------------
$Root  = $PSScriptRoot
Set-Location -LiteralPath $Root
$Cache = Join-Path $Root ".cache"
$Venv  = Join-Path $Root ".venv"
$Py    = Join-Path $Venv "Scripts\python.exe"
$Log   = Join-Path $Root "install_log.txt"
foreach ($d in @($Cache, (Join-Path $Root ".tmp"), (Join-Path $Root "models"))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$env:UV_CACHE_DIR          = Join-Path $Cache "uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Cache "python"
$env:UV_LINK_MODE          = "copy"
$env:TEMP                  = Join-Path $Root ".tmp"
$env:TMP                   = $env:TEMP
$env:MYOGAIT_MODELS_DIR    = Join-Path $Root "models"
$env:HF_HOME               = Join-Path $Cache "huggingface"
$env:TORCH_HOME            = Join-Path $Cache "torch"
$env:PYTHONUTF8            = "1"

Set-Content -LiteralPath $Log -Value "DCM pipeline setup log - $(Get-Date -Format s)" -Encoding UTF8
$Started = Get-Date
$Total = 7

function Write-Log([string]$text) { Add-Content -LiteralPath $Log -Value $text -Encoding UTF8 }
function Step([int]$n, [string]$msg) {
    Write-Host ""; Write-Host "[$n/$Total] $msg" -ForegroundColor Cyan; Write-Log "[$n/$Total] $msg"
}
function Info([string]$msg) { Write-Host "      $msg"; Write-Log "      $msg" }
function Warn([string]$msg) { Write-Host "      WARNING: $msg" -ForegroundColor Yellow; Write-Log "      WARNING: $msg" }
function Fail([string]$msg) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "Nothing is broken: fix the problem above and double-click install_windows_gpu.bat again." -ForegroundColor Yellow
    Write-Host "Full log: $Log" -ForegroundColor Yellow
    Write-Log "SETUP FAILED: $msg"
    exit 1
}
# Run a native command, show + log its output, return $true on exit code 0.
function Run([string]$exe, [string[]]$arguments) {
    Write-Log ">> $exe $($arguments -join ' ')"
    & $exe @arguments 2>&1 | ForEach-Object { $line = "$_"; Write-Host "      $line"; Write-Log $line }
    return ($LASTEXITCODE -eq 0)
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  MyoGait DCM Pipeline - one-step setup for Windows" -ForegroundColor Cyan
Write-Host "  Folder: $Root" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Files from a downloaded ZIP carry an "internet" mark that triggers warnings.
foreach ($dir in @($Root, (Join-Path $Root "scripts"))) {
    Get-ChildItem -Path (Join-Path $dir "*") -Include *.bat, *.ps1, *.py -File -ErrorAction SilentlyContinue |
        Unblock-File -ErrorAction SilentlyContinue
}

if ($Root.Length -gt 80) {
    Warn "This folder path is long ($($Root.Length) characters). Windows limits paths to 260 characters,"
    Warn "which some packages exceed. If the install fails, move the folder to e.g. C:\myogait-private."
}

# 1. GPU -----------------------------------------------------------------
Step 1 "Checking the graphics card..."
$Smi = $null
$cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cmd) { $Smi = $cmd.Source }
elseif (Test-Path "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") {
    $Smi = "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
}
$GpuName = $null; $DriverMajor = 0; $DriverText = ""
if ($Smi) {
    $line = & $Smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    if ($line) {
        $parts = @("$line" -split ",")
        if ($parts.Count -ge 2) {
            $GpuName = $parts[0].Trim()
            $DriverText = $parts[1].Trim()
            [int]::TryParse(($DriverText -split "\.")[0], [ref]$DriverMajor) | Out-Null
        }
    }
}
if ($Cuda -ne "auto") {
    $Candidates = @($Cuda)
    Info "PyTorch build forced to: $Cuda"
} elseif (-not $GpuName) {
    Warn "No NVIDIA GPU found (nvidia-smi not available). Installing the CPU version:"
    Warn "use the 'mediapipe' pose model, Sapiens 2 would be very slow on CPU."
    Warn "If this PC has an NVIDIA card, install its driver from nvidia.com/drivers and run setup again."
    $Candidates = @("cpu")
} else {
    Info "GPU: $GpuName  |  NVIDIA driver: $DriverText"
    if ($DriverMajor -ge 570)     { $Candidates = @("cu128", "cu126", "cu118") }
    elseif ($DriverMajor -ge 528) { $Candidates = @("cu126", "cu118") }
    elseif ($DriverMajor -ge 452) { $Candidates = @("cu118") }
    else {
        Warn "NVIDIA driver $DriverText is too old for PyTorch. Update it from nvidia.com/drivers."
        $Candidates = @()
    }
    $Candidates += "cpu"
}

# 2. uv (downloads Python and packages) ---------------------------------
Step 2 "Getting the package manager (uv $UvVersion)..."
$UvDir = Join-Path $Cache "bin"
$Uv = Join-Path $UvDir "uv.exe"
if (-not (Test-Path $Uv)) {
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $zip = Join-Path $env:TEMP "uv.zip"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        Expand-Archive -LiteralPath $zip -DestinationPath $UvDir -Force
        Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    } catch {
        Warn "Direct download failed ($($_.Exception.Message)); trying the official installer..."
        try {
            $env:UV_INSTALL_DIR = $UvDir; $env:UV_NO_MODIFY_PATH = "1"
            Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
        } catch { Warn "Official installer failed: $($_.Exception.Message)" }
    }
    if (-not (Test-Path $Uv)) {
        $found = Get-ChildItem -LiteralPath $UvDir -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { Copy-Item $found.FullName $Uv -Force }
    }
    if (-not (Test-Path $Uv)) { Fail "Could not download uv. Check the internet connection (github.com must be reachable)." }
}
Info "uv ready: $Uv"

# 3. Python environment --------------------------------------------------
Step 3 "Creating the Python $PythonVersion environment (.venv)..."
if (Test-Path $Py) {
    & $Py -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Info "Existing environment has another Python version; recreating it."
        Remove-Item -LiteralPath $Venv -Recurse -Force
    }
}
if (-not (Test-Path $Py)) {
    if (-not (Run $Uv @("venv", "--python", $PythonVersion, "--seed", $Venv))) {
        Fail "Could not create the Python environment."
    }
} else { Info "Environment already exists; reusing it." }

# 4. PyTorch matching the GPU driver ------------------------------------
Step 4 "Installing PyTorch $TorchVersion for this GPU (about 2.5 GB the first time)..."
$Chosen = $null; $GpuReport = $null
foreach ($tag in $Candidates) {
    $want = "$TorchVersion+$tag"
    $have = (& $Py -c "import torch; print(torch.__version__)" 2>$null | Select-Object -First 1)
    if ("$have" -ne $want) {
        Info "Installing PyTorch build $tag..."
        $ok = Run $Uv @("pip", "install", "--python", $Py,
                        "--index-url", "https://download.pytorch.org/whl/$tag",
                        "--reinstall-package", "torch", "--reinstall-package", "torchvision",
                        "torch==$TorchVersion", "torchvision==$TorchvisionVersion")
        if (-not $ok) { Warn "PyTorch $tag could not be installed; trying the next option."; continue }
    } else { Info "PyTorch $want already installed." }
    if ($tag -eq "cpu") { $Chosen = "cpu"; break }
    $GpuReport = & $Py (Join-Path $Root "scripts\check_gpu.py") 2>$null | Select-Object -Last 1
    if ($LASTEXITCODE -eq 0) { $Chosen = $tag; Info "GPU test passed: $GpuReport"; break }
    Warn "PyTorch $tag cannot run on this GPU ($GpuReport); trying the next option."
}
if (-not $Chosen) { Fail "No PyTorch build could be installed." }
if ($Chosen -eq "cpu" -and $Candidates[0] -ne "cpu") {
    Warn "Falling back to CPU: the GPU could not be used. Update the NVIDIA driver and run setup again."
}

# 5. myogait and the exact package versions -----------------------------
Step 5 "Installing myogait and the pinned packages (requirements\dcm-lock.txt)..."
if (-not (Run $Uv @("pip", "install", "--python", $Py, "--no-deps", "-r", (Join-Path $Root "requirements\dcm-lock.txt")))) {
    Fail "Package installation failed."
}
if (-not (Run $Uv @("pip", "install", "--python", $Py, "--no-deps", "-e", $Root))) { Fail "myogait installation failed." }
if (-not (Run $Uv @("pip", "install", "--python", $Py, "--no-deps", $SapiensUrl))) {
    Warn "Meta's Sapiens 2 package could not be installed; Sapiens 2 models will not load."
}
$have = (& $Py -c "import torch; print(torch.__version__)" 2>$null | Select-Object -First 1)
if ("$have" -ne "$TorchVersion+$Chosen") { Fail "PyTorch changed unexpectedly during package installation ($have)." }
& $Py -c "import myogait, streamlit, dcm_pipeline; print('myogait', myogait.__version__)" 2>&1 | ForEach-Object { Info "$_" }
if ($LASTEXITCODE -ne 0) { Fail "The pipeline does not import correctly (see the log)." }

# 6. Sapiens 2 model -----------------------------------------------------
Step 6 "Preparing the Sapiens 2 Quick pose model (myogait's default)..."
if ($SkipModels) { Info "Skipped (-SkipModels): it will download on first use." }
elseif ($Chosen -eq "cpu") { Info "Skipped on CPU: use the 'mediapipe' model, which downloads automatically." }
else {
    Info "Downloading the weights and building the GPU-optimised copy (1-5 minutes)..."
    $myogaitExe = Join-Path $Venv "Scripts\myogait.exe"
    if (Run $myogaitExe @("setup-sapiens2", "--size", "0.4b", "--no-install")) { Info "Sapiens 2 Quick is ready." }
    else { Warn "Model preparation failed; it will be retried automatically on the first analysis." }
}

# 7. Streamlit settings and desktop shortcut -----------------------------
Step 7 "Final touches..."
$stDir = Join-Path $env:USERPROFILE ".streamlit"
$cred = Join-Path $stDir "credentials.toml"
if (-not (Test-Path $cred)) {
    New-Item -ItemType Directory -Force -Path $stDir | Out-Null
    Set-Content -LiteralPath $cred -Value "[general]`nemail = `"`"" -Encoding ASCII  # skips Streamlit's first-run e-mail prompt
}
if (-not $NoShortcut) {
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        $shell = New-Object -ComObject WScript.Shell
        $lnk = $shell.CreateShortcut((Join-Path $desktop "MyoGait DCM Dashboard.lnk"))
        $lnk.TargetPath = Join-Path $Root "run_dashboard.bat"
        $lnk.WorkingDirectory = $Root
        $lnk.Description = "Start the MyoGait DCM gait analysis dashboard"
        $lnk.Save()
        Info "Desktop shortcut created: 'MyoGait DCM Dashboard'."
    } catch { Warn "Could not create the desktop shortcut: $($_.Exception.Message)" }
}

$mins = [math]::Round(((Get-Date) - $Started).TotalMinutes, 1)
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "  Setup complete in $mins min." -ForegroundColor Green
if ($Chosen -eq "cpu") { Write-Host "  Mode: CPU (no usable NVIDIA GPU) - choose the 'mediapipe' model." -ForegroundColor Yellow }
else { Write-Host "  Mode: GPU - $GpuName (PyTorch $TorchVersion+$Chosen)" -ForegroundColor Green }
Write-Host "  Start the dashboard: double-click 'MyoGait DCM Dashboard' on the desktop" -ForegroundColor Green
Write-Host "  (or run_dashboard.bat in this folder)." -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Log "Setup complete: $Chosen, $GpuName, $mins min"
exit 0
