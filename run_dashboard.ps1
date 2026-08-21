# Launch MyoGait Clinical Dashboard (PowerShell)
$driveRoot = (Get-Item $PSScriptRoot).PSDrive.Root
$envDir = Join-Path $driveRoot "conda_envs\dcm-gait"
if (-not (Test-Path $envDir)) {
    $envDir = Join-Path $PSScriptRoot "env_dcm"
}

# Set HuggingFace and model cache on this drive to prevent C: space exhaustion
$env:MYOGAIT_MODELS_DIR = Join-Path $PSScriptRoot "models"
$env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $PSScriptRoot ".cache\torch"
if (-not (Test-Path $env:MYOGAIT_MODELS_DIR)) { New-Item -ItemType Directory -Path $env:MYOGAIT_MODELS_DIR | Out-Null }
if (-not (Test-Path $env:HF_HOME)) { New-Item -ItemType Directory -Path $env:HF_HOME | Out-Null }

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
    & $condaCmd "shell.powershell" "hook" | Out-String | Invoke-Expression
    conda activate "$envDir"
} elseif (Test-Path (Join-Path $envDir "Scripts\Activate.ps1")) {
    $actScript = Join-Path $envDir "Scripts\Activate.ps1"
    & $actScript
}

# Bind to 127.0.0.1 (localhost only) to bypass Windows Firewall prompts without requiring admin privileges
streamlit run dcm_dashboard.py --server.address 127.0.0.1 --browser.gatherUsageStats false
