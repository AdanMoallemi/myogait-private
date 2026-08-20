# Launch MyoGait Clinical Dashboard (PowerShell)
$driveRoot = (Get-Item $PSScriptRoot).PSDrive.Root
$envDir = Join-Path $driveRoot "conda_envs\dcm-gait"
if (-not (Test-Path $envDir)) {
    $envDir = Join-Path $PSScriptRoot "env_dcm"
}

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

streamlit run dcm_dashboard.py
