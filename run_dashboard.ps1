# Launch MyoGait Clinical Dashboard (PowerShell)
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
    conda activate dcm-gait
} elseif (Test-Path "venv_dcm\Scripts\Activate.ps1") {
    .\venv_dcm\Scripts\Activate.ps1
}

streamlit run dcm_dashboard.py
