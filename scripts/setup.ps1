param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed. See docs/IMPLEMENTATION_PLAN.md for setup troubleshooting." }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip update failed" }
    & .\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows-cpu.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & .\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed" }
    & .\.venv\Scripts\python.exe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Correctness checks failed" }
} finally { Pop-Location }

