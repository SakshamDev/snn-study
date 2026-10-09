param(
    [ValidateSet("check", "smoke", "pilot", "main-train", "main-validate", "main-test")]
    [string]$Action = "check"
)
$ErrorActionPreference = "Stop"
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$validationPython = Join-Path $PSScriptRoot '..\..\work\snn-venv\Scripts\python.exe'
$validationTestRunner = Join-Path $PSScriptRoot '..\..\work\validate_tests.py'
$usesValidationEnvironment = $false
if (Test-Path -LiteralPath $projectPython) { $runner = $projectPython }
elseif (Test-Path -LiteralPath $validationPython) {
    $runner = (Resolve-Path -LiteralPath $validationPython).Path
    $usesValidationEnvironment = $true
}
else { throw 'Run scripts\setup.ps1 first to create the Python environment.' }
function Invoke-ResearchPython {
    param([string[]]$Arguments)
    & $runner @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($Arguments -join ' ')" }
}
Push-Location $PSScriptRoot
try {
    if (-not $env:MPLCONFIGDIR) {
        $env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'runs\.matplotlib-cache'
        New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
    }
    switch ($Action) {
        'check' {
            if ($usesValidationEnvironment -and (Test-Path -LiteralPath $validationTestRunner)) {
                Invoke-ResearchPython -Arguments @($validationTestRunner, '-q', '--basetemp', 'runs/.pytest-temp')
            } else { Invoke-ResearchPython -Arguments @('-m', 'pytest', '-q') }
        }
        'smoke' { Invoke-ResearchPython -Arguments @('scripts/smoke_test.py') }
        'pilot' {
            Invoke-ResearchPython -Arguments @('scripts/prepare_data.py', '--config', 'configs/pilot.json', '--download')
            Invoke-ResearchPython -Arguments @('scripts/run_grid.py', '--config', 'configs/pilot.json', '--stage', 'train')
            Invoke-ResearchPython -Arguments @('scripts/run_grid.py', '--config', 'configs/pilot.json', '--stage', 'validate')
            Invoke-ResearchPython -Arguments @('scripts/analyze.py', '--config', 'configs/pilot.json', '--split', 'validation')
        }
        'main-train' {
            Invoke-ResearchPython -Arguments @('scripts/prepare_data.py', '--config', 'configs/main.json', '--download')
            Invoke-ResearchPython -Arguments @('scripts/freeze_protocol.py', '--config', 'configs/main.json')
            Invoke-ResearchPython -Arguments @('scripts/run_grid.py', '--config', 'configs/main.json', '--stage', 'train')
        }
        'main-validate' {
            Invoke-ResearchPython -Arguments @('scripts/run_grid.py', '--config', 'configs/main.json', '--stage', 'validate')
            Invoke-ResearchPython -Arguments @('scripts/analyze.py', '--config', 'configs/main.json', '--split', 'validation')
        }
        'main-test' {
            Invoke-ResearchPython -Arguments @('scripts/select.py', '--config', 'configs/main.json')
            Invoke-ResearchPython -Arguments @('scripts/run_grid.py', '--config', 'configs/main.json', '--stage', 'test')
            Invoke-ResearchPython -Arguments @('scripts/analyze.py', '--config', 'configs/main.json', '--split', 'test')
        }
    }
} finally { Pop-Location }
