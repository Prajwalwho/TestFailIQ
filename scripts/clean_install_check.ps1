# Verifies that what is pushed to GitHub installs and passes its tests from scratch.
#   powershell -ExecutionPolicy Bypass -File scripts\clean_install_check.ps1
param([string]$Repo = "https://github.com/Prajwalwho/TestFailIQ.git")

$ErrorActionPreference = "Stop"

function Run([string]$what, [scriptblock]$cmd) {
    Write-Host "==> $what"
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit code $LASTEXITCODE)" }
}

$tmp = Join-Path $env:TEMP ("testfailiq-check-" + [guid]::NewGuid().ToString("N").Substring(0, 8))

Run "clone $Repo" { git clone --depth 1 $Repo $tmp }

Push-Location "$tmp\backend"
Run "create virtualenv" { python -m venv .venv }
Run "install backend dependencies" { .\.venv\Scripts\python -m pip install -r requirements-dev.txt }
Run "backend tests" { .\.venv\Scripts\python -m pytest tests -q }
Pop-Location

Push-Location "$tmp\frontend"
Run "install frontend dependencies" { npm ci }
Run "frontend production build" { npm run build }
Pop-Location

Write-Host "Clean install check PASSED ($tmp)"