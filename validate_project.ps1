$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "Validating Forecast Intelligence..." -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Compiling backend..." -ForegroundColor Yellow

python -m py_compile backend\api.py
python -m py_compile backend\prediction_store.py
python -m py_compile backend\live_weather.py
python -m py_compile backend\weather_copilot.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Backend compilation failed." -ForegroundColor Red
    exit 1
}

Write-Host "Backend compilation passed." -ForegroundColor Green

Write-Host "[2/3] Compiling frontend..." -ForegroundColor Yellow

python -m py_compile frontend\app.py

Get-ChildItem `
    frontend\components `
    -Filter "*.py" |
ForEach-Object {
    python -m py_compile $_.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed: $($_.FullName)"
    }
}

Write-Host "Frontend compilation passed." -ForegroundColor Green

Write-Host "[3/3] Running automated tests..." -ForegroundColor Yellow

python -m pytest tests -q

if ($LASTEXITCODE -ne 0) {
    Write-Host "Automated tests failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  PROJECT VALIDATION PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green