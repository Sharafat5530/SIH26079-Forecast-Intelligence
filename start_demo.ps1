$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$BackendUrl = "http://127.0.0.1:8000"
$FrontendUrl = "http://127.0.0.1:8501"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SYNAPSE FUSION" -ForegroundColor White
Write-Host "  Forecast Intelligence Platform" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot

Write-Host "[1/4] Checking required files..." -ForegroundColor Yellow

$RequiredFiles = @(
    "backend\api.py",
    "frontend\app.py",
    "models\xgb.pkl",
    "artifacts\shap_explainer.pkl",
    "artifacts\training_metrics.json"
)

foreach ($File in $RequiredFiles) {
    $FullPath = Join-Path $ProjectRoot $File

    if (-not (Test-Path $FullPath)) {
        Write-Host "Missing file: $File" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
}

Write-Host "Required files found." -ForegroundColor Green

Write-Host "[2/4] Starting FastAPI backend..." -ForegroundColor Yellow

$BackendCommand = @"
Set-Location '$ProjectRoot'
`$Host.UI.RawUI.WindowTitle = 'Forecast Intelligence Backend'
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
"@

Start-Process powershell.exe `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $BackendCommand
    )

Write-Host "Waiting for backend health check..." -ForegroundColor Yellow

$BackendReady = $false

for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
    try {
        $HealthResponse = Invoke-RestMethod `
            -Uri "$BackendUrl/health" `
            -Method Get `
            -TimeoutSec 2

        if ($HealthResponse) {
            $BackendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $BackendReady) {
    Write-Host "Backend did not become ready." -ForegroundColor Red
    Write-Host "Check the backend PowerShell window." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Backend is healthy." -ForegroundColor Green
Write-Host "Checking demonstration records..." -ForegroundColor Yellow

try {
    $HistoryResponse = Invoke-RestMethod `
        -Uri "$BackendUrl/predictions/history?limit=1" `
        -Method Get `
        -TimeoutSec 10

    $HistoryCount = [int]$HistoryResponse.count

    if ($HistoryCount -eq 0) {
        Write-Host(
            "Prediction history is empty. " +
            "Generating model-based demo records..."
        ) -ForegroundColor Yellow

        python -m backend.seed_demo

        if ($LASTEXITCODE -ne 0) {
            Write-Host(
                "Demo records could not be generated. " +
                "The platform will continue without them."
            ) -ForegroundColor Yellow
        }
        else {
            Write-Host(
                "Demonstration records generated successfully."
            ) -ForegroundColor Green
        }
    }
    else {
        Write-Host(
            "Existing prediction records found: $HistoryCount"
        ) -ForegroundColor Green
    }
}
catch {
    Write-Host(
        "Prediction history could not be checked. " +
        "Continuing platform startup."
    ) -ForegroundColor Yellow
}

Write-Host "[3/4] Starting Streamlit frontend..." -ForegroundColor Yellow

$FrontendCommand = @"
Set-Location '$ProjectRoot'
`$Host.UI.RawUI.WindowTitle = 'Forecast Intelligence Frontend'
python -m streamlit run frontend\app.py --server.port 8501
"@

Start-Process powershell.exe `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $FrontendCommand
    )

Write-Host "Waiting for dashboard..." -ForegroundColor Yellow

$FrontendReady = $false

for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
    try {
        $Response = Invoke-WebRequest `
            -Uri $FrontendUrl `
            -UseBasicParsing `
            -TimeoutSec 2

        if ($Response.StatusCode -eq 200) {
            $FrontendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $FrontendReady) {
    Write-Host "Dashboard did not become ready." -ForegroundColor Red
    Write-Host "Check the frontend PowerShell window." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Dashboard is ready." -ForegroundColor Green

Write-Host "[4/4] Opening Forecast Intelligence..." -ForegroundColor Yellow

Start-Process $FrontendUrl

Write-Host ""
Write-Host "Platform launched successfully." -ForegroundColor Green
Write-Host "Backend:  $BackendUrl" -ForegroundColor Gray
Write-Host "Dashboard: $FrontendUrl" -ForegroundColor Gray
Write-Host "API Docs:  $BackendUrl/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "Keep the backend and frontend windows open." -ForegroundColor Cyan