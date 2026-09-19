$ErrorActionPreference = "SilentlyContinue"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stopping Forecast Intelligence" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$Ports = @(8000, 8501)

foreach ($Port in $Ports) {
    Write-Host "Checking port $Port..." -ForegroundColor Yellow

    $Connections = Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue

    if (-not $Connections) {
        Write-Host(
            "No active service found on port $Port."
        ) -ForegroundColor Gray

        continue
    }

    $ProcessIds = $Connections |
        Select-Object `
            -ExpandProperty OwningProcess `
            -Unique

    foreach ($ProcessId in $ProcessIds) {
        $Process = Get-Process `
            -Id $ProcessId `
            -ErrorAction SilentlyContinue

        if ($Process) {
            Write-Host(
                "Stopping $($Process.ProcessName) " +
                "(PID $ProcessId) on port $Port..."
            ) -ForegroundColor Yellow

            Stop-Process `
                -Id $ProcessId `
                -Force `
                -ErrorAction SilentlyContinue

            Write-Host(
                "Port $Port stopped successfully."
            ) -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host(
    "Forecast Intelligence services are stopped."
) -ForegroundColor Green