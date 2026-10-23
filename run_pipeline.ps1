# Runs the full training + evaluation pipeline in one visible window (offline, sequential). Log: results\pipeline.log
Set-Location "C:\Users\jorda\Downloads\Super Resolution Ports"
$host.UI.RawUI.WindowTitle = "SR pipeline - training + evaluation"
& .\.venv\Scripts\python.exe -u -W ignore src\run_all.py 2>&1 | Tee-Object -FilePath "results\pipeline.log"
Write-Host "Finished. You can close this window." -ForegroundColor Green
