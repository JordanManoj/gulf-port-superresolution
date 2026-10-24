# Re-runs evaluation + port evaluation + figures + stats (offline). Log: results\eval.log
Set-Location "C:\Users\jorda\Downloads\Super Resolution Ports"
$host.UI.RawUI.WindowTitle = "SR evaluation (tile overlap 24 px)"
& .\.venv\Scripts\python.exe -u -W ignore src\evaluate.py --methods bicubic realesrgan satlas ft_l1 ft_gan --save-sr 2>&1 | Tee-Object -FilePath "results\eval.log"
& .\.venv\Scripts\python.exe -u -W ignore src\eval_ports.py --methods bicubic realesrgan satlas ft_l1 ft_gan 2>&1 | Tee-Object -FilePath "results\eval_ports.log"
& .\.venv\Scripts\python.exe -u -W ignore src\stats.py 2>&1 | Tee-Object -FilePath "results\stats.log"
& .\.venv\Scripts\python.exe -u -W ignore src\make_figures.py 2>&1 | Tee-Object -FilePath "results\figures.log"
Write-Host "EVALUATION FINISHED" -ForegroundColor Green
