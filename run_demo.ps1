# Starts the Streamlit demo with the project's own Python environment (.venv), which has all dependencies.
# A plain "streamlit run" may pick up a different, global Python install that lacks them.
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m streamlit run demo\app.py
