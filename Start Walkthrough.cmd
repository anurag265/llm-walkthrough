@echo off
rem Double-click to present: sets up on first run, starts the walkthrough and opens it in the browser.
rem Keep this window open while presenting; closing it stops the walkthrough.
cd /d "%~dp0"
title LLM Walkthrough - keep open while presenting

if not exist .venv\Scripts\python.exe (
  echo First run on this machine: setting up, needs internet once. This takes a few minutes...
  python -m venv .venv || goto failed
  .venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu || goto failed
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto failed
)

echo Starting. The browser opens by itself once the model is loaded.
.venv\Scripts\python.exe server.py --open || goto failed
exit /b 0

:failed
echo.
echo The walkthrough did not start. Read the message above; it needs Python 3.11 or newer.
pause
