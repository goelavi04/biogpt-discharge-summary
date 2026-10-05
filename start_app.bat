@echo off
REM Starts the local Discharge Summary Drafter at http://127.0.0.1:8000
REM First run: creates .venv and installs requirements (needs torch already installed
REM system-wide, see README), then downloads BioGPT base weights (~1.6 GB) once.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv --system-site-packages .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

if not exist "checkpoints\lora_adapter\adapter_config.json" (
    echo.
    echo No trained model found at checkpoints\lora_adapter\
    echo Unzip biogpt_results.zip from the last Colab notebook cell into this folder first.
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:"127.0.0.1:8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo The app is already running - opening it in your browser.
    start http://127.0.0.1:8000
    timeout /t 3 >nul
    exit /b 0
)

set USE_TF=0
set OPEN_BROWSER=1
echo Starting... your browser opens automatically once the model is loaded.
echo Keep this window open while using the app; close it to stop the app.
echo.
".venv\Scripts\python.exe" app\server.py
pause
exit /b 0

:error
echo Setup failed - see the message above.
pause
exit /b 1
