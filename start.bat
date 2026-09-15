@echo off
rem Start FlyCraft: the fly brain server for Minecraft Education, and its dashboard.
rem Double-click, or run from a terminal with extra options, e.g.  start.bat --seek oak_log
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Could not find %PYTHON%.
    echo Set up the virtual environment first, as described in README.md.
    pause
    exit /b 1
)

if not exist "data\calibration.json" (
    echo No calibration found. Calibrating the brain first; this runs once.
    "%PYTHON%" -m flyminecraft.calibrate
    if errorlevel 1 (
        echo Calibration failed.
        pause
        exit /b 1
    )
)

rem Open the dashboard once the server answers (it takes a few seconds to load the connectome)
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest http://localhost:8081/ -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process http://localhost:8081; break } catch { Start-Sleep 1 } }"

echo In Minecraft Education, type  /connect localhost:8080  in the chat.
echo Press Ctrl+C to stop the server.
echo.
"%PYTHON%" -m flyminecraft %*

echo.
echo The server has stopped.
pause
