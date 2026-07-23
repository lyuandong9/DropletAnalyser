@echo off
echo ImageIdentification — Droplet/Bubble Analyzer
echo.

REM Try python from PATH first, then common locations
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python main.py
    goto :done
)

if exist "C:\Software\Python3.11\python.exe" (
    "C:\Software\Python3.11\python.exe" main.py
    goto :done
)

echo Python not found. Please install Python 3.11+ and add it to PATH.
:done
pause
