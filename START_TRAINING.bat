@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Bot Detection - Training

echo ============================================================
echo   BOT DETECTION TRAINING - PORTABLE PACKAGE
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" goto venv_ready

echo [1/5] Creating the Python environment...
py -3.11 -m venv .venv >nul 2>&1
if exist ".venv\Scripts\python.exe" goto venv_ready
py -3.10 -m venv .venv >nul 2>&1
if exist ".venv\Scripts\python.exe" goto venv_ready
python -m venv .venv >nul 2>&1
if exist ".venv\Scripts\python.exe" goto venv_ready

echo ERROR: Python 3.10 or 3.11 was not found.
echo Install 64-bit Python from https://www.python.org/downloads/
goto failed

:venv_ready
echo [2/5] Checking the dataset and free space...
".venv\Scripts\python.exe" check_ready.py
if errorlevel 1 goto failed

echo.
echo [3/5] Installing required Python packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed

echo.
echo [4/5] Running all data preparation and training stages...
echo Keep this window open. The 3.2 GB dataset can take a long time.
".venv\Scripts\python.exe" -u run_pipeline.py
if errorlevel 1 goto training_failed

echo.
echo [5/5] Validating outputs and building the HTML report...
".venv\Scripts\python.exe" validate_pipeline.py
if errorlevel 1 goto validation_failed
".venv\Scripts\python.exe" show_results.py
if errorlevel 1 goto failed
start "" "%CD%\output\results_report.html"

echo.
echo ============================================================
echo   TRAINING COMPLETED SUCCESSFULLY
echo   Report: output\results_report.html
echo ============================================================
pause
exit /b 0

:training_failed
echo.
echo Training stopped because one stage failed.
echo Existing completed outputs were kept. Run this file again after fixing the error.
".venv\Scripts\python.exe" show_results.py >nul 2>&1
goto failed

:validation_failed
echo.
echo Training finished, but output validation failed.
echo Do not use the results until the validation error is reviewed.
".venv\Scripts\python.exe" show_results.py >nul 2>&1
goto failed

:failed
echo.
echo PROCESS NOT COMPLETED. Read the error shown above.
pause
exit /b 1
