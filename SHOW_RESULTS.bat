@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Bot Detection - Results

if exist ".venv\Scripts\python.exe" goto use_venv
echo The training environment was not found. Run START_TRAINING.bat first.
pause
exit /b 1

:use_venv
".venv\Scripts\python.exe" show_results.py
if errorlevel 1 goto failed
start "" "%CD%\output\results_report.html"
exit /b 0

:failed
echo Could not create the results report.
pause
exit /b 1
