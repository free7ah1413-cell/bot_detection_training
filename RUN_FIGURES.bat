@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Bot Detection - Figures
echo Creating figures (step 5). Keep this window open.
".venv\Scripts\python.exe" -u run_pipeline.py 5 > output\figures_log.txt 2>&1
echo.
echo Finished. Exit code: %errorlevel%
echo Send the file output\figures_log.txt to Claude.
pause
