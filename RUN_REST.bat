@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Bot Detection - Remaining steps
echo Running steps 6, 7, 8 and 5. Keep this window open.
".venv\Scripts\python.exe" -u run_pipeline.py 6 7 8 5 > output\rest_log.txt 2>&1
echo.
echo Finished. Exit code: %errorlevel%
echo Send the file output\rest_log.txt to Claude.
pause
