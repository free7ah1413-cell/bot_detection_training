@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Bot Detection - Step 9
echo Running step 9. Keep this window open.
".venv\Scripts\python.exe" -u run_pipeline.py 9 > output\step9_log.txt 2>&1
echo.
echo Finished. Exit code: %errorlevel%
echo Send the file output\step9_log.txt to Claude.
pause
