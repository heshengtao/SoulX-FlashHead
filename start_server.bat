@echo off
chcp 65001 >nul
title FlashHead Streaming Server

echo ============================================
echo   SoulX-FlashHead Streaming API Server
echo   http://localhost:8765
echo ============================================
echo.

set CUDA_VISIBLE_DEVICES=0

:: Use the conda env python directly
set PYTHON=D:\anaconda\envs\flashhead\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo Please create conda env first: conda create -n flashhead python=3.10
    pause
    exit /b 1
)

cd /d %~dp0

echo Starting server on port 8765...
echo Press Ctrl+C to stop, or close this window.
echo.

"%PYTHON%" streaming_server.py

pause
