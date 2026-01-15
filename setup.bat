@echo off
REM RideView - Windows Setup Script
REM Run this script to set up the development environment

echo ============================================
echo   RideView - Torque Stripe Verification
echo   Windows Setup Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if uv is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Installing uv package manager...
    pip install uv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install uv
        pause
        exit /b 1
    )
)

echo [OK] uv package manager found
uv --version

echo.
echo [INFO] Creating virtual environment and installing dependencies...
uv sync

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [INFO] Installing dev dependencies...
uv sync --all-extras

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To run RideView:
echo   1. Live Detection:  uv run python -m rideview
echo   2. With Web UI:     uv run python -m rideview --web
echo   3. Color Calibrate: uv run python scripts/calibrate_colors.py
echo.
echo Keyboard shortcuts in live mode:
echo   q - Quit
echo   s - Save snapshot
echo   c - Open color calibration
echo   r - Reset ROI
echo   space - Pause/Resume
echo.
pause
