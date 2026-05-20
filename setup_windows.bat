@echo off
echo ============================================
echo   dots.ocr - Windows Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or higher is required.
    python --version
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo       Done.
) else (
    echo       Already exists, skipping.
)

echo.
echo [2/4] Activating environment...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies (may take a few minutes)...
pip install torch torchvision torchaudio --quiet
pip install -e . --quiet
echo       Done.

echo.
echo [4/4] Setup complete!
echo.
echo ============================================
echo   Run the app with:   start.bat
echo   Or manually:        .venv\Scripts\activate
echo                       python app.py --port 7860
echo ============================================
pause
