@echo off
echo Starting dots.ocr Web UI...
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py --port 7860

pause
