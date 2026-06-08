@echo off
REM ── dots.ocr Kaggle Manager — Build script ───────────────────────────────────
REM Yêu cầu: Python 3.10+ trong PATH, đã pip install -r requirements.txt

echo [1/3] Cai dat phu thuoc...
pip install -r requirements.txt
if errorlevel 1 (
    echo LOI: pip install that bai
    pause
    exit /b 1
)

echo [2/3] Build .exe voi PyInstaller...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "DotsOCR-KaggleManager" ^
    --icon NONE ^
    --add-data "." ^
    main.py

if errorlevel 1 (
    echo LOI: PyInstaller that bai
    pause
    exit /b 1
)

echo [3/3] Hoan thanh!
echo File exe: dist\DotsOCR-KaggleManager.exe
echo.
pause
