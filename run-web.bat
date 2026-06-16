@echo off
REM ============================================================
REM  PodcastCard - launch the web UI
REM  Serves the single-page app at http://localhost:8000
REM ============================================================
setlocal
cd /d "%~dp0"

REM Force UTF-8 so Chinese characters and rich glyphs print without errors.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    exit /b 1
)

echo Starting PodcastCard web server at http://localhost:8000
echo Press Ctrl+C to stop.
call ".venv\Scripts\python.exe" -m uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
endlocal
