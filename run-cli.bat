@echo off
REM ============================================================
REM  PodcastCard - run the CLI against a podcast/video URL
REM
REM  Usage:
REM     run-cli.bat "<url>" [extra options...]
REM
REM  Examples:
REM     run-cli.bat "https://youtu.be/abc123"
REM     run-cli.bat "https://youtu.be/abc123" --model base --hsk-levels 4,5,6 --output ./output
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

if "%~1"=="" (
    echo Usage: run-cli.bat "URL" [--model base] [--hsk-levels 4,5,6] [--output ./output]
    exit /b 1
)

REM Single-command Typer app: the URL is passed directly (no "run" subcommand).
call ".venv\Scripts\python.exe" -m src %*
endlocal
