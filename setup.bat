@echo off
REM ============================================================
REM  PodcastCard - one-time setup
REM  Creates a virtual environment and installs dependencies.
REM  Requires: Python 3.10-3.13 x64 and FFmpeg on PATH.
REM
REM  NOTE: This is a Windows-on-ARM machine. The transcription
REM  stack (faster-whisper -> ctranslate2, av) ships only x64
REM  win_amd64 wheels, not win_arm64. So the venv MUST be built
REM  with an x64 Python; it runs fine under emulation. This
REM  script locates an x64 interpreter automatically.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYEXE="
set "TMPCHK=%TEMP%\pc_plat_%RANDOM%.txt"

REM --- Candidate x64 interpreters, in priority order ---
set "C1=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
set "C2=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "C3=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set "C4=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

for %%C in ("!C1!" "!C2!" "!C3!" "!C4!") do call :try_path "%%~C"

REM --- Fallback: 'python' on PATH, only if it reports x64 ---
if not defined PYEXE call :try_cmd python

if not defined PYEXE goto :no_python

echo Using x64 Python: !PYEXE!

if exist ".venv\Scripts\python.exe" goto :have_venv
echo Creating virtual environment in .venv ...
"!PYEXE!" -m venv .venv
if errorlevel 1 goto :venv_failed

:have_venv
REM --- Guard: ensure the existing venv is actually x64 ---
set "VENVPLAT="
".venv\Scripts\python.exe" -c "import sysconfig;print(sysconfig.get_platform())" > "%TMPCHK%" 2>nul
if exist "%TMPCHK%" set /p VENVPLAT=<"%TMPCHK%"
if /i not "!VENVPLAT!"=="win-amd64" goto :wrong_venv

echo Upgrading pip ...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip

echo Installing dependencies from requirements.txt ...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed

where ffmpeg >nul 2>&1
if errorlevel 1 call :warn_ffmpeg

del "%TMPCHK%" >nul 2>&1
echo.
echo Setup complete. Use run-web.bat or run-cli.bat to start.
endlocal
exit /b 0

REM ============================================================
REM  Subroutines
REM ============================================================

:try_path
REM %1 = full path to a python.exe; sets PYEXE if it is win-amd64
if defined PYEXE goto :eof
if not exist "%~1" goto :eof
set "PLAT="
"%~1" -c "import sysconfig;print(sysconfig.get_platform())" > "%TMPCHK%" 2>nul
if exist "%TMPCHK%" set /p PLAT=<"%TMPCHK%"
if /i "!PLAT!"=="win-amd64" set "PYEXE=%~1"
goto :eof

:try_cmd
REM %1 = command name on PATH; sets PYEXE if it is win-amd64
if defined PYEXE goto :eof
set "PLAT="
%~1 -c "import sysconfig;print(sysconfig.get_platform())" > "%TMPCHK%" 2>nul
if exist "%TMPCHK%" set /p PLAT=<"%TMPCHK%"
if /i "!PLAT!"=="win-amd64" set "PYEXE=%~1"
goto :eof

:warn_ffmpeg
echo.
echo [WARNING] FFmpeg was not found on PATH. Audio download/decoding will
echo           fail until FFmpeg is installed and available on PATH.
echo           Install with: winget install Gyan.FFmpeg
goto :eof

REM ============================================================
REM  Error exits
REM ============================================================

:no_python
del "%TMPCHK%" >nul 2>&1
echo [ERROR] Could not find an x64 win-amd64 Python interpreter.
echo         The transcription dependencies have no Windows ARM64 wheels,
echo         so an x64 Python is required; it runs under emulation.
echo         Install the 64-bit Windows build from:
echo         https://www.python.org/downloads/windows/
endlocal
exit /b 1

:venv_failed
del "%TMPCHK%" >nul 2>&1
echo [ERROR] Failed to create virtual environment.
endlocal
exit /b 1

:wrong_venv
del "%TMPCHK%" >nul 2>&1
echo [ERROR] Existing .venv is "!VENVPLAT!", not win-amd64.
echo         Delete the .venv folder and re-run setup.bat to rebuild it x64.
endlocal
exit /b 1

:pip_failed
del "%TMPCHK%" >nul 2>&1
echo [ERROR] Dependency installation failed.
endlocal
exit /b 1
