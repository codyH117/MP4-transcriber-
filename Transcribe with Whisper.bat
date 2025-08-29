@echo on
setlocal ENABLEDELAYEDEXPANSION

REM --- Set your Python. If "python" isn't found, try:  set "PYTHON=py -3"
set "PYTHON=python"

REM --- Require a dropped file
if "%~1"=="" (
  echo Drag a video/audio file onto this .bat to transcribe.
  echo Or run: "%PYTHON%" "%~dp0transcribe.py" "C:\path\to\your.mp4"
  pause
  exit /b 1
)

REM --- Call the script (quotes handle spaces in paths)
"%PYTHON%" "%~dp0transcribe.py" "%~1"
if errorlevel 1 (
  echo.
  echo [Error] Python or dependencies may be missing.
  echo Try in a terminal: python -V   and   pip show openai-whisper
)

pause
