@echo off
setlocal

REM Jump to this script’s folder
cd /d "%~dp0"

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv...
  py -3 -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM Quietly upgrade pip
python -m pip install --upgrade pip >nul 2>&1

REM Ensure the package is installed in editable mode with GUI extras
pip install -e ".[gui]"

REM Launch the GUI; the app will prompt once and remember the file for next time.
python -m SearchTermRating gui %*

endlocal
