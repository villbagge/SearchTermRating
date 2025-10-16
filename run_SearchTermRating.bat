@echo off
setlocal
cd /d "%~dp0"
IF NOT EXIST .venv (
  echo Creating .venv...
  py -3 -m venv .venv
)
CALL .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip show SearchTermRating >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
  echo Installing SearchTermRating in editable mode...
  pip install -e ".[dev]"
)
python -m SearchTermRating %*
endlocal
