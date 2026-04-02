@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo Installing dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
  echo Setup failed.
  pause
  exit /b 1
)

echo Setup completed.
pause
endlocal
