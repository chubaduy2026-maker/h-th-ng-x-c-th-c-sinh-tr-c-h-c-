@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=py -3"
)

echo Running end-to-end health check...
%PYTHON% src\e2e_check.py
if errorlevel 1 (
  echo E2E check failed.
  pause
  exit /b 1
)

echo E2E check passed.
pause
endlocal
