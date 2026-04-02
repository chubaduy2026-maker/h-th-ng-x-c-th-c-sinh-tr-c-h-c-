@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=py -3"
)

echo [1/2] Reset attendance flags before exam
%PYTHON% src\reset_attendance.py
if errorlevel 1 (
  echo Reset step failed.
  pause
  exit /b 1
)

echo [2/2] Start attendance camera session
%PYTHON% src\attendance_app.py
if errorlevel 1 (
  echo Attendance session failed.
  pause
  exit /b 1
)

echo Exam session finished.
pause
endlocal
