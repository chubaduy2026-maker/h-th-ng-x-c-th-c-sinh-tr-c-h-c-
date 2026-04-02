@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay virtualenv .venv
  echo Hay chay setup_env.bat truoc.
  pause
  exit /b 1
)

set "PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
  set "PID=%%P"
  goto :check_existing
)

goto :start_server

:check_existing
echo [INFO] Port 8000 dang duoc su dung (PID %PID%).
for /f %%S in ('powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8000/health -TimeoutSec 3; $r.StatusCode } catch { 0 }"') do set "HTTP_STATUS=%%S"

if "%HTTP_STATUS%"=="200" (
  echo [INFO] TVU web app dang chay san.
  start "" "http://127.0.0.1:8000/attendance"
  goto :eof
)

echo [WARN] Cong 8000 dang bi app khac chiem, khong phai TVU web app.
echo [INFO] Dang dung PID %PID% de khoi dong lai dung server...
taskkill /PID %PID% /F >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Khong the dung PID %PID%.
  echo [HINT] Hay chay run_web_stop.bat hoac taskkill /PID %PID% /F roi thu lai.
  pause
  exit /b 1
)

:start_server
echo [INFO] Starting FastAPI web server at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000/attendance"
.venv\Scripts\python.exe -m uvicorn src.web_app:app --host 0.0.0.0 --port 8000

endlocal
