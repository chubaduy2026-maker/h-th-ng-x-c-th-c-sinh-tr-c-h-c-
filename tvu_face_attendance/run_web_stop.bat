@echo off
setlocal

set "PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
  set "PID=%%P"
  goto :kill_it
)

echo [INFO] Khong co server nao dang nghe o cong 8000.
goto :eof

:kill_it
echo [INFO] Dang dung server PID %PID% tren cong 8000...
taskkill /PID %PID% /F >nul 2>&1
if %errorlevel%==0 (
  echo [OK] Da dung server.
) else (
  echo [ERROR] Khong dung duoc server PID %PID%.
)

endlocal
