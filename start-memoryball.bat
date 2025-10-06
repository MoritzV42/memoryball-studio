@echo off
setlocal

REM Start MemoryBall Studio from the repository directory.
cd /d "%~dp0"

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)


set "LOGFILE=%~dp0startup-errors.log"
if exist "%LOGFILE%" del "%LOGFILE%"

"%PYTHON%" "%~dp0main.py" %* 2> "%LOGFILE%"
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo ===== Fehlerprotokoll =====
    if exist "%LOGFILE%" type "%LOGFILE%"
    echo ===========================
    echo.
    echo Der Start ist fehlgeschlagen. Details stehen in startup-errors.log.
    echo Bitte pruefe die angezeigte Fehlermeldung.
    pause
) else (
    if exist "%LOGFILE%" del "%LOGFILE%"
)

endlocal
exit /b %EXITCODE%
