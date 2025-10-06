@echo off
setlocal

REM Start MemoryBall Studio from the repository directory.
cd /d "%~dp0"

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "%~dp0main.py" %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo Der Start ist fehlgeschlagen. Details stehen in startup-errors.log.
    echo Bitte pruefe die angezeigte Fehlermeldung.
    pause
)

endlocal
exit /b %EXITCODE%
