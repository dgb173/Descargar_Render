@echo off
setlocal
REM Ejecuta run_scraper.py para regenerar data.json sin hacer commit ni push.

cd /d "%~dp0"

set "PYTHON_CMD="

if exist "muestra_sin_fallos\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=muestra_sin_fallos\.venv\Scripts\python.exe"
    goto run_scraper
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    goto run_scraper
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    goto run_scraper
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py"
    goto run_scraper
)

echo.
echo No se encontro Python en el PATH ni en los entornos virtuales esperados.
echo Ejecuta manualmente: python run_scraper.py
pause
exit /b 1

:run_scraper
"%PYTHON_CMD%" run_scraper.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hubo un problema al ejecutar run_scraper.py. Revisa la salida anterior.
) else (
    echo.
    echo data.json actualizado correctamente.
)

pause
