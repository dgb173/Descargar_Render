@echo off
setlocal
REM Ejecuta el scraper y sube data.json al repositorio.

cd /d "%~dp0"

set "PYTHON_CMD="

if exist "muestra_sin_fallos\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=muestra_sin_fallos\.venv\Scripts\python.exe"
)

if "%PYTHON_CMD%"=="" (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_CMD=.venv\Scripts\python.exe"
    )
)

if "%PYTHON_CMD%"=="" (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_CMD=python"
    ) else (
        where py >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            set "PYTHON_CMD=py"
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo.
    echo No se encontro Python en el PATH. Agrega Python o ejecuta el script manualmente.
    pause
    exit /b 1
)

"%PYTHON_CMD%" scripts\update_data_and_push.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hubo un problema al actualizar o subir los datos.
) else (
    echo.
    echo Proceso terminado. Revisa la consola anterior para detalles.
)

pause
