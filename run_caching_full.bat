@echo off
echo --- MODO COMPLETO: PROCESANDO TODOS LOS PARTIDOS FINALIZADOS ---
set GEMINI_TEST_RUN=false
py cache_previews.py
echo.
echo --- Proceso completo ---
pause
