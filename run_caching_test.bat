@echo off
echo --- MODO TESTEO: PROCESANDO HASTA 3 PARTIDOS ---
set GEMINI_TEST_RUN=true
py cache_previews.py
echo.
echo --- Testeo finalizado ---
pause
