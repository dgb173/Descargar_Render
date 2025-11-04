# Descargar_Render

## Actualizar `data.json` y subir cambios

Para refrescar los partidos igual que antes (generar `data.json` y empujar al repositorio), tienes dos opciones:

- **Windows (.bat):** haz doble clic en `update_data_and_push.bat` o ejecútalo desde PowerShell/CMD.
- **Python directo:** `python scripts/update_data_and_push.py` (o `py scripts/update_data_and_push.py`).

El script:

- Lanza `run_scraper.py` para crear un `data.json` actualizado.
- Comprueba si el archivo cambió.
- Si hay cambios, hace `git add`, `git commit` y `git push` con un mensaje automático.

Asegúrate de que tu entorno local tenga git configurado con acceso al remoto y que el scraper pueda ejecutarse (Chrome headless disponible).
