# Descargar_Render

## Actualizar `data.json` y subir cambios

### Refrescar solo `data.json`

- **Windows (.bat):** ejecuta `actualizar_data_json.bat` para lanzar `run_scraper.py` y regenerar el archivo sin tocar git.
- **Python directo:** `python run_scraper.py`

Asegúrate de que el scraper puede funcionar en tu entorno (dependencias instaladas y acceso a internet).

### Refrescar `data.json` y subir cambios a git

- **Windows (.bat):** `update_data_and_push.bat`
- **Python directo:** `python scripts/update_data_and_push.py`

El script:

- Lanza `run_scraper.py` para crear un `data.json` actualizado.
- Comprueba si el archivo cambió.
- Si hay cambios, hace `git add`, `git commit` y `git push` con un mensaje automático.

Asegúrate de que tu entorno local tenga git configurado con acceso al remoto y que el scraper pueda ejecutarse (Chrome headless disponible).
