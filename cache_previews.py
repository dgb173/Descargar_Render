
import json
import os
import sys
import time

# --- INICIO: Añadido para encontrar módulos en la subcarpeta ---
# Esto permite que el script encuentre la lógica de scraping sin mover ficheros.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Apuntamos a la carpeta que CONTIENE la carpeta 'modules'
# En este caso, 'muestra_sin_fallos' está al mismo nivel que este script.
# Python buscará ahora en 'muestra_sin_fallos/modules/...'
sys.path.append(os.path.join(script_dir, 'muestra_sin_fallos'))
# --- FIN: Añadido para encontrar módulos ---

# Ahora la importación funciona gracias a la modificación del sys.path
from modules.estudio_scraper import obtener_datos_preview_rapido
import numpy

# --- INICIO: Codificador JSON Personalizado ---
# Esta clase convierte tipos de datos de NumPy (como int64) a tipos nativos de Python
# que la librería json puede entender y serializar.
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)
# --- FIN: Codificador JSON Personalizado ---


# --- CONFIGURACIÓN ---
DATA_FILE = 'data.json'
# La carpeta de caché debe estar dentro de 'static' para que la app la pueda servir si es necesario
# y seguimos la estructura del proyecto anidado.
CACHE_DIR = os.path.join(script_dir, 'muestra_sin_fallos', 'static', 'cached_previews')
TESTRUN_LIMIT = 3 # Límite de IDs para la primera ejecución de testeo.

def pre_cache_previews(limit=None):
    """
    Rellena la caché con las vistas previas de los partidos finalizados.
    """
    print("--- Iniciando proceso de cacheo de vistas previas ---")

    # 1. Asegurarse de que el directorio de caché existe
    if not os.path.exists(CACHE_DIR):
        print(f"Creando directorio de caché en: {CACHE_DIR}")
        os.makedirs(CACHE_DIR)

    # 2. Cargar el fichero de datos principal
    if not os.path.exists(DATA_FILE):
        print(f"Error: No se encuentra el fichero de datos '{DATA_FILE}'. Abortando.")
        return

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: El fichero '{DATA_FILE}' está corrupto o mal formateado. Abortando.")
        return

    finished_matches = data.get('finished_matches', [])
    if not finished_matches:
        print("No se encontraron partidos finalizados en el fichero de datos.")
        return

    print(f"Se encontraron {len(finished_matches)} partidos finalizados. Procesando...")
    
    # Aplicar límite si se ha especificado (para el testeo)
    matches_to_process = finished_matches[:limit] if limit is not None else finished_matches
    if limit is not None:
        print(f"\n--- MODO DE TESTEO: Procesando solo los primeros {limit} partidos. ---\n")


    processed_count = 0
    skipped_count = 0

    # 3. Iterar sobre los partidos y cachear
    for i, match in enumerate(matches_to_process):
        match_id = match.get('id')
        if not match_id:
            continue

        cache_file_path = os.path.join(CACHE_DIR, f"{match_id}.json")
        
        print(f"({i+1}/{len(matches_to_process)}) Procesando ID: {match_id}... ", end="")

        # 4. Comprobar si ya está en caché (evitar duplicados)
        if os.path.exists(cache_file_path):
            print("YA EN CACHÉ (omitido)")
            skipped_count += 1
            continue
        
        # 5. Si no está, extraer los datos
        print("EXTRAYENDO DATOS...")
        try:
            # Usamos la función de vista previa "rápida" (la que usa Selenium)
            start_time = time.time()
            preview_data = obtener_datos_preview_rapido(match_id)
            end_time = time.time()

            # 6. Guardar en el fichero de caché
            if preview_data and "error" not in preview_data:
                with open(cache_file_path, 'w', encoding='utf-8') as f:
                    # Usamos el codificador personalizado para manejar tipos de NumPy
                    json.dump(preview_data, f, ensure_ascii=False, indent=4, cls=NumpyJSONEncoder)
                processed_count += 1
                # Corregido el SyntaxWarning usando doble barra \\
                print(f"    \\_> GUARDADO en caché. (Tardó {end_time - start_time:.2f}s)")
            else:
                error_msg = preview_data.get("error", "Error desconocido")
                # Corregido el SyntaxWarning usando doble barra \\
                print(f"    \\_> ERROR al extraer datos para ID {match_id}: {error_msg}")

        except Exception as e:
            # Corregido el SyntaxWarning usando doble barra \\
            print(f"    \\_> ERROR CRÍTICO al procesar ID {match_id}: {e}")
        
        # Pequeña pausa para no saturar el servidor de origen
        time.sleep(1)

    print("\n--- Proceso de cacheo finalizado ---")
    print(f"Resultados: {processed_count} nuevos análisis guardados, {skipped_count} omitidos (ya en caché).")


if __name__ == "__main__":
    # Esta variable de entorno la usaremos para controlar si es un test o no
    is_test_run = os.environ.get('GEMINI_TEST_RUN', 'true').lower() == 'true'
    
    if is_test_run:
        pre_cache_previews(limit=TESTRUN_LIMIT)
        # Después del primer test, le decimos al sistema que las próximas veces debe ser una ejecución completa
        print("\nPara ejecutar el script completo (sin límite), puedes crear un .bat que primero quite la variable de entorno:")
        print("set GEMINI_TEST_RUN=false")
        print("py cache_previews.py")
        print("pause")
    else:
        pre_cache_previews()
