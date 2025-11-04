
# app.py - Servidor web principal (Flask) - VERSIÓN DE PRODUCCIÓN
from flask import Flask, render_template, abort, request, jsonify, Response, stream_with_context
import json
import os
import sys
import redis  # Añadido para la caché de Redis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Importar el nuevo módulo de filtrado
from modules.pattern_filter_fast import PatternFilter

import numpy

# --- INICIO: Codificador JSON Personalizado ---
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


# Las funciones de scraping en tiempo real para las vistas de "estudio" siguen aquí
from modules.estudio_scraper import (
    obtener_datos_completos_partido, 
    format_ah_as_decimal_string_of, 
    obtener_datos_preview_rapido, 
    obtener_datos_preview_ligero, 
    generar_analisis_mercado_simplificado,
)
# La lógica de normalización de handicap está en su propio módulo
from app_utils import normalize_handicap_to_half_bucket_str

# --- INICIO: Configuración de Redis ---
# Render proporcionará la URL de conexión a través de esta variable de entorno.
redis_client = None
try:
    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        print("Conexión con Redis establecida con éxito.")
    else:
        print("ADVERTENCIA: Variable de entorno REDIS_URL no encontrada. La caché estará desactivada.")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo conectar a Redis. La caché estará desactivada. Error: {e}")
# --- FIN: Configuración de Redis ---


app = Flask(__name__, template_folder='templates', static_folder='static')
app.json_encoder = NumpyJSONEncoder

DATA_FILE = 'data.json'

def load_data_from_file():
    """Carga los datos desde el archivo JSON."""
    if not os.path.exists(DATA_FILE):
        return {"upcoming_matches": [], "finished_matches": []}
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"upcoming_matches": [], "finished_matches": []}

@app.route('/')
def index():
    """Muestra los próximos partidos desde el archivo de datos."""
    try:
        hf = request.args.get('handicap')
        all_data = load_data_from_file()
        matches = all_data.get('upcoming_matches', [])
        
        # Asegurarse de que los datos de handicap existen antes de procesar
        opts = sorted({
            normalize_handicap_to_half_bucket_str(m.get('handicap'))
            for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap')) is not None
        }, key=lambda x: float(x))

        if hf:
            target = normalize_handicap_to_half_bucket_str(hf)
            if target is not None:
                matches = [m for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap', '')) == target]

        return render_template('index.html', matches=matches, handicap_filter=hf, handicap_options=opts, page_mode='upcoming', page_title='Próximos Partidos')
    except Exception as e:
        print(f"ERROR en la ruta principal: {e}")
        return render_template('index.html', matches=[], error=f"No se pudieron cargar los partidos: {e}", page_mode='upcoming', page_title='Próximos Partidos')

@app.route('/resultados')
def resultados():
    """Muestra los partidos finalizados desde el archivo de datos."""
    try:
        hf = request.args.get('handicap')
        all_data = load_data_from_file()
        matches = all_data.get('finished_matches', [])

        opts = sorted({
            normalize_handicap_to_half_bucket_str(m.get('handicap'))
            for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap')) is not None
        }, key=lambda x: float(x))

        if hf:
            target = normalize_handicap_to_half_bucket_str(hf)
            if target is not None:
                matches = [m for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap', '')) == target]

        return render_template('index.html', matches=matches, handicap_filter=hf, handicap_options=opts, page_mode='finished', page_title='Resultados Finalizados')
    except Exception as e:
        print(f"ERROR en la ruta de resultados: {e}")
        return render_template('index.html', matches=[], error=f"No se pudieron cargar los partidos: {e}", page_mode='finished', page_title='Resultados Finalizados')


@app.route('/api/matches')
def api_matches():
    """Devuelve un fragmento de los próximos partidos para paginación."""
    try:
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 10))
        hf = request.args.get('handicap')
        
        all_data = load_data_from_file()
        matches = all_data.get('upcoming_matches', [])

        if hf:
            target = normalize_handicap_to_half_bucket_str(hf)
            if target is not None:
                matches = [m for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap', '')) == target]

        paginated_matches = matches[offset:offset+limit]
        return jsonify({'matches': paginated_matches})
    except Exception as e:
        print(f"Error en la ruta /api/matches: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/finished_matches')
def api_finished_matches():
    """Devuelve un fragmento de los partidos finalizados para paginación."""
    try:
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 10))
        hf = request.args.get('handicap')
        
        all_data = load_data_from_file()
        matches = all_data.get('finished_matches', [])

        if hf:
            target = normalize_handicap_to_half_bucket_str(hf)
            if target is not None:
                matches = [m for m in matches if m and normalize_handicap_to_half_bucket_str(m.get('handicap', '')) == target]

        paginated_matches = matches[offset:offset+limit]
        return jsonify({'matches': paginated_matches})
    except Exception as e:
        print(f"Error en la ruta /api/finished_matches: {e}")
        return jsonify({'error': str(e)}), 500

# --- Las rutas de API y estudio que dependen de scraping en tiempo real se mantienen ---
# --- Estas rutas seguirán haciendo scraping bajo demanda si es necesario ---


@app.route('/estudio/<string:match_id>')
def mostrar_estudio(match_id):
    print(f"Recibida petición para el estudio del partido ID: {match_id}")
    datos_partido = obtener_datos_completos_partido(match_id)
    if not datos_partido or "error" in datos_partido:
        print(f"Error al obtener datos para {match_id}: {datos_partido.get('error')}")
        abort(500, description=datos_partido.get('error', 'Error desconocido'))
    print(f"Datos obtenidos para {datos_partido['home_name']} vs {datos_partido['away_name']}. Renderizando plantilla...")
    return render_template('estudio.html', data=datos_partido, format_ah=format_ah_as_decimal_string_of)

@app.route('/analizar_partido', methods=['GET', 'POST'])
def analizar_partido():
    if request.method == 'POST':
        match_id = request.form.get('match_id')
        if match_id:
            print(f"Recibida petición para analizar partido finalizado ID: {match_id}")
            datos_partido = obtener_datos_completos_partido(match_id)
            if not datos_partido or "error" in datos_partido:
                return render_template('analizar_partido.html', error=datos_partido.get('error', 'Error desconocido'))
            
            main_odds = datos_partido.get("main_match_odds_data")
            h2h_data = datos_partido.get("h2h_data")
            home_name = datos_partido.get("home_name")
            away_name = datos_partido.get("away_name")

            analisis_simplificado_html = ""
            if all([main_odds, h2h_data, home_name, away_name]):
                analisis_simplificado_html = generar_analisis_mercado_simplificado(main_odds, h2h_data, home_name, away_name)

            print(f"Datos obtenidos para {datos_partido['home_name']} vs {datos_partido['away_name']}. Renderizando plantilla...")
            return render_template('estudio.html', 
                                   data=datos_partido, 
                                   format_ah=format_ah_as_decimal_string_of,
                                   analisis_simplificado_html=analisis_simplificado_html)
        else:
            return render_template('analizar_partido.html', error="Por favor, introduce un ID de partido válido.")
    return render_template('analizar_partido.html')

@app.route('/api/preview/<string:match_id>')
def api_preview(match_id):
    """
    Endpoint de API para obtener la vista previa de un partido.
    Utiliza una caché de Redis para evitar el scraping repetido.
    """
    cache_key = f"preview:{match_id}"

    # 1. Intentar obtener el resultado desde la caché de Redis
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                print(f"Sirviendo vista previa para ID {match_id} desde la CACHÉ (Redis).")
                return jsonify(json.loads(cached_data))
        except Exception as e:
            print(f"Error al leer desde Redis: {e}. Se procederá con scraping en vivo.")

    # 2. Si no está en caché, proceder con el scraping en vivo
    print(f"No se encontró caché en Redis para ID {match_id}. Realizando scraping en vivo.")
    try:
        mode = request.args.get('mode', 'light').lower()
        if mode in ['full', 'selenium']:
            # Esta opción puede consumir muchos recursos. Usar con precaución en producción.
            driver = None
            try:
                options = ChromeOptions()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36")
                options.add_argument('--blink-settings=imagesEnabled=false')
                options.binary_location = "/opt/render/project/.render/chrome/opt/google/chrome/chrome"
                driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
                preview_data = obtener_datos_preview_rapido(match_id, driver)
            finally:
                if driver:
                    driver.quit()
        else:
            preview_data = obtener_datos_preview_ligero(match_id)
        
        if isinstance(preview_data, dict) and "error" in preview_data:
            return jsonify(preview_data), 500
            
        # 3. Guardar el nuevo resultado en la caché de Redis para futuras peticiones
        if redis_client:
            try:
                # Guardar con una expiración de 24 horas (86400 segundos)
                redis_client.set(cache_key, json.dumps(preview_data, cls=NumpyJSONEncoder), ex=86400)
                print(f"Guardado nuevo resultado para ID {match_id} en la caché de Redis.")
            except Exception as e:
                print(f"Error al guardar en la caché de Redis: {e}")

        return jsonify(preview_data)

    except Exception as e:
        error_message = f"Ocurrió una excepción inesperada durante el scraping: {str(e)}"
        print(f"Error en la ruta /api/preview/{match_id}: {error_message}")
        return jsonify({'error': error_message}), 500

@app.route('/analisis_patrones/<string:seed_match_id>')
def analisis_patrones(seed_match_id):
    """Muestra la página de resultados del análisis de patrones."""
    all_data = load_data_from_file()
    all_matches = all_data.get('upcoming_matches', []) + all_data.get('finished_matches', [])
    
    seed_match_info = "Partido no encontrado"
    for match in all_matches:
        if match.get('match_id') == seed_match_id:
            seed_match_info = f"{match.get('home_team', '')} vs {match.get('away_team', '')}"
            break
            
    return render_template('analisis_patrones.html', seed_match_id=seed_match_id, seed_match_info=seed_match_info)


@app.route('/api/find_similar/<string:seed_match_id>')
def api_find_similar(seed_match_id):
    """API endpoint para encontrar partidos similares usando streaming."""
    def generate_results():
        """Carga datos, ejecuta el generador de búsqueda y transmite los resultados."""
        try:
            # Cargar todos los datos
            all_data = load_data_from_file()
            all_matches = all_data.get('upcoming_matches', []) + all_data.get('finished_matches', [])
            upcoming_matches = all_data.get('upcoming_matches', [])

            if not all_matches or not upcoming_matches:
                yield json.dumps({"error": "No hay datos de partidos disponibles."})
                return

            # Convertir a DataFrames de Pandas
            import pandas as pd
            seed_df = pd.DataFrame.from_records(all_matches)
            upcoming_df = pd.DataFrame.from_records(upcoming_matches)

            # Instanciar y ejecutar la búsqueda
            filter_tool = PatternFilter(seed_df, upcoming_df)
            top_k_param = request.args.get('top_k', type=int)
            top_k_value = None if top_k_param is None or top_k_param <= 0 else top_k_param
            
            # Iterar directamente sobre el generador y transmitir cada resultado
            for result in filter_tool.incremental_search(seed_match_id, top_k=top_k_value):
                yield json.dumps(result, cls=NumpyJSONEncoder) + '\n'

        except Exception as e:
            yield json.dumps({"error": f"Ocurrió una excepción: {str(e)}"})

    # Devolver una respuesta de streaming
    return Response(stream_with_context(generate_results()), mimetype='application/json')


if __name__ == '__main__':
    # Para desarrollo local, puedes ejecutar esto.
    # Render usará Gunicorn, así que no usará este bloque en producción.
    app.run(host='0.0.0.0', port=8080, debug=True)
