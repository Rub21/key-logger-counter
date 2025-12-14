# file_handler.py
# Manejo de archivos CSV y JSON

import csv
import json
import logging
from .config import (
    char_count_file, temp_data_file, ALL_POSSIBLE_KEYS, get_csv_headers, 
    dynamic_combinations, STATS_HEADERS, MOUSE_STATS_BASE_HEADERS, APP_INFO_HEADERS,
    get_mouse_stats_headers
)

def setup_temp_json():
    """Inicializa el archivo JSON temporal para guardar datos durante la ejecución."""
    if not temp_data_file.exists():
        data = {
            "description": "Datos temporales del keylogger (se convertirá a CSV al finalizar)",
            "intervals": []
        }
        with open(temp_data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Created temporary JSON file: {temp_data_file}")

def update_csv_headers(new_combinations):
    """Actualiza los headers del CSV si hay nuevas combinaciones."""
    if not new_combinations:
        return False
    
    try:
        # Leer todas las filas existentes
        rows = []
        existing_headers = None
        
        if char_count_file.exists():
            with open(char_count_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_headers = list(reader.fieldnames) or []
                rows = list(reader)
        
        # Crear nuevos headers con combinaciones
        if existing_headers:
            # Encontrar posición antes de STATS_HEADERS y active_application
            from .config import STATS_HEADERS
            stats_start_idx = None
            for i, h in enumerate(existing_headers):
                if h in STATS_HEADERS:
                    stats_start_idx = i
                    break
            
            # Insertar nuevas combinaciones antes de las estadísticas
            insert_pos = stats_start_idx if stats_start_idx else len(existing_headers) - len(STATS_HEADERS) - 1
            
            for combo in sorted(new_combinations):
                if combo not in existing_headers:
                    existing_headers.insert(insert_pos, combo)
                    insert_pos += 1
            
            # Reescribir archivo con nuevos headers
            with open(char_count_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=existing_headers)
                writer.writeheader()
                for row in rows:
                    # Asegurar que todas las columnas estén presentes
                    full_row = {h: row.get(h, 0) for h in existing_headers}
                    writer.writerow(full_row)
            
            logging.info(f"Updated CSV headers with {len(new_combinations)} new combinations")
            return True
    except Exception as e:
        logging.error(f"Error updating CSV headers: {e}")
    
    return False

def save_to_json(timestamp_numeric, counts, stats, active_app, app_info=None, mouse_stats=None, mouse_counts=None):
    """Guarda datos temporales en JSON.
    
    Args:
        timestamp_numeric: Timestamp del intervalo
        counts: Conteos de teclas presionadas
        stats: Estadísticas del teclado
        active_app: Nombre de la aplicación activa
        app_info: Diccionario con información detallada de la aplicación
        mouse_stats: Estadísticas del mouse
        mouse_counts: Conteos de eventos del mouse
    """
    # Guardar incluso si no hay teclas (puede haber datos del mouse)
    if not counts and (not mouse_stats or mouse_stats.get('total_clicks', 0) == 0):
        return
    
    # Detectar nuevas combinaciones y agregarlas al conjunto global
    new_combinations = []
    for key_name in counts.keys():
        if '+' in key_name and key_name not in dynamic_combinations:
            dynamic_combinations.add(key_name)
            new_combinations.append(key_name)
    
    # Guardar nuevas combinaciones en JSON
    if new_combinations:
        from .config import save_combinations
        save_combinations(dynamic_combinations)
        logging.info(f"Found {len(new_combinations)} new combinations: {new_combinations}")
    
    # Leer datos existentes del JSON
    try:
        if temp_data_file.exists():
            with open(temp_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"intervals": []}
    except:
        data = {"intervals": []}
    
    # Crear entrada para este intervalo
    interval_data = {
        'timestamp': timestamp_numeric,
        'active_application': active_app,
        'counts': dict(counts),
        'stats': stats
    }
    
    # Agregar información de la aplicación si está disponible
    if app_info:
        interval_data['app_info'] = app_info
    
    # Agregar estadísticas del mouse si están disponibles
    if mouse_stats:
        interval_data['mouse_stats'] = mouse_stats
    
    # Agregar conteos del mouse si están disponibles
    if mouse_counts:
        interval_data['mouse_counts'] = dict(mouse_counts)
    
    # Agregar intervalo a los datos
    data['intervals'].append(interval_data)
    
    # Guardar en JSON
    with open(temp_data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    total_keys = sum(counts.values()) if counts else 0
    total_clicks = mouse_stats.get('total_clicks', 0) if mouse_stats else 0
    logging.info(f"Saved to JSON: {total_keys} keystrokes, {total_clicks} clicks - App: {active_app}")

def convert_json_to_csv():
    """Convierte el JSON temporal a CSV al finalizar."""
    if not temp_data_file.exists():
        logging.info("No temporary JSON file to convert")
        return
    
    try:
        # Leer datos del JSON
        with open(temp_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        intervals = data.get('intervals', [])
        if not intervals:
            logging.info("No data to convert")
            return
        
        # Recopilar todas las combinaciones de todos los intervalos
        all_combinations = set()
        for interval in intervals:
            for key_name in interval.get('counts', {}).keys():
                if '+' in key_name:
                    all_combinations.add(key_name)
        
        # Agregar combinaciones al conjunto global
        dynamic_combinations.update(all_combinations)
        from .config import save_combinations
        save_combinations(dynamic_combinations)
        
        # Obtener headers finales
        headers = get_csv_headers()
        
        # Crear CSV
        with open(char_count_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            # Convertir cada intervalo a fila CSV
            for interval in intervals:
                timestamp = interval['timestamp']
                active_app = interval.get('active_application', 'Unknown')
                counts = interval.get('counts', {})
                stats = interval.get('stats', {})
                app_info = interval.get('app_info', {})
                mouse_stats = interval.get('mouse_stats', {})
                
                # Crear fila base
                row = {'timestamp': timestamp}
                
                # Agregar información de la aplicación
                row['active_application'] = active_app
                row['app_bundle_id'] = app_info.get('bundle_id', 'Unknown')
                row['app_window_title'] = app_info.get('window_title', 'Unknown')
                row['app_process_id'] = app_info.get('process_id', '')
                
                # Agregar estadísticas del teclado
                row.update(stats)
                
                # Agregar estadísticas del mouse
                row.update(mouse_stats)
                
                # Inicializar todas las columnas de teclas con 0
                # Excluir campos conocidos (timestamp, stats, app_info, mouse_stats)
                excluded_fields = set(['timestamp'] + APP_INFO_HEADERS + STATS_HEADERS + 
                                     MOUSE_STATS_BASE_HEADERS + 
                                     [f'clicks_screen_{i}' for i in range(10)] +  # Hasta 10 pantallas
                                     [f'scroll_screen_{i}' for i in range(10)])
                
                for key in headers:
                    if key not in excluded_fields and key not in row:
                        row[key] = 0
                
                # Llenar con conteos reales de teclas
                for char, count in counts.items():
                    if char in headers:
                        row[char] = count
                
                # Escribir fila
                filtered_row = {k: row.get(k, 0 if k not in ['timestamp', 'app_process_id'] else '') for k in headers}
                writer.writerow(filtered_row)
        
        total_intervals = len(intervals)
        logging.info(f"Converted {total_intervals} intervals from JSON to CSV: {char_count_file}")
        
        # Eliminar archivo JSON temporal
        temp_data_file.unlink()
        logging.info(f"Deleted temporary JSON file: {temp_data_file}")
        
    except Exception as e:
        logging.error(f"Error converting JSON to CSV: {e}")
        import traceback
        traceback.print_exc()

