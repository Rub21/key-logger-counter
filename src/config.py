# config.py
# Configuración y constantes del keylogger

from pathlib import Path
import json
import logging
from datetime import datetime

# Directorio base
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas y archivos
data_folder = BASE_DIR / "keyboard_data"
data_folder.mkdir(exist_ok=True)

log_system = BASE_DIR / "keylogger_char_count.log"
blocked_apps_file = BASE_DIR / "blocked_apps.json"
combinations_file = BASE_DIR / "key_combinations.json"  # Archivo JSON para guardar combinaciones

# Generar nombre de archivo con timestamp (formato: data_YYYYMMDD_HHMMSS)
timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
char_count_file = data_folder / f"data_{timestamp_suffix}.csv"
temp_data_file = data_folder / f"data_{timestamp_suffix}.json"  # Archivo temporal JSON

# Configuración de intervalo
STATS_INTERVAL = 5  # Segundos entre guardados

# Headers de estadísticas del teclado
STATS_HEADERS = [
    'avg_hold_time_ms',      # Tiempo promedio de presión (hold time)
    'avg_inter_key_time_ms', # Tiempo promedio entre pulsaciones
    'std_inter_key_time_ms', # Desviación estándar del tiempo entre pulsaciones
    'min_inter_key_time_ms', # Tiempo mínimo entre pulsaciones
    'max_inter_key_time_ms', # Tiempo máximo entre pulsaciones
    'keystrokes_per_second', # Velocidad de escritura (pulsaciones/segundo)
    'total_keystrokes',      # Total de teclas presionadas en el intervalo
]

# Headers de estadísticas del mouse (base - se expanden dinámicamente con pantallas)
MOUSE_STATS_BASE_HEADERS = [
    'total_clicks',              # Total de clics en el intervalo
    'left_clicks',               # Clics con botón izquierdo
    'right_clicks',              # Clics con botón derecho
    'middle_clicks',             # Clics con botón medio
    'scroll_events',             # Eventos de scroll
    'avg_x_position',            # Posición X promedio de los clics
    'avg_y_position',            # Posición Y promedio de los clics
    'min_x',                     # Posición X mínima
    'max_x',                     # Posición X máxima
    'min_y',                     # Posición Y mínima
    'max_y',                     # Posición Y máxima
    'avg_inter_click_time_ms',   # Tiempo promedio entre clics (ms)
    'clicks_per_second',         # Velocidad de clics (clics/segundo)
    'total_screens',             # Número total de pantallas detectadas
    'most_used_screen',          # Índice de la pantalla más usada
    'scroll_up',                 # Eventos de scroll hacia arriba
    'scroll_down',               # Eventos de scroll hacia abajo
    'scroll_left',               # Eventos de scroll hacia izquierda
    'scroll_right',              # Eventos de scroll hacia derecha
    'avg_scroll_magnitude',      # Magnitud promedio del scroll
    'avg_inter_scroll_time_ms',  # Tiempo promedio entre scrolls (ms)
    'scrolls_per_second',        # Velocidad de scrolls (scrolls/segundo)
]

def get_mouse_stats_headers(max_screens=3):
    """Obtiene los headers de estadísticas del mouse, expandidos dinámicamente por pantallas.
    
    Args:
        max_screens: Número máximo de pantallas a incluir (por defecto 3)
    """
    headers = MOUSE_STATS_BASE_HEADERS.copy()
    
    # Agregar headers dinámicos por pantalla
    # Se expandirán automáticamente según las pantallas detectadas
    try:
        from .screen_detection import get_total_screens
        detected_screens = get_total_screens()
        max_screens = max(max_screens, detected_screens)
    except:
        pass  # Usar el valor por defecto si hay error
    
    for i in range(max_screens):
        headers.append(f'clicks_screen_{i}')
        headers.append(f'scroll_screen_{i}')
    
    return headers

# Para compatibilidad, mantener MOUSE_STATS_HEADERS
MOUSE_STATS_HEADERS = MOUSE_STATS_BASE_HEADERS

# Headers de información de aplicación
APP_INFO_HEADERS = [
    'active_application',        # Nombre de la aplicación activa
    'app_bundle_id',             # Bundle ID (macOS) o ruta del ejecutable
    'app_window_title',          # Título de la ventana activa
    'app_process_id',            # ID del proceso
]

def get_all_possible_keys():
    """Genera lista de todas las teclas posibles para los headers del CSV."""
    keys = []
    # Letras minúsculas (a-z)
    keys.extend([chr(i) for i in range(ord('a'), ord('z')+1)])
    # Números (0-9)
    keys.extend([str(i) for i in range(10)])
    # Caracteres especiales comunes
    keys.extend([' ', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', 
                '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|', ';', 
                ':', "'", '"', ',', '.', '<', '>', '/', '?', '`', '~'])
    # Teclas especiales
    keys.extend(['[ENTER]', '[BACKSPACE]', '[DELETE]', '[TAB]', '[ESC]',
                '[UP]', '[DOWN]', '[LEFT]', '[RIGHT]', '[HOME]', '[END]',
                '[PAGE_UP]', '[PAGE_DOWN]', '[CTRL]', '[CTRL_L]', '[CTRL_R]',
                '[ALT]', '[ALT_L]', '[ALT_R]', '[SHIFT]', '[SHIFT_L]', '[SHIFT_R]',
                '[CMD]', '[CMD_L]', '[CMD_R]'])
    return sorted(keys)

ALL_POSSIBLE_KEYS = get_all_possible_keys()

# Combinaciones dinámicas que se van agregando
dynamic_combinations = set()  # Set de combinaciones encontradas (ej: "[SHIFT_L]+a")

def load_combinations():
    """Carga combinaciones guardadas desde JSON."""
    global dynamic_combinations
    try:
        if combinations_file.exists():
            with open(combinations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dynamic_combinations = set(data.get('combinations', []))
                return dynamic_combinations
        else:
            # Crear archivo vacío
            save_combinations(set())
            return set()
    except Exception as e:
        logging.error(f"Error loading combinations: {e}")
        return set()

def save_combinations(combinations):
    """Guarda combinaciones en JSON."""
    try:
        data = {
            "description": "Combinaciones de teclas detectadas (modificador+tecla)",
            "combinations": sorted(list(combinations)),
            "total": len(combinations)
        }
        with open(combinations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving combinations: {e}")

def get_csv_headers():
    """Obtiene los headers del CSV incluyendo combinaciones dinámicas."""
    all_keys = ALL_POSSIBLE_KEYS + sorted(dynamic_combinations)
    mouse_stats_headers = get_mouse_stats_headers()
    return (['timestamp'] + all_keys + STATS_HEADERS + mouse_stats_headers + APP_INFO_HEADERS)

CSV_HEADERS = get_csv_headers()

