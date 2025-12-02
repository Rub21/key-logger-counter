# config.py
# Configuración y constantes del keylogger

from pathlib import Path
import random
import string
import json
import logging

# Directorio base
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas y archivos
data_folder = BASE_DIR / "keyboard_data"
data_folder.mkdir(exist_ok=True)

log_system = BASE_DIR / "keylogger_char_count.log"
blocked_apps_file = BASE_DIR / "blocked_apps.json"
combinations_file = BASE_DIR / "key_combinations.json"  # Archivo JSON para guardar combinaciones

# Generar nombre de archivo aleatorio
random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
char_count_file = data_folder / f"keyboard-data-{random_suffix}.csv"
temp_data_file = data_folder / f"keyboard-data-{random_suffix}.json"  # Archivo temporal JSON

# Configuración de intervalo
STATS_INTERVAL = 5  # Segundos entre guardados

# Headers de estadísticas
STATS_HEADERS = [
    'avg_hold_time_ms',      # Tiempo promedio de presión (hold time)
    'avg_inter_key_time_ms', # Tiempo promedio entre pulsaciones
    'std_inter_key_time_ms', # Desviación estándar del tiempo entre pulsaciones
    'min_inter_key_time_ms', # Tiempo mínimo entre pulsaciones
    'max_inter_key_time_ms', # Tiempo máximo entre pulsaciones
    'keystrokes_per_second', # Velocidad de escritura (pulsaciones/segundo)
    'total_keystrokes',      # Total de teclas presionadas en el intervalo
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
    return ['timestamp'] + all_keys + STATS_HEADERS + ['active_application']

CSV_HEADERS = get_csv_headers()

