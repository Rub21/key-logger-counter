# keylogger.py
# Lógica principal del keylogger

import logging
import threading
import time
import datetime
from collections import Counter
from pynput import keyboard

from .config import STATS_INTERVAL
from .app_detection import get_active_application, is_blocked_application, get_active_application_info
from .key_processing import get_key_name, calculate_statistics
from .file_handler import save_to_json
from .mouse_tracking import get_mouse_statistics, get_mouse_counts, start_mouse_listener

# Variables globales de estado
running = True
character_counter = Counter()
character_lock = threading.Lock()

# Estadísticas temporales
key_press_times = []
last_key_time = None
hold_times = []
inter_key_times = []

# Estado de teclas modificadoras presionadas
modifier_keys_pressed = set()  # Conjunto de modificadoras actualmente presionadas

def on_press(key, blocked_applications):
    """Maneja presión de tecla - Acumula conteos en memoria y tiempos."""
    global character_counter, last_key_time, key_press_times, inter_key_times, modifier_keys_pressed
    
    # Verificar si app está bloqueada
    if is_blocked_application(get_active_application(), blocked_applications):
        return
    
    current_time = time.time()
    key_name = get_key_name(key)
    
    # Detectar teclas modificadoras directamente desde el objeto key
    is_modifier = False
    modifier_name = None
    
    if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
        is_modifier = True
        if key == keyboard.Key.shift_l:
            modifier_name = '[SHIFT_L]'
        elif key == keyboard.Key.shift_r:
            modifier_name = '[SHIFT_R]'
        else:
            modifier_name = '[SHIFT]'
    elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        is_modifier = True
        if key == keyboard.Key.ctrl_l:
            modifier_name = '[CTRL_L]'
        elif key == keyboard.Key.ctrl_r:
            modifier_name = '[CTRL_R]'
        else:
            modifier_name = '[CTRL]'
    elif key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
        is_modifier = True
        if key == keyboard.Key.alt_l:
            modifier_name = '[ALT_L]'
        elif key == keyboard.Key.alt_r:
            modifier_name = '[ALT_R]'
        else:
            modifier_name = '[ALT]'
    elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
        is_modifier = True
        if key == keyboard.Key.cmd_l:
            modifier_name = '[CMD_L]'
        elif key == keyboard.Key.cmd_r:
            modifier_name = '[CMD_R]'
        else:
            modifier_name = '[CMD]'
    
    # Si es modificadora, agregarla al conjunto y contarla
    if is_modifier and modifier_name:
        with character_lock:
            modifier_keys_pressed.add(modifier_name)
            character_counter[modifier_name] += 1
    
    # Calcular tiempo entre pulsaciones (solo para teclas no modificadoras)
    if not is_modifier and last_key_time is not None:
        inter_key_time = current_time - last_key_time
        with character_lock:
            inter_key_times.append(inter_key_time)
    
    # Guardar tiempo de presión
    with character_lock:
        key_press_times.append(current_time)
        if not is_modifier:
            last_key_time = current_time
    
    # Incrementar contador para teclas no modificadoras
    if key_name and not is_modifier:
        with character_lock:
            # Contar la tecla sola
            character_counter[key_name] += 1
            
            # Si hay modificadoras presionadas, crear combinaciones
            if modifier_keys_pressed:
                for mod in modifier_keys_pressed:
                    combination = f"{mod}+{key_name}"
                    character_counter[combination] += 1

def on_release(key, blocked_applications):
    """Maneja liberación de tecla - Calcula tiempo de presión (hold time)."""
    global hold_times, key_press_times, modifier_keys_pressed
    
    # Verificar si app está bloqueada
    if is_blocked_application(get_active_application(), blocked_applications):
        return
    
    current_time = time.time()
    
    # Detectar si es modificadora
    is_modifier = False
    modifier_name = None
    
    if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
        is_modifier = True
        if key == keyboard.Key.shift_l:
            modifier_name = '[SHIFT_L]'
        elif key == keyboard.Key.shift_r:
            modifier_name = '[SHIFT_R]'
        else:
            modifier_name = '[SHIFT]'
    elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        is_modifier = True
        if key == keyboard.Key.ctrl_l:
            modifier_name = '[CTRL_L]'
        elif key == keyboard.Key.ctrl_r:
            modifier_name = '[CTRL_R]'
        else:
            modifier_name = '[CTRL]'
    elif key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
        is_modifier = True
        if key == keyboard.Key.alt_l:
            modifier_name = '[ALT_L]'
        elif key == keyboard.Key.alt_r:
            modifier_name = '[ALT_R]'
        else:
            modifier_name = '[ALT]'
    elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
        is_modifier = True
        if key == keyboard.Key.cmd_l:
            modifier_name = '[CMD_L]'
        elif key == keyboard.Key.cmd_r:
            modifier_name = '[CMD_R]'
        else:
            modifier_name = '[CMD]'
    
    # Si es modificadora, removerla del conjunto
    if is_modifier and modifier_name:
        with character_lock:
            modifier_keys_pressed.discard(modifier_name)
    
    # Buscar el tiempo de presión correspondiente
    with character_lock:
        if key_press_times:
            press_time = key_press_times[-1]
            hold_time = current_time - press_time
            hold_times.append(hold_time)
            key_press_times.pop()

def save_character_counts(blocked_applications):
    """Guarda conteos acumulados en memoria al disco cada X segundos."""
    global character_counter, last_key_time, hold_times, inter_key_times, running
    
    if not running:
        return
    
    try:
        timestamp_dt = datetime.datetime.now()
        timestamp_numeric = timestamp_dt.timestamp()
        active_app = get_active_application()
        app_info = get_active_application_info()
        
        # Si la app está bloqueada, resetear contador y continuar
        if is_blocked_application(active_app, blocked_applications):
            with character_lock:
                character_counter.clear()
                hold_times.clear()
                inter_key_times.clear()
                last_key_time = None
            threading.Timer(STATS_INTERVAL, save_character_counts, args=(blocked_applications,)).start()
            return
        
        # Obtener conteos y estadísticas temporales del teclado
        with character_lock:
            counts = dict(character_counter)
            character_counter.clear()
            
            hold_times_copy = hold_times.copy()
            inter_key_times_copy = inter_key_times.copy()
            hold_times.clear()
            inter_key_times.clear()
            last_key_time = None
        
        # Calcular estadísticas del teclado
        total_keystrokes = sum(counts.values())
        stats = calculate_statistics(hold_times_copy, inter_key_times_copy, total_keystrokes, STATS_INTERVAL)
        
        # Obtener estadísticas del mouse
        mouse_stats = get_mouse_statistics(STATS_INTERVAL)
        mouse_counts = get_mouse_counts()
        
        # Guardar al JSON temporal (incluye datos del teclado, mouse y app)
        save_to_json(timestamp_numeric, counts, stats, active_app, app_info, mouse_stats, mouse_counts)
        
        # Programar siguiente guardado
        threading.Timer(STATS_INTERVAL, save_character_counts, args=(blocked_applications,)).start()
    except Exception as e:
        logging.error(f"Error saving counts: {e}")
        if running:
            threading.Timer(STATS_INTERVAL, save_character_counts, args=(blocked_applications,)).start()

def cleanup():
    """Limpia recursos al cerrar y convierte JSON a CSV."""
    global running
    running = False
    
    # Convertir JSON temporal a CSV antes de salir
    from .file_handler import convert_json_to_csv
    logging.info("Converting temporary JSON to CSV...")
    convert_json_to_csv()

def start_listener(blocked_applications):
    """Inicia los listeners del teclado y mouse."""
    def on_press_wrapper(key):
        on_press(key, blocked_applications)
    
    def on_release_wrapper(key):
        on_release(key, blocked_applications)
    
    # Iniciar listener del mouse en un thread separado
    mouse_thread = threading.Thread(
        target=start_mouse_listener,
        args=(blocked_applications,),
        daemon=True
    )
    mouse_thread.start()
    
    # Iniciar listener del teclado (bloqueante)
    with keyboard.Listener(on_press=on_press_wrapper, on_release=on_release_wrapper) as listener:
        listener.join()

