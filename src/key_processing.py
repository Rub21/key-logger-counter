# key_processing.py
# Procesamiento de teclas y cálculo de estadísticas

from pynput import keyboard
import time
from collections import Counter

def get_key_name(key):
    """Convierte tecla a nombre representativo."""
    # Caracteres imprimibles
    if hasattr(key, 'char') and key.char and key.char.isprintable():
        return key.char.lower()
    
    # Mapeo de teclas especiales
    key_map = {
        keyboard.Key.space: ' ',
        keyboard.Key.enter: '[ENTER]',
        keyboard.Key.backspace: '[BACKSPACE]',
        keyboard.Key.delete: '[DELETE]',
        keyboard.Key.tab: '[TAB]',
        keyboard.Key.esc: '[ESC]',
        keyboard.Key.up: '[UP]',
        keyboard.Key.down: '[DOWN]',
        keyboard.Key.left: '[LEFT]',
        keyboard.Key.right: '[RIGHT]',
        keyboard.Key.ctrl: '[CTRL]',
        keyboard.Key.ctrl_l: '[CTRL_L]',
        keyboard.Key.ctrl_r: '[CTRL_R]',
        keyboard.Key.alt: '[ALT]',
        keyboard.Key.alt_l: '[ALT_L]',
        keyboard.Key.alt_r: '[ALT_R]',
        keyboard.Key.shift: '[SHIFT]',
        keyboard.Key.shift_l: '[SHIFT_L]',
        keyboard.Key.shift_r: '[SHIFT_R]',
        keyboard.Key.cmd: '[CMD]',  # Command en Mac
        keyboard.Key.cmd_l: '[CMD_L]',
        keyboard.Key.cmd_r: '[CMD_R]',
    }
    
    if key in key_map:
        return key_map[key]
    
    # Otras teclas
    return f'[{key.name.upper()}]' if hasattr(key, 'name') else f'[{str(key)}]'

def calculate_statistics(hold_times, inter_key_times, total_keystrokes, stats_interval):
    """Calcula estadísticas temporales."""
    stats = {}
    
    # Estadísticas de hold time
    if hold_times:
        stats['avg_hold_time_ms'] = round((sum(hold_times) / len(hold_times)) * 1000, 2)
    else:
        stats['avg_hold_time_ms'] = 0
    
    # Estadísticas de tiempo entre pulsaciones
    if inter_key_times:
        mean_val = sum(inter_key_times) / len(inter_key_times)
        variance = sum((x - mean_val) ** 2 for x in inter_key_times) / len(inter_key_times)
        std_val = variance ** 0.5
        stats['avg_inter_key_time_ms'] = round(mean_val * 1000, 2)
        stats['std_inter_key_time_ms'] = round(std_val * 1000, 2)
        stats['min_inter_key_time_ms'] = round(min(inter_key_times) * 1000, 2)
        stats['max_inter_key_time_ms'] = round(max(inter_key_times) * 1000, 2)
    else:
        stats['avg_inter_key_time_ms'] = 0
        stats['std_inter_key_time_ms'] = 0
        stats['min_inter_key_time_ms'] = 0
        stats['max_inter_key_time_ms'] = 0
    
    # Estadísticas de velocidad
    stats['total_keystrokes'] = total_keystrokes
    stats['keystrokes_per_second'] = round(total_keystrokes / stats_interval, 2) if stats_interval > 0 else 0
    
    return stats

