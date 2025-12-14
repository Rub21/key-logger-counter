# mouse_tracking.py
# Tracking de eventos del mouse (clics, posición, scroll)

import logging
import threading
import time
from collections import Counter
from pynput import mouse

from .app_detection import get_active_application, is_blocked_application
from .screen_detection import get_screen_for_position, get_total_screens

# Variables globales de estado
mouse_counter = Counter()  # Contador de eventos del mouse
mouse_lock = threading.Lock()

# Estadísticas temporales del mouse
click_positions = []  # Lista de (x, y, button, timestamp, screen_index)
scroll_events = []  # Lista de (x, y, dx, dy, timestamp, screen_index, direction)
last_click_time = None
inter_click_times = []
last_scroll_time = None
inter_scroll_times = []

# Estadísticas acumuladas por intervalo
mouse_stats = {
    'total_clicks': 0,
    'left_clicks': 0,
    'right_clicks': 0,
    'middle_clicks': 0,
    'scroll_events': 0,
    'avg_x_position': 0,
    'avg_y_position': 0,
    'min_x': 0,
    'max_x': 0,
    'min_y': 0,
    'max_y': 0,
    'avg_inter_click_time_ms': 0,
    'clicks_per_second': 0
}

def on_click(x, y, button, pressed, blocked_applications):
    """Maneja eventos de clic del mouse."""
    global mouse_counter, click_positions, last_click_time, inter_click_times
    
    # Solo registrar cuando se presiona el botón (no cuando se suelta)
    if not pressed:
        return
    
    # Verificar si app está bloqueada
    if is_blocked_application(get_active_application(), blocked_applications):
        return
    
    current_time = time.time()
    
    # Determinar en qué pantalla está el clic
    screen_info = get_screen_for_position(x, y)
    screen_index = screen_info['index'] if screen_info else 0
    
    # Determinar tipo de botón
    button_name = None
    if button == mouse.Button.left:
        button_name = 'left'
    elif button == mouse.Button.right:
        button_name = 'right'
    elif button == mouse.Button.middle:
        button_name = 'middle'
    
    if button_name:
        with mouse_lock:
            # Contar clic
            mouse_counter[f'click_{button_name}'] += 1
            mouse_counter['total_clicks'] += 1
            mouse_counter[f'clicks_screen_{screen_index}'] += 1
            
            # Guardar posición, tiempo y pantalla
            click_positions.append((x, y, button_name, current_time, screen_index))
            
            # Calcular tiempo entre clics
            if last_click_time is not None:
                inter_click_time = current_time - last_click_time
                inter_click_times.append(inter_click_time)
            
            last_click_time = current_time

def on_scroll(x, y, dx, dy, blocked_applications):
    """Maneja eventos de scroll del mouse."""
    global mouse_counter, scroll_events, last_scroll_time, inter_scroll_times
    
    # Verificar si app está bloqueada
    if is_blocked_application(get_active_application(), blocked_applications):
        return
    
    current_time = time.time()
    
    # Determinar en qué pantalla está el scroll
    screen_info = get_screen_for_position(x, y)
    screen_index = screen_info['index'] if screen_info else 0
    
    # Determinar dirección del scroll
    direction = "unknown"
    if dy > 0:
        direction = "down"
    elif dy < 0:
        direction = "up"
    elif dx > 0:
        direction = "right"
    elif dx < 0:
        direction = "left"
    
    # Calcular magnitud del scroll
    scroll_magnitude = (dx**2 + dy**2)**0.5
    
    with mouse_lock:
        mouse_counter['scroll_events'] += 1
        mouse_counter[f'scroll_screen_{screen_index}'] += 1
        mouse_counter[f'scroll_{direction}'] += 1
        
        scroll_events.append((x, y, dx, dy, current_time, screen_index, direction, scroll_magnitude))
        
        # Calcular tiempo entre scrolls
        if last_scroll_time is not None:
            inter_scroll_time = current_time - last_scroll_time
            inter_scroll_times.append(inter_scroll_time)
        
        last_scroll_time = current_time

def get_mouse_statistics(interval_seconds):
    """Calcula estadísticas del mouse para el intervalo actual."""
    global click_positions, scroll_events, inter_click_times, inter_scroll_times, mouse_stats
    
    with mouse_lock:
        # Copiar datos para procesar
        clicks_copy = click_positions.copy()
        scrolls_copy = scroll_events.copy()
        inter_click_times_copy = inter_click_times.copy()
        inter_scroll_times_copy = inter_scroll_times.copy()
        
        # Limpiar contadores
        click_positions.clear()
        scroll_events.clear()
        inter_click_times.clear()
        inter_scroll_times.clear()
    
    # Obtener número total de pantallas
    total_screens = get_total_screens()
    
    # Calcular estadísticas básicas
    stats = {
        'total_clicks': len(clicks_copy),
        'left_clicks': sum(1 for _, _, btn, _, _ in clicks_copy if btn == 'left'),
        'right_clicks': sum(1 for _, _, btn, _, _ in clicks_copy if btn == 'right'),
        'middle_clicks': sum(1 for _, _, btn, _, _ in clicks_copy if btn == 'middle'),
        'scroll_events': len(scrolls_copy),
        'total_screens': total_screens,
    }
    
    # Estadísticas de posición de clics
    if clicks_copy:
        x_positions = [x for x, _, _, _, _ in clicks_copy]
        y_positions = [y for _, y, _, _, _ in clicks_copy]
        
        stats['avg_x_position'] = round(sum(x_positions) / len(x_positions), 2)
        stats['avg_y_position'] = round(sum(y_positions) / len(y_positions), 2)
        stats['min_x'] = min(x_positions)
        stats['max_x'] = max(x_positions)
        stats['min_y'] = min(y_positions)
        stats['max_y'] = max(y_positions)
        
        # Estadísticas por pantalla
        screen_clicks = {}
        for _, _, _, _, screen_idx in clicks_copy:
            screen_clicks[screen_idx] = screen_clicks.get(screen_idx, 0) + 1
        
        for i in range(total_screens):
            stats[f'clicks_screen_{i}'] = screen_clicks.get(i, 0)
        
        # Pantalla más usada
        if screen_clicks:
            most_used_screen = max(screen_clicks.items(), key=lambda x: x[1])[0]
            stats['most_used_screen'] = most_used_screen
        else:
            stats['most_used_screen'] = 0
    else:
        stats['avg_x_position'] = 0
        stats['avg_y_position'] = 0
        stats['min_x'] = 0
        stats['max_x'] = 0
        stats['min_y'] = 0
        stats['max_y'] = 0
        stats['most_used_screen'] = 0
        for i in range(total_screens):
            stats[f'clicks_screen_{i}'] = 0
    
    # Estadísticas de scroll
    if scrolls_copy:
        # Direcciones de scroll
        scroll_directions = {}
        scroll_magnitudes = []
        screen_scrolls = {}
        
        for _, _, dx, dy, _, screen_idx, direction, magnitude in scrolls_copy:
            scroll_directions[direction] = scroll_directions.get(direction, 0) + 1
            scroll_magnitudes.append(magnitude)
            screen_scrolls[screen_idx] = screen_scrolls.get(screen_idx, 0) + 1
        
        stats['scroll_up'] = scroll_directions.get('up', 0)
        stats['scroll_down'] = scroll_directions.get('down', 0)
        stats['scroll_left'] = scroll_directions.get('left', 0)
        stats['scroll_right'] = scroll_directions.get('right', 0)
        stats['avg_scroll_magnitude'] = round(sum(scroll_magnitudes) / len(scroll_magnitudes), 2) if scroll_magnitudes else 0
        
        # Scroll por pantalla
        for i in range(total_screens):
            stats[f'scroll_screen_{i}'] = screen_scrolls.get(i, 0)
    else:
        stats['scroll_up'] = 0
        stats['scroll_down'] = 0
        stats['scroll_left'] = 0
        stats['scroll_right'] = 0
        stats['avg_scroll_magnitude'] = 0
        for i in range(total_screens):
            stats[f'scroll_screen_{i}'] = 0
    
    # Estadísticas de tiempo entre clics
    if inter_click_times_copy:
        avg_inter_click = sum(inter_click_times_copy) / len(inter_click_times_copy)
        stats['avg_inter_click_time_ms'] = round(avg_inter_click * 1000, 2)
    else:
        stats['avg_inter_click_time_ms'] = 0
    
    # Estadísticas de tiempo entre scrolls
    if inter_scroll_times_copy:
        avg_inter_scroll = sum(inter_scroll_times_copy) / len(inter_scroll_times_copy)
        stats['avg_inter_scroll_time_ms'] = round(avg_inter_scroll * 1000, 2)
    else:
        stats['avg_inter_scroll_time_ms'] = 0
    
    # Clicks por segundo
    stats['clicks_per_second'] = round(stats['total_clicks'] / interval_seconds, 2) if interval_seconds > 0 else 0
    
    # Scrolls por segundo
    stats['scrolls_per_second'] = round(stats['scroll_events'] / interval_seconds, 2) if interval_seconds > 0 else 0
    
    return stats

def get_mouse_counts():
    """Obtiene y limpia los conteos del mouse."""
    global mouse_counter
    
    with mouse_lock:
        counts = dict(mouse_counter)
        mouse_counter.clear()
    
    return counts

def start_mouse_listener(blocked_applications):
    """Inicia el listener del mouse."""
    def on_click_wrapper(x, y, button, pressed):
        on_click(x, y, button, pressed, blocked_applications)
    
    def on_scroll_wrapper(x, y, dx, dy):
        on_scroll(x, y, dx, dy, blocked_applications)
    
    with mouse.Listener(
        on_click=on_click_wrapper,
        on_scroll=on_scroll_wrapper
    ) as listener:
        listener.join()

