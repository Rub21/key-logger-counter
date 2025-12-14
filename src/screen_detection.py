# screen_detection.py
# Detección de pantallas/monitores y determinación de pantalla activa

import platform
import logging

# Información de pantallas
screens_info = []
screens_initialized = False

def get_screens_info():
    """Obtiene información de todas las pantallas disponibles.
    
    Returns:
        list: Lista de diccionarios con información de cada pantalla:
            - index: Índice de la pantalla (0, 1, 2, ...)
            - x: Posición X del inicio de la pantalla
            - y: Posición Y del inicio de la pantalla
            - width: Ancho de la pantalla en píxeles
            - height: Alto de la pantalla en píxeles
            - is_primary: Si es la pantalla principal
    """
    global screens_info, screens_initialized
    
    if screens_initialized:
        return screens_info
    
    screens_info = []
    
    try:
        if platform.system() == "Darwin":  # macOS
            try:
                from AppKit import NSScreen
                
                screens = NSScreen.screens()
                for idx, screen in enumerate(screens):
                    frame = screen.frame()
                    
                    # Obtener dimensiones
                    width = int(frame.size.width)
                    height = int(frame.size.height)
                    x = int(frame.origin.x)
                    y = int(frame.origin.y)
                    
                    # Determinar si es la pantalla principal (la que tiene y=0 o x=0)
                    is_primary = (idx == 0) or (x == 0 and y == 0)
                    
                    screens_info.append({
                        'index': idx,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'is_primary': is_primary
                    })
            except ImportError:
                # Fallback: usar system_profiler o tkinter
                try:
                    import subprocess
                    import json
                    
                    # Intentar usar system_profiler para obtener información de pantallas
                    result = subprocess.run(
                        ['system_profiler', 'SPDisplaysDataType', '-json'],
                        capture_output=True, text=True, timeout=2
                    )
                    
                    if result.returncode == 0:
                        try:
                            data = json.loads(result.stdout)
                            displays = data.get('SPDisplaysDataType', [])
                            
                            for idx, display in enumerate(displays):
                                # Obtener resolución
                                resolution = display.get('spdisplays_resolution', '')
                                if resolution:
                                    try:
                                        # Formato típico: "1920 x 1080 @ 60Hz"
                                        parts = resolution.split(' x ')
                                        if len(parts) >= 2:
                                            width = int(parts[0].strip())
                                            height_part = parts[1].split(' @')[0].strip()
                                            height = int(height_part)
                                            
                                            screens_info.append({
                                                'index': idx,
                                                'x': 0 if idx == 0 else idx * width,  # Aproximación
                                                'y': 0,
                                                'width': width,
                                                'height': height,
                                                'is_primary': idx == 0
                                            })
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Si no se encontraron pantallas, usar una pantalla por defecto
                    if not screens_info:
                        screens_info.append({
                            'index': 0,
                            'x': 0,
                            'y': 0,
                            'width': 1920,
                            'height': 1080,
                            'is_primary': True
                        })
                except Exception as e:
                    logging.debug(f"Error detecting screens on macOS: {e}")
                    # Último fallback
                    screens_info.append({
                        'index': 0,
                        'x': 0,
                        'y': 0,
                        'width': 1920,
                        'height': 1080,
                        'is_primary': True
                    })
        
        elif platform.system() == "Windows":
            try:
                import win32api
                import win32con
                
                def get_monitor_info(monitor_handle):
                    """Obtiene información de un monitor."""
                    monitor_info = win32api.GetMonitorInfo(monitor_handle)
                    work_area = monitor_info['Work']
                    monitor_area = monitor_info['Monitor']
                    
                    return {
                        'x': monitor_area[0],
                        'y': monitor_area[1],
                        'width': monitor_area[2] - monitor_area[0],
                        'height': monitor_area[3] - monitor_area[1],
                        'is_primary': monitor_info['Flags'] == win32con.MONITORINFOF_PRIMARY
                    }
                
                def enum_display_monitors_callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    """Callback para enumerar monitores."""
                    info = get_monitor_info(hMonitor)
                    info['index'] = len(screens_info)
                    screens_info.append(info)
                    return True
                
                win32api.EnumDisplayMonitors(None, None, enum_display_monitors_callback, None)
                
            except ImportError:
                # Fallback: usar una sola pantalla por defecto
                screens_info.append({
                    'index': 0,
                    'x': 0,
                    'y': 0,
                    'width': 1920,
                    'height': 1080,
                    'is_primary': True
                })
        
        else:  # Linux
            try:
                import subprocess
                import json
                
                # Intentar usar xrandr
                result = subprocess.run(
                    ['xrandr', '--query'],
                    capture_output=True, text=True, timeout=1
                )
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    primary_index = 0
                    current_index = 0
                    
                    for line in lines:
                        if ' connected' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                # Buscar resolución
                                for part in parts:
                                    if 'x' in part and '+' in part:
                                        try:
                                            res_part = part.split('+')[0]
                                            width, height = map(int, res_part.split('x'))
                                            pos_part = part.split('+')[1:]
                                            x = int(pos_part[0]) if len(pos_part) > 0 else 0
                                            y = int(pos_part[1]) if len(pos_part) > 1 else 0
                                            
                                            is_primary = 'primary' in line.lower() or current_index == 0
                                            
                                            screens_info.append({
                                                'index': current_index,
                                                'x': x,
                                                'y': y,
                                                'width': width,
                                                'height': height,
                                                'is_primary': is_primary
                                            })
                                            current_index += 1
                                        except:
                                            pass
                
                # Si no se encontraron pantallas, usar fallback
                if not screens_info:
                    screens_info.append({
                        'index': 0,
                        'x': 0,
                        'y': 0,
                        'width': 1920,
                        'height': 1080,
                        'is_primary': True
                    })
            
            except Exception as e:
                logging.debug(f"Error detecting screens on Linux: {e}")
                # Fallback a una pantalla
                screens_info.append({
                    'index': 0,
                    'x': 0,
                    'y': 0,
                    'width': 1920,
                    'height': 1080,
                    'is_primary': True
                })
    
    except Exception as e:
        logging.error(f"Error detecting screens: {e}")
        # Fallback a una pantalla por defecto
        screens_info.append({
            'index': 0,
            'x': 0,
            'y': 0,
            'width': 1920,
            'height': 1080,
            'is_primary': True
        })
    
    screens_initialized = True
    return screens_info

def get_screen_for_position(x, y):
    """Determina en qué pantalla está una posición (x, y).
    
    Args:
        x: Coordenada X
        y: Coordenada Y
    
    Returns:
        dict: Información de la pantalla o None si no se encuentra
    """
    screens = get_screens_info()
    
    for screen in screens:
        screen_x = screen['x']
        screen_y = screen['y']
        screen_width = screen['width']
        screen_height = screen['height']
        
        # Verificar si la posición está dentro de esta pantalla
        if (screen_x <= x < screen_x + screen_width and 
            screen_y <= y < screen_y + screen_height):
            return screen
    
    # Si no se encuentra, retornar la pantalla principal o la primera
    for screen in screens:
        if screen.get('is_primary', False):
            return screen
    
    return screens[0] if screens else None

def get_total_screens():
    """Obtiene el número total de pantallas."""
    return len(get_screens_info())

def get_primary_screen():
    """Obtiene la información de la pantalla principal."""
    screens = get_screens_info()
    for screen in screens:
        if screen.get('is_primary', False):
            return screen
    return screens[0] if screens else None

