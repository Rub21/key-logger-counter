# app_detection.py
# Detección de aplicación activa y manejo de aplicaciones bloqueadas

import platform
import json
import logging
from pathlib import Path

# Importar librerías para detectar aplicación activa (opcional)
APP_DETECTION_AVAILABLE = False
try:
    if platform.system() == "Darwin":  # macOS
        from AppKit import NSWorkspace
        APP_DETECTION_AVAILABLE = True
    elif platform.system() == "Windows":
        import win32gui
        import win32process
        import psutil
        APP_DETECTION_AVAILABLE = True
    else:  # Linux
        import subprocess
        import psutil
        APP_DETECTION_AVAILABLE = True
except ImportError:
    pass

def get_active_application():
    """Obtiene el nombre de la aplicación activa."""
    if not APP_DETECTION_AVAILABLE:
        return "Unknown"
    try:
        if platform.system() == "Darwin":  # macOS
            app = NSWorkspace.sharedWorkspace().activeApplication()
            return app.get('NSApplicationName', 'Unknown')
        elif platform.system() == "Windows":
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return psutil.Process(pid).name()
        else:  # Linux
            result = subprocess.run(['xdotool', 'getactivewindow', 'getwindowname'],
                                  capture_output=True, text=True, timeout=0.1)
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
    except:
        return "Unknown"

def get_active_application_info():
    """Obtiene información detallada de la aplicación activa.
    
    Returns:
        dict: Diccionario con información de la aplicación:
            - name: Nombre de la aplicación
            - bundle_id: Bundle ID (macOS) o ruta del ejecutable (Windows/Linux)
            - window_title: Título de la ventana activa
            - process_id: ID del proceso
    """
    if not APP_DETECTION_AVAILABLE:
        return {
            'name': 'Unknown',
            'bundle_id': 'Unknown',
            'window_title': 'Unknown',
            'process_id': None
        }
    
    try:
        if platform.system() == "Darwin":  # macOS
            workspace = NSWorkspace.sharedWorkspace()
            app = workspace.activeApplication()
            
            # Obtener información de la ventana activa
            window_title = "Unknown"
            try:
                # Intentar obtener el título de la ventana activa
                front_app = workspace.frontmostApplication()
                if front_app:
                    # En macOS, obtener el título puede requerir AppleScript o APIs adicionales
                    # Por ahora, usamos el nombre de la aplicación
                    window_title = app.get('NSApplicationName', 'Unknown')
            except:
                pass
            
            return {
                'name': app.get('NSApplicationName', 'Unknown'),
                'bundle_id': app.get('NSApplicationBundleIdentifier', 'Unknown'),
                'window_title': window_title,
                'process_id': app.get('NSApplicationProcessIdentifier', None)
            }
            
        elif platform.system() == "Windows":
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                process = psutil.Process(pid)
                
                # Obtener título de la ventana
                window_title = win32gui.GetWindowText(hwnd)
                
                return {
                    'name': process.name(),
                    'bundle_id': process.exe(),  # Ruta del ejecutable
                    'window_title': window_title if window_title else 'Unknown',
                    'process_id': pid
                }
            else:
                return {
                    'name': 'Unknown',
                    'bundle_id': 'Unknown',
                    'window_title': 'Unknown',
                    'process_id': None
                }
        else:  # Linux
            # Obtener nombre de la aplicación
            result_name = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowname'],
                capture_output=True, text=True, timeout=0.1
            )
            window_title = result_name.stdout.strip() if result_name.returncode == 0 else "Unknown"
            
            # Obtener PID del proceso
            result_pid = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowpid'],
                capture_output=True, text=True, timeout=0.1
            )
            pid = None
            if result_pid.returncode == 0:
                try:
                    pid = int(result_pid.stdout.strip())
                except:
                    pass
            
            # Obtener nombre del proceso
            process_name = "Unknown"
            if pid:
                try:
                    process = psutil.Process(pid)
                    process_name = process.name()
                    exe_path = process.exe()
                except:
                    exe_path = "Unknown"
            else:
                exe_path = "Unknown"
            
            return {
                'name': process_name,
                'bundle_id': exe_path,
                'window_title': window_title,
                'process_id': pid
            }
    except Exception as e:
        logging.debug(f"Error getting app info: {e}")
        return {
            'name': 'Unknown',
            'bundle_id': 'Unknown',
            'window_title': 'Unknown',
            'process_id': None
        }

def load_blocked_applications(blocked_apps_file):
    """Carga aplicaciones bloqueadas desde JSON."""
    try:
        if blocked_apps_file.exists():
            with open(blocked_apps_file, 'r', encoding='utf-8') as f:
                return json.load(f).get('blocked_applications', [])
        else:
            # Crear archivo por defecto
            default = {
                "blocked_applications": [
                    "Google Chrome", "Chrome", "Safari", "Firefox", "Microsoft Edge",
                    "1Password", "LastPass", "Bitwarden", "KeePass"
                ]
            }
            with open(blocked_apps_file, 'w', encoding='utf-8') as f:
                json.dump(default, f, indent=2)
            return default['blocked_applications']
    except Exception as e:
        logging.error(f"Error loading blocked apps: {e}")
        return []

def is_blocked_application(app_name, blocked_applications):
    """Verifica si la aplicación está bloqueada."""
    if not app_name or app_name == "Unknown":
        return False
    app_lower = app_name.lower()
    return any(blocked.lower() in app_lower for blocked in blocked_applications)

