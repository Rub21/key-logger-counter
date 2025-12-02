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

