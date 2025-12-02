# key-logger-char-count.py
# Script principal del keylogger

import sys
import logging
import traceback
from pathlib import Path

from src.config import char_count_file, STATS_INTERVAL, log_system, blocked_apps_file, load_combinations
from src.app_detection import load_blocked_applications
from src.file_handler import setup_temp_json
from src.keylogger import start_listener, save_character_counts, cleanup

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_system), logging.StreamHandler(sys.stdout)]
)

def main():
    """Función principal."""
    # Cargar combinaciones existentes
    combinations = load_combinations()
    
    # Cargar aplicaciones bloqueadas
    blocked_applications = load_blocked_applications(blocked_apps_file)
    
    # Configurar archivo JSON temporal (se convertirá a CSV al finalizar)
    setup_temp_json()
    
    print("="*70)
    print("KEYLOGGER DE CONTEO DE CARACTERES")
    print("="*70)
    print(f"✓ Acumula pulsaciones en MEMORIA durante {STATS_INTERVAL} segundos")
    print(f"✓ Guarda datos temporales en JSON cada {STATS_INTERVAL} segundos")
    print(f"✓ Convierte JSON a CSV al finalizar (Ctrl+C)")
    print(f"✓ NO guarda orden ni palabras completas - Solo conteos")
    print(f"✓ Captura combinaciones de teclas (Ctrl+C, Shift+A, etc.)")
    print(f"✓ Combinaciones conocidas: {len(combinations)}")
    print(f"✓ Aplicaciones bloqueadas: {len(blocked_applications)}")
    print(f"✓ CSV final: {char_count_file}")
    print(f"\nFlujo: Presionar teclas → Acumular en memoria → Guardar JSON cada {STATS_INTERVAL}s → CSV al finalizar")
    print(f"Presiona Ctrl+C para detener y convertir a CSV\n")
    
    # Iniciar guardado periódico
    save_character_counts(blocked_applications)
    
    # Iniciar listener del teclado
    start_listener(blocked_applications)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        cleanup()
