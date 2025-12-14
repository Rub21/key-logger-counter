# Keylogger de Conteo de Caracteres


Keylogger que cuenta cuántas veces se presionó cada carácter en intervalos de 5 segundos. **NO guarda palabras completas ni el orden de los caracteres**, solo conteos agregados.


## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements-minimal.txt

# 2. Ejecutar
python main.py

# 3. Usar teclado y mouse normalmente...

# 4. Detener con Ctrl+C (convierte automáticamente a CSV)
```

## ✨ Características

- ✅ **Teclado**: Cuenta caracteres presionados cada 5 segundos
- ✅ **Mouse**: Captura clics, posición, scroll y pantalla usada
- ✅ **Aplicación**: Detecta nombre, bundle ID, título de ventana y PID
- ✅ **Estadísticas**: Velocidad de escritura, tiempos entre teclas/clics, etc.
- ✅ **Combinaciones**: Detecta automáticamente Ctrl+C, Shift+A, etc.
- ✅ **Múltiples pantallas**: Identifica en qué pantalla ocurren los eventos
- ✅ **Scroll mejorado**: Dirección, magnitud y velocidad del scroll
- ✅ **Bloqueo inteligente**: No captura en navegadores ni gestores de contraseñas

## 📋 Requisitos

- Python 3.7 o superior
- macOS, Windows o Linux

## 📁 Estructura del Proyecto

```
keylogger-simulator/
├── src/                      # Código fuente organizado en módulos
│   ├── __init__.py
│   ├── config.py            # Configuración y constantes
│   ├── app_detection.py     # Detección de aplicación activa
│   ├── key_processing.py    # Procesamiento de teclas y estadísticas
│   ├── mouse_tracking.py    # Tracking de eventos del mouse
│   ├── screen_detection.py  # Detección de múltiples pantallas
│   ├── file_handler.py      # Manejo de archivos CSV/JSON
│   └── keylogger.py         # Lógica principal del keylogger
├── keyboard_data/            # Carpeta donde se guardan los CSVs y JSONs temporales
├── main.py                   # Script principal (punto de entrada)
├── blocked_apps.json         # Configuración de apps bloqueadas
├── key_combinations.json     # Combinaciones de teclas detectadas
└── requirements-minimal.txt  # Dependencias
```

## ✨ Qué Captura

### ⌨️ Teclado
- Conteo de cada tecla presionada (a-z, 0-9, símbolos, teclas especiales)
- Combinaciones de teclas (Ctrl+C, Shift+A, Cmd+V, etc.)
- Estadísticas temporales: velocidad, tiempos de presión, intervalos entre teclas

### 🖱️ Mouse
- **Clics**: Total, izquierdo, derecho, medio
- **Posición**: Coordenadas X/Y promedio, mínimas y máximas
- **Scroll**: Dirección (arriba, abajo, izquierda, derecha), magnitud, velocidad
- **Pantallas**: Detecta múltiples monitores y registra en qué pantalla ocurren los eventos
- **Estadísticas**: Tiempo entre clics, velocidad de clics, velocidad de scroll

### 💻 Aplicación
- Nombre de la aplicación activa
- Bundle ID (macOS) o ruta del ejecutable (Windows/Linux)
- Título de la ventana activa
- Process ID (PID)

## 📁 Archivos Generados

Los datos se guardan en `keyboard_data/` con formato:

- `data_YYYYMMDD_HHMMSS.csv` - Datos finales (ej: `data_20251214_143022.csv`)
- `data_YYYYMMDD_HHMMSS.json` - Temporal (se elimina al finalizar)

**Ventaja**: Los nombres con timestamp permiten ordenar y combinar fácilmente múltiples sesiones.

## 📊 Formato CSV

Cada fila = 5 segundos de actividad (teclado + mouse + aplicación):

```csv
timestamp,a,b,c,...,total_clicks,left_clicks,scroll_up,scroll_down,clicks_screen_0,clicks_screen_1,active_application,...
1703123456.789,5,2,1,...,15,12,5,3,12,3,"Safari",...
```

### Columnas Principales

**Teclado**:
- `timestamp` - Marca de tiempo Unix
- `a`, `b`, `c`... `z` - Conteo de cada letra
- `0`, `1`... `9` - Conteo de cada número
- `[ENTER]`, `[BACKSPACE]`, etc. - Teclas especiales
- `[CTRL_L]+c`, `[SHIFT_L]+a` - Combinaciones detectadas

**Mouse - Clics**:
- `total_clicks` - Total de clics en el intervalo
- `left_clicks`, `right_clicks`, `middle_clicks` - Clics por botón
- `avg_x_position`, `avg_y_position` - Posición promedio de clics
- `min_x`, `max_x`, `min_y`, `max_y` - Área de interacción
- `clicks_per_second` - Velocidad de clics

**Mouse - Scroll**:
- `scroll_events` - Total de eventos de scroll
- `scroll_up`, `scroll_down`, `scroll_left`, `scroll_right` - Dirección del scroll
- `avg_scroll_magnitude` - Intensidad promedio del scroll
- `scrolls_per_second` - Velocidad de scrolls
- `avg_inter_scroll_time_ms` - Tiempo promedio entre scrolls

**Mouse - Pantallas**:
- `total_screens` - Número de pantallas detectadas (1, 2, 3...)
- `most_used_screen` - Índice de la pantalla más usada
- `clicks_screen_0`, `clicks_screen_1`, `clicks_screen_2` - Clics por pantalla
- `scroll_screen_0`, `scroll_screen_1`, `scroll_screen_2` - Scroll por pantalla

**Aplicación**:
- `active_application` - Nombre de la app activa
- `app_bundle_id` - Bundle ID o ruta del ejecutable
- `app_window_title` - Título de la ventana
- `app_process_id` - ID del proceso

**Estadísticas del Teclado**:
- `keystrokes_per_second` - Velocidad de escritura
- `avg_hold_time_ms` - Tiempo promedio de presión
- `avg_inter_key_time_ms` - Tiempo promedio entre teclas
- `total_keystrokes` - Total de teclas en el intervalo

## 🔒 Aplicaciones Bloqueadas

Por defecto NO captura datos en:
- Navegadores (Chrome, Safari, Firefox)
- Gestores de contraseñas (1Password, LastPass)

Edita `blocked_apps.json` para agregar/quitar apps.

## 📋 Requisitos

- Python 3.7+
- macOS, Windows o Linux
- `pynput` y `psutil` (instalación automática)

## 🔍 Ejemplos

### Ejemplo 1: Solo Teclado
Si escribes "hola" en 5 segundos:
```
h: 1, o: 1, l: 1, a: 1
total_keystrokes: 4
keystrokes_per_second: 0.8
(todas las demás columnas: 0)
```

### Ejemplo 2: Teclado + Mouse
Si escribes y haces clics en 5 segundos:
```
h: 1, o: 1, l: 1, a: 1
total_clicks: 5
left_clicks: 4, right_clicks: 1
scroll_up: 3, scroll_down: 2
clicks_screen_0: 5  (todos en pantalla principal)
most_used_screen: 0
```

**NO se puede reconstruir** que escribiste "hola" o qué hiciste exactamente, solo conteos y estadísticas.

## 🖥️ Detección de Múltiples Pantallas

El sistema detecta automáticamente cuántas pantallas tienes conectadas y registra en qué pantalla ocurren los eventos:

- **1 pantalla**: Solo `clicks_screen_0` y `scroll_screen_0`
- **2 pantallas**: `clicks_screen_0`, `clicks_screen_1`, `scroll_screen_0`, `scroll_screen_1`
- **3+ pantallas**: Se agregan campos dinámicamente

El campo `most_used_screen` indica qué pantalla tuvo más actividad (clics + scrolls) en cada intervalo.

## 📚 Documentación Completa

Ver `DICCIONARIO_DATOS.md` para la lista completa de campos, tipos de datos, unidades y ejemplos detallados.

## ⚠️ Uso Responsable

- Solo para uso educativo y con consentimiento
- NO usar en sistemas de otros sin permiso
- Revisar datos antes de compartir

---

**Uso educativo y de investigación únicamente** 🛡️
