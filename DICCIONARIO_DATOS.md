# Diccionario de Datos - Keylogger Simulator

Este documento describe todos los datos recolectados por el keylogger y su estructura en el archivo CSV final.

## Estructura General

Los datos se guardan en intervalos de tiempo (por defecto cada 5 segundos). Cada fila del CSV representa un intervalo y contiene:

1. **Timestamp**: Marca de tiempo del intervalo
2. **Datos de teclas**: Conteos de cada tecla presionada
3. **Estadísticas del teclado**: Métricas calculadas sobre el uso del teclado
4. **Estadísticas del mouse**: Métricas calculadas sobre el uso del mouse
5. **Información de aplicación**: Datos sobre la aplicación activa

---

## 1. Timestamp

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `timestamp` | float | Timestamp Unix (segundos desde 1970-01-01) del inicio del intervalo | `1703123456.789` |

---

## 2. Datos de Teclas

### 2.1 Teclas Básicas

El keylogger cuenta cuántas veces se presionó cada tecla en el intervalo. Las columnas pueden tener valores enteros (0 o mayor).

#### Letras (a-z)
- **Formato**: Caracteres minúsculos
- **Valores**: Enteros (conteo de pulsaciones)
- **Ejemplo**: `a`, `b`, `c`, ..., `z`
- **Nota**: Todas las letras se convierten a minúsculas

#### Números (0-9)
- **Formato**: Caracteres numéricos
- **Valores**: Enteros (conteo de pulsaciones)
- **Ejemplo**: `0`, `1`, `2`, ..., `9`

#### Caracteres Especiales
- **Formato**: Caracteres imprimibles
- **Valores**: Enteros (conteo de pulsaciones)
- **Caracteres incluidos**: ` ` (espacio), `!`, `@`, `#`, `$`, `%`, `^`, `&`, `*`, `(`, `)`, `-`, `_`, `=`, `+`, `[`, `]`, `{`, `}`, `\`, `|`, `;`, `:`, `'`, `"`, `,`, `.`, `<`, `>`, `/`, `?`, `` ` ``, `~`

#### Teclas Especiales del Teclado
| Campo | Descripción | Ejemplo de valor |
|-------|-------------|------------------|
| `[ENTER]` | Tecla Enter/Return | `5` (5 veces presionada) |
| `[BACKSPACE]` | Tecla Backspace | `3` |
| `[DELETE]` | Tecla Delete | `1` |
| `[TAB]` | Tecla Tab | `2` |
| `[ESC]` | Tecla Escape | `0` |
| `[UP]` | Flecha arriba | `4` |
| `[DOWN]` | Flecha abajo | `2` |
| `[LEFT]` | Flecha izquierda | `1` |
| `[RIGHT]` | Flecha derecha | `3` |
| `[HOME]` | Tecla Home | `0` |
| `[END]` | Tecla End | `0` |
| `[PAGE_UP]` | Página arriba | `0` |
| `[PAGE_DOWN]` | Página abajo | `0` |

#### Teclas Modificadoras
| Campo | Descripción | Ejemplo de valor |
|-------|-------------|------------------|
| `[CTRL]` | Control genérico | `10` |
| `[CTRL_L]` | Control izquierdo | `8` |
| `[CTRL_R]` | Control derecho | `2` |
| `[ALT]` | Alt genérico | `5` |
| `[ALT_L]` | Alt izquierdo | `4` |
| `[ALT_R]` | Alt derecho | `1` |
| `[SHIFT]` | Shift genérico | `15` |
| `[SHIFT_L]` | Shift izquierdo | `12` |
| `[SHIFT_R]` | Shift derecho | `3` |
| `[CMD]` | Command (Mac) genérico | `7` |
| `[CMD_L]` | Command izquierdo (Mac) | `6` |
| `[CMD_R]` | Command derecho (Mac) | `1` |

### 2.2 Combinaciones de Teclas

Las combinaciones se detectan automáticamente cuando se presiona una tecla modificadora junto con otra tecla.

- **Formato**: `[MODIFICADOR]+[TECLA]`
- **Valores**: Enteros (conteo de combinaciones)
- **Ejemplos**:
  - `[CTRL_L]+c` (Ctrl+C)
  - `[SHIFT_L]+a` (Shift+A)
  - `[CMD_L]+v` (Cmd+V en Mac)
  - `[ALT_L]+f4` (Alt+F4)

**Nota**: Las combinaciones se agregan dinámicamente al CSV cuando se detectan por primera vez.

---

## 3. Estadísticas del Teclado

| Campo | Tipo | Unidad | Descripción | Ejemplo |
|-------|------|--------|-------------|---------|
| `avg_hold_time_ms` | float | milisegundos | Tiempo promedio que se mantiene presionada una tecla | `125.50` |
| `avg_inter_key_time_ms` | float | milisegundos | Tiempo promedio entre pulsaciones de teclas consecutivas | `250.75` |
| `std_inter_key_time_ms` | float | milisegundos | Desviación estándar del tiempo entre pulsaciones (variabilidad) | `85.30` |
| `min_inter_key_time_ms` | float | milisegundos | Tiempo mínimo entre dos pulsaciones consecutivas | `50.00` |
| `max_inter_key_time_ms` | float | milisegundos | Tiempo máximo entre dos pulsaciones consecutivas | `1500.00` |
| `keystrokes_per_second` | float | pulsaciones/segundo | Velocidad de escritura (WPM aproximado = keystrokes_per_second * 12) | `8.50` |
| `total_keystrokes` | integer | conteo | Total de teclas presionadas en el intervalo | `42` |

**Notas**:
- Si no hay teclas presionadas en el intervalo, todos los valores serán `0`
- `keystrokes_per_second` se calcula dividiendo `total_keystrokes` entre la duración del intervalo (5 segundos por defecto)

---

## 4. Estadísticas del Mouse

### 4.1 Estadísticas Básicas de Clics

| Campo | Tipo | Unidad | Descripción | Ejemplo |
|-------|------|--------|-------------|---------|
| `total_clicks` | integer | conteo | Total de clics realizados en el intervalo | `15` |
| `left_clicks` | integer | conteo | Clics con el botón izquierdo del mouse | `12` |
| `right_clicks` | integer | conteo | Clics con el botón derecho del mouse | `2` |
| `middle_clicks` | integer | conteo | Clics con el botón medio/rueda del mouse | `1` |
| `avg_x_position` | float | píxeles | Posición X promedio de todos los clics en la pantalla | `856.50` |
| `avg_y_position` | float | píxeles | Posición Y promedio de todos los clics en la pantalla | `432.25` |
| `min_x` | integer | píxeles | Posición X mínima de los clics | `200` |
| `max_x` | integer | píxeles | Posición X máxima de los clics | `1680` |
| `min_y` | integer | píxeles | Posición Y mínima de los clics | `100` |
| `max_y` | integer | píxeles | Posición Y máxima de los clics | `900` |
| `avg_inter_click_time_ms` | float | milisegundos | Tiempo promedio entre clics consecutivos | `350.20` |
| `clicks_per_second` | float | clics/segundo | Velocidad de clics (clics por segundo) | `3.00` |

### 4.2 Estadísticas de Scroll

| Campo | Tipo | Unidad | Descripción | Ejemplo |
|-------|------|--------|-------------|---------|
| `scroll_events` | integer | conteo | Total de eventos de scroll (rueda del mouse) | `8` |
| `scroll_up` | integer | conteo | Eventos de scroll hacia arriba | `5` |
| `scroll_down` | integer | conteo | Eventos de scroll hacia abajo | `3` |
| `scroll_left` | integer | conteo | Eventos de scroll hacia izquierda | `0` |
| `scroll_right` | integer | conteo | Eventos de scroll hacia derecha | `0` |
| `avg_scroll_magnitude` | float | píxeles | Magnitud promedio del scroll (distancia/intensidad) | `12.50` |
| `avg_inter_scroll_time_ms` | float | milisegundos | Tiempo promedio entre eventos de scroll consecutivos | `250.75` |
| `scrolls_per_second` | float | scrolls/segundo | Velocidad de scrolls (scrolls por segundo) | `1.60` |

**Notas sobre Scroll**:
- `scroll_up` y `scroll_down` son los más comunes (scroll vertical)
- `scroll_left` y `scroll_right` son menos comunes (scroll horizontal, trackpad)
- `avg_scroll_magnitude` mide la intensidad del scroll (mayor = scroll más rápido/intenso)
- `scrolls_per_second` se calcula dividiendo `scroll_events` entre la duración del intervalo

### 4.3 Estadísticas de Pantallas/Monitores

| Campo | Tipo | Unidad | Descripción | Ejemplo |
|-------|------|--------|-------------|---------|
| `total_screens` | integer | conteo | Número total de pantallas detectadas | `2` |
| `most_used_screen` | integer | índice | Índice de la pantalla más usada (0, 1, 2...) | `0` |
| `clicks_screen_0` | integer | conteo | Clics realizados en la pantalla 0 (principal) | `12` |
| `clicks_screen_1` | integer | conteo | Clics realizados en la pantalla 1 (secundaria) | `3` |
| `clicks_screen_2` | integer | conteo | Clics realizados en la pantalla 2 (terciaria) | `0` |
| `scroll_screen_0` | integer | conteo | Eventos de scroll en la pantalla 0 | `8` |
| `scroll_screen_1` | integer | conteo | Eventos de scroll en la pantalla 1 | `2` |
| `scroll_screen_2` | integer | conteo | Eventos de scroll en la pantalla 2 | `0` |

**Notas sobre Pantallas**:
- Los campos `clicks_screen_X` y `scroll_screen_X` se agregan dinámicamente según el número de pantallas detectadas
- Si hay 1 pantalla: solo `clicks_screen_0` y `scroll_screen_0`
- Si hay 2 pantallas: `clicks_screen_0`, `clicks_screen_1`, `scroll_screen_0`, `scroll_screen_1`
- Si hay 3+ pantallas: se agregan campos adicionales automáticamente
- `most_used_screen` indica qué pantalla tuvo más actividad (clics + scrolls) en el intervalo
- La pantalla 0 es típicamente la pantalla principal

**Notas Generales**:
- Las coordenadas (X, Y) son absolutas en el espacio de pantallas combinadas (0,0 es la esquina superior izquierda de la pantalla principal)
- Si no hay clics en el intervalo, los valores de posición serán `0` y los conteos serán `0`
- `clicks_per_second` se calcula dividiendo `total_clicks` entre la duración del intervalo
- El sistema detecta automáticamente el número de pantallas al iniciar

---

## 5. Información de la Aplicación Activa

| Campo | Tipo | Descripción | Ejemplo (macOS) | Ejemplo (Windows) | Ejemplo (Linux) |
|-------|------|-------------|-----------------|-------------------|-----------------|
| `active_application` | string | Nombre de la aplicación activa | `"Safari"` | `"chrome.exe"` | `"firefox"` |
| `app_bundle_id` | string | Bundle ID (macOS) o ruta del ejecutable (Windows/Linux) | `"com.apple.Safari"` | `"C:\Program Files\Google\Chrome\Application\chrome.exe"` | `"/usr/bin/firefox"` |
| `app_window_title` | string | Título de la ventana activa | `"Google - Safari"` | `"Google - Google Chrome"` | `"Mozilla Firefox"` |
| `app_process_id` | integer/string | ID del proceso (PID) | `12345` | `6789` | `5432` |

**Notas**:
- En macOS: `app_bundle_id` contiene el Bundle Identifier (ej: `com.apple.Safari`)
- En Windows: `app_bundle_id` contiene la ruta completa del ejecutable
- En Linux: `app_bundle_id` contiene la ruta del ejecutable si está disponible
- Si la aplicación no se puede detectar, los valores serán `"Unknown"` o vacíos
- `app_process_id` puede estar vacío si no se puede obtener

---

## Estructura del CSV

El CSV tiene el siguiente orden de columnas:

1. `timestamp`
2. Teclas básicas (a-z, 0-9, caracteres especiales, teclas especiales)
3. Combinaciones dinámicas (se agregan cuando se detectan)
4. Estadísticas del teclado (`STATS_HEADERS`)
5. Estadísticas del mouse (`MOUSE_STATS_HEADERS`)
6. Información de aplicación (`APP_INFO_HEADERS`)

### Ejemplo de Fila CSV

```csv
timestamp,a,b,c,...,z,0,1,...,9,[ENTER],[CTRL_L]+c,...,avg_hold_time_ms,...,total_clicks,...,active_application,...
1703123456.789,5,2,1,...,0,0,1,...,0,2,...,125.50,...,15,...,"Safari",...
```

---

## Formato JSON Temporal

Antes de convertirse a CSV, los datos se guardan en un archivo JSON temporal con esta estructura:

```json
{
  "description": "Datos temporales del keylogger",
  "intervals": [
    {
      "timestamp": 1703123456.789,
      "active_application": "Safari",
      "app_info": {
        "name": "Safari",
        "bundle_id": "com.apple.Safari",
        "window_title": "Google - Safari",
        "process_id": 12345
      },
      "counts": {
        "a": 5,
        "b": 2,
        "[ENTER]": 2,
        "[CTRL_L]+c": 1
      },
      "stats": {
        "avg_hold_time_ms": 125.50,
        "avg_inter_key_time_ms": 250.75,
        "std_inter_key_time_ms": 85.30,
        "min_inter_key_time_ms": 50.00,
        "max_inter_key_time_ms": 1500.00,
        "keystrokes_per_second": 8.50,
        "total_keystrokes": 42
      },
      "mouse_stats": {
        "total_clicks": 15,
        "left_clicks": 12,
        "right_clicks": 2,
        "middle_clicks": 1,
        "scroll_events": 8,
        "avg_x_position": 856.50,
        "avg_y_position": 432.25,
        "min_x": 200,
        "max_x": 1680,
        "min_y": 100,
        "max_y": 900,
        "avg_inter_click_time_ms": 350.20,
        "clicks_per_second": 3.00,
        "total_screens": 2,
        "most_used_screen": 0,
        "clicks_screen_0": 12,
        "clicks_screen_1": 3,
        "scroll_screen_0": 8,
        "scroll_screen_1": 0,
        "scroll_up": 5,
        "scroll_down": 3,
        "scroll_left": 0,
        "scroll_right": 0,
        "avg_scroll_magnitude": 12.50,
        "avg_inter_scroll_time_ms": 250.75,
        "scrolls_per_second": 1.60
      },
      "mouse_counts": {
        "click_left": 12,
        "click_right": 2,
        "click_middle": 1,
        "total_clicks": 15,
        "scroll_events": 8
      }
    }
  ]
}
```

---

## Casos Especiales

### Aplicaciones Bloqueadas
Si la aplicación activa está en la lista de aplicaciones bloqueadas (`blocked_apps.json`), **NO se guardan datos** para ese intervalo. El intervalo se omite completamente.

### Intervalos Vacíos
Si en un intervalo no hay:
- Teclas presionadas
- Clics del mouse
- Eventos de scroll

El intervalo **NO se guarda** (se omite).

### Valores por Defecto
- Si no hay datos para una métrica, el valor será `0` (para números) o `"Unknown"` (para strings)
- Las posiciones del mouse serán `0` si no hay clics
- Los tiempos serán `0` si no hay eventos suficientes para calcularlos

---

## Uso para Análisis

### Métricas de Productividad
- `keystrokes_per_second`: Velocidad de escritura
- `total_keystrokes`: Actividad total del teclado
- `clicks_per_second`: Actividad del mouse

### Patrones de Uso
- `active_application`: Qué aplicación se usa más
- `app_window_title`: Qué tareas específicas se realizan
- Distribución de teclas: Qué teclas se usan más frecuentemente

### Análisis de Comportamiento
- `avg_inter_key_time_ms` y `std_inter_key_time_ms`: Patrones de escritura (ritmo, pausas)
- `avg_hold_time_ms`: Tiempo de presión de teclas
- Posiciones del mouse: Áreas de la pantalla más utilizadas
- Combinaciones de teclas: Atajos de teclado más usados
- Distribución por pantallas: Qué pantalla se usa más (`clicks_screen_X`, `scroll_screen_X`)
- Patrones de scroll: Dirección preferida (`scroll_up`, `scroll_down`) e intensidad (`avg_scroll_magnitude`)

### Correlación Temporal
- El `timestamp` permite correlacionar eventos del teclado y mouse con la aplicación activa
- Se pueden identificar cambios de contexto (cambio de aplicación) y su impacto en la actividad

---

## Notas Técnicas

- **Intervalo de guardado**: Por defecto 5 segundos (configurable en `STATS_INTERVAL`)
- **Resolución temporal**: Los datos se agregan por intervalo, no son eventos individuales
- **Precisión**: Los tiempos están en milisegundos con 2 decimales
- **Coordenadas del mouse**: Absolutas en píxeles de la pantalla principal
- **Thread-safe**: Los datos se capturan de forma thread-safe para evitar pérdida de información

---

## Archivos Relacionados

- `keyboard_data/*.csv`: Archivos CSV finales con todos los datos
- `keyboard_data/*.json`: Archivos JSON temporales (se eliminan después de convertir a CSV)
- `key_combinations.json`: Lista de todas las combinaciones de teclas detectadas
- `blocked_apps.json`: Lista de aplicaciones bloqueadas (no se capturan datos)

---

**Última actualización**: 
- Tracking de mouse con detección de múltiples pantallas
- Tracking mejorado de scroll (dirección, magnitud, velocidad)
- Estadísticas por pantalla (clics y scroll por monitor)
- Detección mejorada de aplicaciones

