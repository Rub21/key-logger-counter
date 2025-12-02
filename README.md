# Keylogger de Conteo de Caracteres

Keylogger que cuenta cuántas veces se presionó cada carácter en intervalos de 5 segundos. **NO guarda palabras completas ni el orden de los caracteres**, solo conteos agregados.

## 🎯 Características

- ✅ Cuenta caracteres presionados cada 5 segundos
- ✅ Guarda solo conteos (ej: 'a': 5, 'b': 3)
- ✅ NO guarda el orden ni palabras completas
- ✅ Incluye nombre de la aplicación activa
- ✅ Captura combinaciones de teclas (Ctrl+C, Shift+A, etc.)
- ✅ Guarda datos temporales en JSON durante la ejecución
- ✅ Convierte automáticamente a CSV al finalizar (Ctrl+C)
- ✅ Bloquea aplicaciones sensibles (navegadores, gestores de contraseñas)

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
│   ├── file_handler.py      # Manejo de archivos CSV/JSON
│   └── keylogger.py         # Lógica principal del keylogger
├── keyboard_data/            # Carpeta donde se guardan los CSVs y JSONs temporales
├── main.py                   # Script principal (punto de entrada)
├── blocked_apps.json         # Configuración de apps bloqueadas
├── key_combinations.json     # Combinaciones de teclas detectadas
└── requirements-minimal.txt  # Dependencias
```

## 🚀 Instalación


### Opción 1: Instalación mínima (si hay problemas)
Si tienes problemas con las dependencias opcionales, usa la versión mínima:
```bash
pip install -r requirements-minimal.txt
```
El keylogger funcionará pero mostrará "Unknown" en `active_application`.

**Si tienes problemas con dependencias opcionales:**

**macOS** - Si `pyobjc-framework-AppKit` falla:
```bash
# Opción 1: Instalar sin versión específica
pip install pyobjc-framework-AppKit

# Opción 2: Usar versión mínima
pip install -r requirements-minimal.txt
```

**Windows** - Si `pywin32` falla:
```bash
pip install pywin32
```

**Linux** - Para detectar aplicaciones activas, instala `xdotool`:
```bash
sudo apt-get install xdotool  # Ubuntu/Debian
sudo yum install xdotool      # CentOS/RHEL
```

**Nota**: Las dependencias de detección de aplicación son **opcionales**. El keylogger funcionará sin ellas, pero mostrará "Unknown" en la columna `active_application`.

## 💻 Uso

1. Ejecuta el keylogger:
```bash
python main.py
```

2. Escribe normalmente en tu computadora

3. Los datos se guardan temporalmente en JSON cada 5 segundos

4. Para detener y convertir a CSV, presiona `Ctrl+C`

## 📊 Formato de Datos

### Flujo de Guardado

1. **Durante la ejecución**: Los datos se guardan temporalmente en `keyboard_data/keyboard-data-<random>.json` cada 5 segundos
2. **Al finalizar (Ctrl+C)**: El JSON se convierte automáticamente a CSV y se elimina el archivo temporal

### Formato CSV Final

Los datos finales se guardan en `keyboard_data/keyboard-data-<random>.csv` con formato de columnas:

```csv
timestamp,a,b,c,...,z,0,1,...,9,[ENTER],[CTRL_L]+c,[SHIFT_L]+a,...,avg_hold_time_ms,avg_inter_key_time_ms,...,active_application
1733094304.092,5,3,2,...,0,0,0,...,2,1,0,...,150.5,200.3,...,Visual Studio Code
1733094309.123,0,0,1,...,0,0,0,...,1,0,1,...,145.2,180.5,...,Terminal
```

**Cada 5 segundos** se crea una fila con:
- `timestamp`: Unix timestamp (número)
- Columnas de caracteres: conteo de cada carácter (0 si no se presionó)
- Columnas de combinaciones: conteo de combinaciones detectadas (ej: `[CTRL_L]+c`)
- Estadísticas temporales: tiempos promedio, velocidad, etc.
- `active_application`: aplicación activa

## 🔒 Aplicaciones Bloqueadas

El archivo `blocked_apps.json` contiene la lista de aplicaciones donde **NO se capturan datos**:

- Navegadores web (Chrome, Safari, Firefox, etc.)
- Gestores de contraseñas (1Password, LastPass, etc.)

Para agregar más aplicaciones, edita `blocked_apps.json`:

```json
{
  "blocked_applications": [
    "Google Chrome",
    "Tu App Aquí"  ← Agregar aquí
  ]
}
```

## 📁 Archivos Generados

- `keyboard_data/keyboard-data-<random>.csv` - Datos finales capturados (se crea al finalizar)
- `keyboard_data/keyboard-data-<random>.json` - Datos temporales durante la ejecución (se elimina al finalizar)
- `keylogger_char_count.log` - Logs del sistema
- `blocked_apps.json` - Configuración de aplicaciones bloqueadas
- `key_combinations.json` - Combinaciones de teclas detectadas (se actualiza automáticamente)

## ⚠️ Advertencias

- **Solo para uso educativo y con consentimiento**
- **NO uses en sistemas de otros sin permiso**
- **Revisa los datos capturados antes de compartirlos**

## 🔍 Ejemplo

Si escribes "hola" en 5 segundos, el CSV guardará una fila con:
```
timestamp: 1733094304.092
h: 1
o: 1
l: 1
a: 1
(todas las demás columnas: 0)
active_application: Visual Studio Code
```

**NO se puede reconstruir** que escribiste "hola", solo que usaste esos caracteres.

## 📊 Atributos Estadísticos Capturados

Cada fila incluye estadísticas temporales:
- `avg_hold_time_ms`: Tiempo promedio de presión de teclas
- `avg_inter_key_time_ms`: Tiempo promedio entre pulsaciones
- `std_inter_key_time_ms`: Variabilidad en el ritmo
- `min_inter_key_time_ms` / `max_inter_key_time_ms`: Rango de tiempos
- `keystrokes_per_second`: Velocidad de escritura
- `total_keystrokes`: Total de teclas en el intervalo

## 📝 Notas

- Los caracteres se normalizan a minúsculas (A = a)
- Se cuentan todas las teclas: letras, números, símbolos, teclas especiales y modificadoras
- Las combinaciones de teclas (Ctrl+C, Shift+A, etc.) se detectan automáticamente y se agregan como columnas dinámicas
- Si una aplicación bloqueada está activa, no se captura nada
- El archivo JSON temporal se elimina automáticamente después de convertirse a CSV

---

**Uso responsable**: Este software es solo para fines educativos y de investigación. 🛡️

