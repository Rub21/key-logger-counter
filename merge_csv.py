import pandas as pd
import os
import glob
from pathlib import Path

# Directorio de entrada y salida
input_dir = 'keyboard_data'
output_dir = 'results'

# Crear carpeta Results si no existe
os.makedirs(output_dir, exist_ok=True)

# Buscar todos los archivos CSV en keyboard_data
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

if not csv_files:
    print(f"No se encontraron archivos CSV en {input_dir}")
    exit(1)

print(f"Encontrados {len(csv_files)} archivos CSV:")
for file in csv_files:
    print(f"  - {file}")

# Leer y concatenar todos los archivos CSV
dataframes = []
for csv_file in sorted(csv_files):
    print(f"Leyendo {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
        print(f"  ✓ {len(df)} filas leídas")
    except Exception as e:
        print(f"  ✗ Error leyendo {csv_file}: {e}")

if not dataframes:
    print("No se pudieron leer archivos CSV válidos")
    exit(1)

# Concatenar todos los dataframes
print("\nConcatenando archivos...")
merged_data = pd.concat(dataframes, ignore_index=True)

print(f"\nTotal de filas en el archivo mergeado: {len(merged_data)}")

# Generar nombre del archivo de salida con timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f'merged_keyboard_data_{timestamp}.csv')

# Guardar el archivo mergeado
print(f"\nGuardando resultado en {output_file}...")
merged_data.to_csv(output_file, index=False)

print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"  Total de filas: {len(merged_data)}")
print(f"  Total de columnas: {len(merged_data.columns)}")