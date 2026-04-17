import pandas as pd
from pathlib import Path

## 1.- CONFIGURACION DE LAS RUTAS UTILIZADAS
#OBS: Se utiliza la funcion "Path" de la libreria "pathlib" con el fin de tratar las rutas como objetos y tambien para asegurar la portabilidad (macOS - Windows - Linux)

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
## 2.- LIMPIEZA

files = sorted(RAW_DIR.glob("oro_*.xls")) # Esto nos da una lista de archivos que coinciden con el patrón "oro_*.xls" en la carpeta RAW_DIR, ordenados alfabéticamente.

dfs = [] 

for file in files: # Iteramos sobre cada archivo encontrado en la carpeta raw, y para cada uno realizamos el proceso de lectura y limpieza.
    print(f"Leyendo archivo: {file.name}")

    df = pd.read_excel(file, skiprows=3) # Esto asume que las primeras 3 filas del archivo son encabezados o información no relevante, y que los datos comienzan a partir de la fila 4.
    # El formato de los archivos excel que provee el Banco Central ha demostrado que siempre tienen 3 filas de encabezado, por lo que esta es una suposición razonable para estandarizar la carga de datos.

    df = df.iloc[:, :2].copy() # Nos quedamos solo con las dos primeras columnas.

    df.columns = ["Fecha", "Precio"] # Nombramos las columnas.

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce") # Convertimos la columna "Fecha" a formato datetime.

    # Limpieza de precio: convierte algo como '1820,32' en 1820.32
    df["Precio"] = (
        df["Precio"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")

    # Eliminamos filas que no tienen valor de fecha o de precio
    df = df.dropna(subset=["Fecha", "Precio"])

    dfs.append(df) # Agregamos el DataFrame limpio a la lista de DataFrames.

## 3.- UNION

df_gold = pd.concat(dfs, ignore_index=True) # Concatenamos los DataFrames.

df_gold = df_gold.sort_values("Fecha").reset_index(drop=True) # Ordenamos por fecha el DataFrame resultante y reiniciamos el índice.

df_gold = df_gold.drop_duplicates(subset=["Fecha"]) # Eliminamos eventuales duplicados.

## 4.- GUARDADO

output_path = PROCESSED_DIR / "gold_prices_clean.csv"
df_gold.to_csv(output_path, index=False)