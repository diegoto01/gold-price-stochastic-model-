import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
files = sorted(RAW_DIR.glob("oro_*.xls"))

dfs = [] 

for file in files:
    print(f"Leyendo archivo: {file.name}")

    # Los archivos del Banco Central tienen tres filas iniciales de encabezado.
    df = pd.read_excel(file, skiprows=3)

    df = df.iloc[:, :2].copy()

    df.columns = ["Fecha", "Precio"]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Convierte precios con coma decimal al formato numérico de Python.
    df["Precio"] = (
        df["Precio"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")

    # Se descartan filas incompletas.
    df = df.dropna(subset=["Fecha", "Precio"])

    dfs.append(df)

df_gold = pd.concat(dfs, ignore_index=True)

df_gold = df_gold.sort_values("Fecha").reset_index(drop=True)

df_gold = df_gold.drop_duplicates(subset=["Fecha"])

output_path = PROCESSED_DIR / "gold_prices_clean.csv"
df_gold.to_csv(output_path, index=False)
