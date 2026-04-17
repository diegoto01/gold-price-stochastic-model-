import pandas as pd
from pathlib import Path
import numpy as np

## 0.- CARGAMOS RUTAS

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

## 1.- CARGAMOS LOS DATOS LIMPIOS

file_path = PROCESSED_DIR / "gold_prices_clean.csv"
df = pd.read_csv(file_path)

df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.sort_values("Fecha").reset_index(drop=True)

## 2.- CALCULAMOS LOS RETORNOS

df["Return"] = np.log(df["Precio"] / df["Precio"].shift(1))

df = df.dropna(subset=["Return"])

## 3.- GUARDAMOS LOS RETORNOS

output_path = PROCESSED_DIR / "gold_returns.csv"
df.to_csv(output_path, index=False)

print(df.head())
print(df.info())