from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1.- RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_returns.csv"
PARAMS_PATH = BASE_DIR / "outputs" / "tables" / "01_statistics" / "gbm_parameters.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "02_gbm_simulation"
TABLES_DIR = OUTPUT_DIR / "tables" / "02_gbm_simulation"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2.- CARGA DE DATOS Y PARÁMETROS
# ============================================================

def load_price_data(path: Path) -> pd.DataFrame:
    """
    Carga la serie procesada del precio del oro y sus retornos logarítmicos.
    """

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)

    # El archivo actual contiene: Fecha, Precio, Return.
    # Se normalizan los nombres para trabajar internamente con:
    # date, price, log_return.
    df.columns = [col.strip().lower() for col in df.columns]

    df = df.rename(columns={
        "fecha": "date",
        "precio": "price",
        "valor": "price",
        "return": "log_return",
        "retorno": "log_return",
        "retorno_logaritmico": "log_return",
        "log_returns": "log_return",
    })

    required_columns = {"date", "price", "log_return"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"El archivo debe contener las columnas {required_columns}. "
            f"Columnas encontradas: {set(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")

    df = df.dropna(subset=["date", "price", "log_return"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_gbm_parameters(path: Path) -> tuple[float, float]:
    """
    Carga los parámetros diarios del modelo GBM desde gbm_parameters.csv.

    Para simular día a día se utilizan:
    - mu_daily
    - sigma_daily

    Los parámetros anualizados se reservan para interpretación y reporte.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}. "
            "Ejecuta primero analyze_returns.py para generarlo."
        )

    params_df = pd.read_csv(path)

    params = dict(zip(params_df["parameter"], params_df["value"]))

    mu_daily = float(params["mu_daily"])
    sigma_daily = float(params["sigma_daily"])

    return mu_daily, sigma_daily


# ============================================================
# 3.- SIMULACIÓN DEL MOVIMIENTO BROWNIANO GEOMÉTRICO
# ============================================================

def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    dt: float = 1.0,
    random_seed: int = 42
) -> np.ndarray:
    """
    Simula trayectorias del precio del oro mediante Movimiento Browniano
    Geométrico (GBM).

    La discretización utilizada es:

        S_{t+dt} = S_t * exp[(mu - 0.5*sigma^2)*dt
                             + sigma*sqrt(dt)*Z_t]

    donde Z_t ~ N(0,1).

    Parámetros:
    - s0: precio inicial.
    - mu: deriva diaria estimada desde retornos logarítmicos.
    - sigma: volatilidad diaria estimada desde retornos logarítmicos.
    - n_steps: número de pasos temporales simulados.
    - n_paths: número de trayectorias Monte Carlo.
    - dt: paso temporal. Se usa dt = 1 para un día.
    - random_seed: semilla para reproducibilidad.

    Retorna:
    - Array de tamaño (n_steps + 1, n_paths) con las trayectorias simuladas.
    """

    np.random.seed(random_seed)

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = s0

    # Ruido gaussiano estándar para todos los pasos y trayectorias.
    z = np.random.normal(loc=0.0, scale=1.0, size=(n_steps, n_paths))

    # Factor multiplicativo del GBM para cada paso temporal.
    increments = np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    )

    # Construcción vectorizada de trayectorias.
    # np.cumprod aplica el producto acumulado en el eje temporal.
    paths[1:, :] = s0 * np.cumprod(increments, axis=0)

    return paths


# ============================================================
# 4.- VISUALIZACIONES
# ============================================================

def plot_monte_carlo_paths(paths: np.ndarray, n_to_plot: int = 100) -> None:
    """
    Grafica un subconjunto de trayectorias Monte Carlo simuladas.
    """

    n_steps = paths.shape[0] - 1
    time = np.arange(n_steps + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(time, paths[:, :n_to_plot], linewidth=0.8, alpha=0.5)
    plt.xlabel("Tiempo [días]")
    plt.ylabel("Precio simulado del oro [US$/ozt]")
    plt.title("Trayectorias Monte Carlo del precio del oro mediante GBM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gbm_monte_carlo_paths.png", dpi=300)
    plt.close()


def plot_mean_path(paths: np.ndarray) -> None:
    """
    Grafica la trayectoria media simulada junto con bandas percentiles.
    """

    n_steps = paths.shape[0] - 1
    time = np.arange(n_steps + 1)

    mean_path = np.mean(paths, axis=1)
    p5 = np.percentile(paths, 5, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(time, mean_path, linewidth=2, label="Trayectoria media")
    plt.fill_between(time, p5, p95, alpha=0.3, label="Percentiles 5%-95%")
    plt.xlabel("Tiempo [días]")
    plt.ylabel("Precio simulado del oro [US$/ozt]")
    plt.title("Trayectoria media y banda de incertidumbre GBM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gbm_mean_path_with_band.png", dpi=300)
    plt.close()


def plot_final_price_distribution(paths: np.ndarray) -> None:
    """
    Grafica la distribución de precios finales simulados.
    """

    final_prices = paths[-1, :]

    plt.figure(figsize=(8, 5))
    plt.hist(final_prices, bins=50, density=True, alpha=0.7)
    plt.xlabel("Precio final simulado del oro [US$/ozt]")
    plt.ylabel("Densidad de probabilidad")
    plt.title("Distribución de precios finales simulados mediante GBM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gbm_final_price_distribution.png", dpi=300)
    plt.close()


# ============================================================
# 5.- EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    df = load_price_data(DATA_PATH)
    mu_daily, sigma_daily = load_gbm_parameters(PARAMS_PATH)

    # Se usa como precio inicial el último precio observado en la serie real.
    s0 = df["price"].iloc[-1]

    # Horizonte de simulación: 252 días, aproximadamente un año bursátil.
    n_steps = 252

    # Número de trayectorias Monte Carlo.
    n_paths = 100000

    paths = simulate_gbm(
        s0=s0,
        mu=mu_daily,
        sigma=sigma_daily,
        n_steps=n_steps,
        n_paths=n_paths,
        dt=1.0,
        random_seed=42
    )

    plot_monte_carlo_paths(paths, n_to_plot=100)
    plot_mean_path(paths)
    plot_final_price_distribution(paths)
    # ============================================================
    # Resumen estadístico de la simulación
    # ============================================================

    # No se guardan todas las trayectorias simuladas, porque para un número
    # grande de trayectorias el archivo resultante puede ser muy pesado.
    # En su lugar, se guarda un resumen estadístico de los precios finales.
    final_prices = paths[-1, :]

    summary = pd.DataFrame({
        "statistic": [
            "s0",
            "n_steps",
            "n_paths",
            "mu_daily",
            "sigma_daily",
            "mean_final_price",
            "std_final_price",
            "p5_final_price",
            "p50_final_price",
            "p95_final_price",
            "min_final_price",
            "max_final_price"
        ],
        "value": [
            s0,
            n_steps,
            n_paths,
            mu_daily,
            sigma_daily,
            np.mean(final_prices),
            np.std(final_prices),
            np.percentile(final_prices, 5),
            np.percentile(final_prices, 50),
            np.percentile(final_prices, 95),
            np.min(final_prices),
            np.max(final_prices)
        ]
    })

    summary_output_path = TABLES_DIR / "gbm_simulation_summary.csv"
    summary.to_csv(summary_output_path, index=False)

    print("Simulación GBM completada.")
    print(f"Precio inicial utilizado S0: {s0:.4f} [US$/ozt]")
    print(f"mu_daily utilizado: {mu_daily:.8f}")
    print(f"sigma_daily utilizado: {sigma_daily:.8f}")
    print(f"Número de pasos simulados: {n_steps}")
    print(f"Número de trayectorias: {n_paths}")
    print(f"Media de precios finales simulados: {np.mean(final_prices):.4f} [US$/ozt]")
    print(f"Desviación estándar de precios finales simulados: {np.std(final_prices):.4f} [US$/ozt]")
    print(f"Percentil 5% de precios finales: {np.percentile(final_prices, 5):.4f} [US$/ozt]")
    print(f"Percentil 50% de precios finales: {np.percentile(final_prices, 50):.4f} [US$/ozt]")
    print(f"Percentil 95% de precios finales: {np.percentile(final_prices, 95):.4f} [US$/ozt]")
    print(f"Resumen de simulación guardado en: {summary_output_path}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()