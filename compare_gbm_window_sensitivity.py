# ============================================================
# Sensibilidad temporal del GBM
# Proyecto FIS205 - Modelamiento estocástico del precio del oro
# ============================================================
#
# Este script estudia cómo cambia el desempeño del modelo GBM
# cuando se calibra usando distintas ventanas históricas.
#
# La motivación física/estadística es que la serie del precio del oro
# puede no ser estacionaria: distintos periodos históricos pueden tener
# diferentes tendencias y volatilidades.
#
# Se comparan tres ventanas de calibración:
#
#   1. Desde 2009 hasta el inicio del test
#   2. Desde 2018 hasta el inicio del test
#   3. Desde 2024 hasta el inicio del test
#
# El tramo de test se mantiene fijo:
#
#   Test = últimos 252 días disponibles
#
# Esto permite comparar si usar más o menos memoria histórica cambia
# la capacidad del GBM para representar el régimen reciente del precio
# del oro.
#
# Importante:
# Esto no corresponde a una estrategia de inversión ni a un modelo de
# predicción financiera, sino a un análisis de sensibilidad de parámetros
# dentro de un modelo estocástico.


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuración general
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_clean.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures" / "03_gbm_validation"
TABLES_DIR = BASE_DIR / "outputs" / "tables" / "03_gbm_validation"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# Ventanas históricas a comparar
START_YEARS = [2009, 2018, 2024]

# Horizonte de test: aproximadamente un año bursátil
TEST_SIZE = 252

# Número de trayectorias Monte Carlo
N_PATHS = 100_000

# Semilla para reproducibilidad
RANDOM_SEED = 42


# ============================================================
# Funciones auxiliares
# ============================================================

def load_price_data(path):
    """
    Carga la serie limpia de precios del oro.

    El archivo esperado es data/processed/gold_prices_clean.csv.
    Se aceptan columnas tipo Fecha/Date y Precio/Price.
    """

    data = pd.read_csv(path)

    date_candidates = ["Fecha", "fecha", "Date", "date"]
    price_candidates = [
        "Precio", "precio",
        "Price", "price",
        "Close", "close",
        "Gold_Price", "gold_price"
    ]

    date_col = None
    price_col = None

    for col in date_candidates:
        if col in data.columns:
            date_col = col
            break

    for col in price_candidates:
        if col in data.columns:
            price_col = col
            break

    if date_col is None:
        raise ValueError(
            "No se encontró columna de fecha. "
            "Revisa que exista una columna llamada Fecha o Date."
        )

    if price_col is None:
        raise ValueError(
            "No se encontró columna de precio. "
            "Revisa que exista una columna llamada Precio o Price."
        )

    data = data[[date_col, price_col]].copy()
    data.columns = ["date", "price"]

    data["date"] = pd.to_datetime(data["date"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna()
    data = data.sort_values("date").reset_index(drop=True)

    return data


def estimate_gbm_parameters(prices):
    """
    Estima los parámetros diarios del GBM usando retornos logarítmicos.

    Para el GBM:

        dS_t = mu S_t dt + sigma S_t dW_t

    los retornos logarítmicos discretos tienen media aproximada:

        E[r] = mu - 0.5 sigma^2

    Por eso:

        mu = mean(r) + 0.5 sigma^2
    """

    log_returns = np.log(prices[1:] / prices[:-1])

    sigma_daily = np.std(log_returns, ddof=1)
    mu_daily = np.mean(log_returns) + 0.5 * sigma_daily**2

    return mu_daily, sigma_daily


def simulate_gbm_paths(S0, mu, sigma, n_steps, n_paths, dt=1.0, seed=42):
    """
    Simula trayectorias de GBM usando la solución discreta exacta:

        S_{t+1} = S_t exp[(mu - 0.5 sigma^2)dt + sigma sqrt(dt) Z_t]
    """

    rng = np.random.default_rng(seed)

    Z = rng.normal(0.0, 1.0, size=(n_steps, n_paths))

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = S0
    paths[1:, :] = S0 * np.exp(np.cumsum(increments, axis=0))

    return paths


def compute_metrics(real_values, predicted_values):
    """
    Calcula métricas de error entre la trayectoria real y una trayectoria media.
    """

    errors = predicted_values - real_values

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / real_values)) * 100

    final_abs_error = np.abs(predicted_values[-1] - real_values[-1])
    final_pct_error = final_abs_error / real_values[-1] * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "final_absolute_error": final_abs_error,
        "final_percentage_error": final_pct_error,
    }


# ============================================================
# Programa principal
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Cargar datos
    # --------------------------------------------------------

    data = load_price_data(DATA_PATH)

    if len(data) <= TEST_SIZE:
        raise ValueError("La serie es demasiado corta para usar 252 días de test.")

    # --------------------------------------------------------
    # 2. Separar test fijo
    # --------------------------------------------------------

    train_full = data.iloc[:-TEST_SIZE].copy()
    test_data = data.iloc[-TEST_SIZE:].copy()

    test_dates = test_data["date"].to_numpy()
    test_prices = test_data["price"].to_numpy()

    S0 = test_prices[0]
    n_steps = TEST_SIZE - 1

    # --------------------------------------------------------
    # 3. Simular GBM para cada ventana histórica
    # --------------------------------------------------------

    results = []
    mean_paths = {}

    for start_year in START_YEARS:
        train_window = train_full[train_full["date"].dt.year >= start_year].copy()

        if len(train_window) < 30:
            print(f"Ventana {start_year}: omitida por tener pocos datos.")
            continue

        train_prices = train_window["price"].to_numpy()

        mu_daily, sigma_daily = estimate_gbm_parameters(train_prices)

        paths = simulate_gbm_paths(
            S0=S0,
            mu=mu_daily,
            sigma=sigma_daily,
            n_steps=n_steps,
            n_paths=N_PATHS,
            dt=1.0,
            seed=RANDOM_SEED,
        )

        mean_path = paths.mean(axis=1)
        mean_paths[start_year] = mean_path

        metrics = compute_metrics(test_prices, mean_path)

        results.append(
            {
                "start_year": start_year,
                "n_train_prices": len(train_prices),
                "train_start_date": train_window["date"].iloc[0].date(),
                "train_end_date": train_window["date"].iloc[-1].date(),
                "mu_daily": mu_daily,
                "sigma_daily": sigma_daily,
                "mu_annualized": mu_daily * 252,
                "sigma_annualized": sigma_daily * np.sqrt(252),
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # 4. Guardar tabla de métricas
    # --------------------------------------------------------

    metrics_path = TABLES_DIR / "gbm_window_sensitivity_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------
    # 5. Graficar trayectorias medias
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        test_dates,
        test_prices,
        label="Precio real",
        linewidth=2.4,
    )

    for start_year, mean_path in mean_paths.items():
        ax.plot(
            test_dates,
            mean_path,
            linewidth=2.0,
            linestyle="--",
            label=f"GBM desde {start_year}",
        )

    ax.set_title("Sensibilidad temporal del GBM")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio del oro [US$/ozt]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "gbm_window_sensitivity_price.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 6. Mostrar resumen
    # --------------------------------------------------------

    print("Sensibilidad temporal del GBM completada.")
    print()
    print(f"Test fijo: últimos {TEST_SIZE} precios")
    print(f"Fecha inicial test: {test_data['date'].iloc[0].date()}")
    print(f"Fecha final test: {test_data['date'].iloc[-1].date()}")
    print(f"Precio inicial test S0: {S0:.4f} [US$/ozt]")
    print()
    print("Resultados por ventana:")
    print(
        results_df[
            [
                "start_year",
                "n_train_prices",
                "mu_annualized",
                "sigma_annualized",
                "MAE",
                "RMSE",
                "MAPE_percent",
                "final_percentage_error",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tabla guardada en: {metrics_path}")
    print(f"Gráfico guardado en: {figure_path}")


if __name__ == "__main__":
    main()
