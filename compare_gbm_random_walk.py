# ============================================================
# Comparación entre GBM y Random Walk
# Proyecto FIS205 - Modelamiento estocástico del precio del oro
# ============================================================
#
# Este script compara el modelo de Movimiento Browniano Geométrico
# con un modelo base ingenuo tipo Random Walk.
#
# La idea central es evaluar si el GBM entrega una mejora cuantitativa
# frente a una referencia mínima. En este caso, el Random Walk usado
# como baseline mantiene constante el precio inicial del tramo de test:
#
#     S_hat(t) = S0
#
# Este baseline no intenta predecir tendencias ni fluctuaciones; solo
# sirve como punto de comparación simple. Si el GBM no mejora claramente
# frente a este modelo, entonces sus limitaciones deben discutirse en
# el análisis del proyecto.
#
# Importante:
# Este análisis no se interpreta como una estrategia de inversión ni
# como una predicción financiera, sino como una comparación entre modelos
# estocásticos desde una perspectiva de física computacional.


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


# ============================================================
# Funciones auxiliares
# ============================================================

def load_price_data(path):
    """
    Carga la serie limpia de precios del oro.

    Se espera un archivo CSV con una columna de fecha y una columna de
    precio. El script intenta detectar automáticamente esas columnas
    para mantener compatibilidad con el preprocesamiento anterior.
    """

    data = pd.read_csv(path)

    date_candidates = ["Date", "date", "Fecha", "fecha"]
    price_candidates = ["Precio", "precio", "Price", "price", "Close", "close", "Gold_Price", "gold_price"]

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
            "No se encontró una columna de fecha. "
            "Revisa que el archivo tenga una columna tipo Date o Fecha."
        )

    if price_col is None:
        raise ValueError(
            "No se encontró una columna de precio. "
            "Revisa que el archivo tenga una columna tipo Price o Close."
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
    Estima los parámetros diarios del GBM a partir de precios históricos.

    Para un GBM:

        dS_t = mu S_t dt + sigma S_t dW_t

    los retornos logarítmicos discretos cumplen aproximadamente:

        r_t = ln(S_t / S_{t-1})
            ~ Normal((mu - 0.5 sigma^2) dt, sigma^2 dt)

    Por eso, si se estima la media de retornos logarítmicos como r_mean,
    entonces la deriva del proceso de precios se aproxima mediante:

        mu = r_mean + 0.5 sigma^2
    """

    log_returns = np.log(prices[1:] / prices[:-1])

    sigma_daily = np.std(log_returns, ddof=1)
    mu_daily = np.mean(log_returns) + 0.5 * sigma_daily**2

    return mu_daily, sigma_daily


def simulate_gbm_paths(S0, mu, sigma, n_steps, n_paths, dt=1.0, seed=42):
    """
    Simula trayectorias de Movimiento Browniano Geométrico.

    Se usa la solución discreta exacta del GBM:

        S_{t+1} = S_t exp[(mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_t]

    donde Z_t es una variable normal estándar.
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
    Calcula métricas simples de distancia entre trayectoria real
    y trayectoria modelada.
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


def main():
    # --------------------------------------------------------
    # 1. Cargar datos y separar entrenamiento/test
    # --------------------------------------------------------

    data = load_price_data(DATA_PATH)

    dates = data["date"].to_numpy()
    prices = data["price"].to_numpy()

    test_size = 252

    if len(prices) <= test_size:
        raise ValueError("La serie es demasiado corta para separar 252 días de test.")

    train_prices = prices[:-test_size]
    test_prices = prices[-test_size:]
    test_dates = dates[-test_size:]

    S0 = test_prices[0]

    # --------------------------------------------------------
    # 2. Calibrar GBM usando solo el tramo de entrenamiento
    # --------------------------------------------------------

    mu_daily, sigma_daily = estimate_gbm_parameters(train_prices)

    mu_annualized = mu_daily * 252
    sigma_annualized = sigma_daily * np.sqrt(252)

    # --------------------------------------------------------
    # 3. Simular GBM en el horizonte de test
    # --------------------------------------------------------

    n_steps = test_size - 1
    n_paths = 100_000

    gbm_paths = simulate_gbm_paths(
        S0=S0,
        mu=mu_daily,
        sigma=sigma_daily,
        n_steps=n_steps,
        n_paths=n_paths,
        dt=1.0,
        seed=42,
    )

    gbm_mean_path = gbm_paths.mean(axis=1)

    # --------------------------------------------------------
    # 4. Construir baseline Random Walk constante
    # --------------------------------------------------------

    random_walk_constant = np.full_like(test_prices, fill_value=S0, dtype=float)

    # --------------------------------------------------------
    # 5. Calcular métricas
    # --------------------------------------------------------

    metrics_gbm = compute_metrics(test_prices, gbm_mean_path)
    metrics_rw = compute_metrics(test_prices, random_walk_constant)

    metrics_df = pd.DataFrame(
        [
            {"model": "GBM_mean", **metrics_gbm},
            {"model": "Random_Walk_constant", **metrics_rw},
        ]
    )

    metrics_path = TABLES_DIR / "gbm_random_walk_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------
    # 6. Graficar comparación
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(test_dates, test_prices, label="Precio real", linewidth=2.2)
    ax.plot(test_dates, gbm_mean_path, label="Media GBM", linewidth=2.0, linestyle="--")
    ax.plot(
        test_dates,
        random_walk_constant,
        label="Random Walk constante",
        linewidth=2.0,
        linestyle=":",
    )

    ax.set_title("Comparación: precio real vs GBM vs Random Walk")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio del oro [US$/ozt]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "gbm_vs_random_walk_price.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 7. Mostrar resumen en terminal
    # --------------------------------------------------------

    print("Comparación GBM vs Random Walk completada.")
    print(f"Datos de entrenamiento: {len(train_prices)} precios")
    print(f"Datos de test: {len(test_prices)} precios")
    print(f"Precio inicial del test S0: {S0:.4f} [US$/ozt]")
    print()
    print(f"mu_daily calibrado: {mu_daily:.8f}")
    print(f"sigma_daily calibrado: {sigma_daily:.8f}")
    print(f"mu anualizado: {mu_annualized:.4%}")
    print(f"sigma anualizado: {sigma_annualized:.4%}")
    print()
    print("Métricas comparativas:")
    print(metrics_df.to_string(index=False))
    print()
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Gráfico guardado en: {figure_path}")


if __name__ == "__main__":
    main()
