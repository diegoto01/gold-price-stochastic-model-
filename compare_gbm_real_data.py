from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "03_gbm_validation"
TABLES_DIR = OUTPUT_DIR / "tables" / "03_gbm_validation"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 252
N_PATHS = 100_000
RANDOM_SEED = 42



def find_date_column(df: pd.DataFrame) -> str:
    """
    Intenta encontrar automáticamente la columna de fecha.
    """
    possible_names = ["date", "fecha", "time", "datetime"]

    for col in df.columns:
        if col.lower() in possible_names:
            return col

    for col in df.columns:
        if "date" in col.lower() or "fecha" in col.lower():
            return col

    raise ValueError(
        "No se encontró una columna de fecha. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def find_price_column(df: pd.DataFrame) -> str:
    """
    Intenta encontrar automáticamente la columna de precio.
    """
    possible_names = [
        "price",
        "precio",
        "close",
        "adj close",
        "adj_close",
        "gold_price",
        "value",
        "usd",
    ]

    for name in possible_names:
        for col in df.columns:
            if col.lower() == name:
                return col

    for col in df.columns:
        lower_col = col.lower()
        if "price" in lower_col or "precio" in lower_col or "close" in lower_col:
            return col

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    raise ValueError(
        "No se encontró una columna clara de precio. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def load_price_data(path: Path) -> pd.DataFrame:
    """
    Carga, limpia y ordena la serie de precios del oro.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)

    date_col = find_date_column(df)
    price_col = find_price_column(df)

    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "price"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")


    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calcula retornos logarítmicos.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns


def estimate_gbm_parameters(returns: pd.Series) -> tuple[float, float]:
    """
    Estima los parámetros diarios del GBM.

    En el modelo GBM:

        dS_t = mu S_t dt + sigma S_t dW_t

    los retornos logarítmicos cumplen aproximadamente:

        r_t = (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_t

    Para dt = 1 día:

        mu = media_retornos + 0.5 * varianza_retornos
        sigma = desviación_estándar_retornos
    """
    mean_return = returns.mean()
    variance_return = returns.var()
    sigma_daily = returns.std()

    mu_daily = mean_return + 0.5 * variance_return

    return mu_daily, sigma_daily


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Simula trayectorias GBM usando la solución discreta exacta:

        S_{t+1} = S_t * exp[(mu - 0.5 sigma^2) + sigma Z_t]

    con dt = 1 día.
    """
    rng = np.random.default_rng(random_seed)

    z = rng.normal(loc=0.0, scale=1.0, size=(n_paths, n_steps))

    increments = np.exp((mu - 0.5 * sigma**2) + sigma * z)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.cumprod(increments, axis=1)

    return paths


def calculate_metrics(
    real_prices: np.ndarray,
    mean_simulated: np.ndarray,
    p5: np.ndarray,
    p95: np.ndarray,
) -> dict:
    """
    Calcula métricas de comparación entre trayectoria real y media GBM.
    """
    errors = mean_simulated - real_prices

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / real_prices)) * 100

    final_real_price = real_prices[-1]
    final_mean_price = mean_simulated[-1]

    final_absolute_error = abs(final_mean_price - final_real_price)
    final_percentage_error = abs(final_mean_price - final_real_price) / final_real_price * 100

    inside_band = (real_prices >= p5) & (real_prices <= p95)
    coverage_5_95 = np.mean(inside_band) * 100

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "final_real_price": final_real_price,
        "final_mean_gbm_price": final_mean_price,
        "final_absolute_error": final_absolute_error,
        "final_percentage_error": final_percentage_error,
        "coverage_percentile_5_95": coverage_5_95,
    }

    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """
    Guarda métricas en formato CSV.
    """
    metrics_df = pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
        }
    )

    metrics_df.to_csv(output_path, index=False)


def plot_gbm_vs_real(
    dates: pd.Series,
    real_prices: np.ndarray,
    mean_simulated: np.ndarray,
    p5: np.ndarray,
    p95: np.ndarray,
    output_path: Path,
) -> None:
    """
    Grafica precio real vs media GBM y banda percentil 5%-95%.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(dates, real_prices, label="Precio real", linewidth=2)
    plt.plot(dates, mean_simulated, label="Media GBM simulada", linestyle="--", linewidth=2)

    plt.fill_between(
        dates,
        p5,
        p95,
        alpha=0.25,
        label="Banda percentil 5%-95%",
    )

    plt.xlabel("Fecha")
    plt.ylabel("Precio del oro [US$/ozt]")
    plt.title("Comparación entre precio real del oro y simulación GBM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_final_price_distribution(
    final_prices: np.ndarray,
    real_final_price: float,
    output_path: Path,
) -> None:
    """
    Grafica la distribución de precios finales simulados y marca el precio real final.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(final_prices, bins=60, density=True, alpha=0.7)
    plt.axvline(real_final_price, linestyle="--", linewidth=2, label="Precio real final")

    plt.xlabel("Precio final simulado [US$/ozt]")
    plt.ylabel("Densidad")
    plt.title("Distribución de precios finales simulados mediante GBM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()



def main() -> None:
    # Datos
    df = load_price_data(DATA_PATH)

    if len(df) <= TEST_DAYS:
        raise ValueError(
            f"La serie tiene {len(df)} datos, pero se requieren más de {TEST_DAYS}."
        )

    # Calibración y validación
    calibration_df = df.iloc[:-TEST_DAYS].copy()
    test_df = df.iloc[-TEST_DAYS:].copy()

    # Retornos de calibración
    calibration_returns = compute_log_returns(calibration_df["price"])

    # Parámetros del GBM
    mu_daily, sigma_daily = estimate_gbm_parameters(calibration_returns)

    # Simulación
    s0 = test_df["price"].iloc[0]
    n_steps = len(test_df) - 1

    paths = simulate_gbm_paths(
        s0=s0,
        mu=mu_daily,
        sigma=sigma_daily,
        n_steps=n_steps,
        n_paths=N_PATHS,
        random_seed=RANDOM_SEED,
    )

    # Estadísticos simulados
    mean_simulated = paths.mean(axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    real_prices = test_df["price"].to_numpy()
    dates = test_df["date"]

    # Métricas
    metrics = calculate_metrics(
        real_prices=real_prices,
        mean_simulated=mean_simulated,
        p5=p5,
        p95=p95,
    )

    # Información de calibración
    extra_info = {
        "n_calibration_prices": len(calibration_df),
        "n_test_prices": len(test_df),
        "n_paths": N_PATHS,
        "n_steps": n_steps,
        "s0": s0,
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annualized": mu_daily * 252,
        "sigma_annualized": sigma_daily * np.sqrt(252),
    }

    all_metrics = {**extra_info, **metrics}

    # Guardar resultados
    metrics_path = TABLES_DIR / "gbm_real_comparison_metrics.csv"
    plot_path = FIGURES_DIR / "gbm_vs_real_price.png"
    distribution_path = FIGURES_DIR / "gbm_vs_real_final_distribution.png"

    save_metrics(all_metrics, metrics_path)

    plot_gbm_vs_real(
        dates=dates,
        real_prices=real_prices,
        mean_simulated=mean_simulated,
        p5=p5,
        p95=p95,
        output_path=plot_path,
    )

    plot_final_price_distribution(
        final_prices=paths[:, -1],
        real_final_price=real_prices[-1],
        output_path=distribution_path,
    )

    # Resumen final
    print("Comparación GBM vs datos reales completada.")
    print(f"Datos de calibración: {len(calibration_df)} observaciones")
    print(f"Datos de test: {len(test_df)} observaciones")
    print(f"Precio inicial del tramo test S0: {s0:.4f} [US$/ozt]")
    print(f"mu_daily calibrado: {mu_daily:.8f}")
    print(f"sigma_daily calibrado: {sigma_daily:.8f}")
    print(f"mu anualizado: {mu_daily * 252:.4%}")
    print(f"sigma anualizado: {sigma_daily * np.sqrt(252):.4%}")
    print()
    print(f"MAE: {metrics['mae']:.4f} [US$/ozt]")
    print(f"RMSE: {metrics['rmse']:.4f} [US$/ozt]")
    print(f"MAPE: {metrics['mape_percent']:.4f} %")
    print(f"Error final absoluto: {metrics['final_absolute_error']:.4f} [US$/ozt]")
    print(f"Error final porcentual: {metrics['final_percentage_error']:.4f} %")
    print(f"Días reales dentro de banda 5%-95%: {metrics['coverage_percentile_5_95']:.2f} %")
    print()
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Gráfico comparación guardado en: {plot_path}")
    print(f"Distribución final guardada en: {distribution_path}")


if __name__ == "__main__":
    main()
