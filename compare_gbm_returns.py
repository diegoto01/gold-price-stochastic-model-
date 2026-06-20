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

    En GBM:

        dS_t = mu S_t dt + sigma S_t dW_t

    Para dt = 1 día, los retornos logarítmicos cumplen:

        r_t = (mu - 0.5 sigma^2) + sigma Z_t

    Luego:

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

        S_{t+1} = S_t exp[(mu - 0.5 sigma^2) + sigma Z_t]

    con dt = 1 día.
    """
    rng = np.random.default_rng(random_seed)

    z = rng.normal(loc=0.0, scale=1.0, size=(n_paths, n_steps))

    increments = np.exp((mu - 0.5 * sigma**2) + sigma * z)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.cumprod(increments, axis=1)

    return paths


def compute_simulated_returns(paths: np.ndarray) -> np.ndarray:
    """
    Calcula retornos logarítmicos simulados a partir de las trayectorias GBM.

    Retorna un arreglo plano con todos los retornos simulados.
    """
    simulated_returns = np.log(paths[:, 1:] / paths[:, :-1])
    simulated_returns = simulated_returns.ravel()
    return simulated_returns


def calculate_distribution_statistics(returns: np.ndarray) -> dict:
    """
    Calcula estadísticos básicos de una distribución de retornos.
    """
    series = pd.Series(returns)

    stats = {
        "mean": series.mean(),
        "variance": series.var(),
        "std": series.std(),
        "skewness": series.skew(),
        "excess_kurtosis": series.kurtosis(),
        "min": series.min(),
        "max": series.max(),
        "p1": series.quantile(0.01),
        "p5": series.quantile(0.05),
        "p50": series.quantile(0.50),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
    }

    return stats


def save_comparison_metrics(
    real_stats: dict,
    simulated_stats: dict,
    output_path: Path,
) -> None:
    """
    Guarda una tabla comparativa de estadísticos reales vs simulados.
    """
    rows = []

    for key in real_stats.keys():
        real_value = real_stats[key]
        simulated_value = simulated_stats[key]
        difference = real_value - simulated_value

        rows.append(
            {
                "statistic": key,
                "real_returns": real_value,
                "gbm_simulated_returns": simulated_value,
                "difference_real_minus_gbm": difference,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)


def plot_returns_distribution(
    real_returns: np.ndarray,
    simulated_returns: np.ndarray,
    output_path: Path,
) -> None:
    """
    Compara la distribución de retornos reales con retornos simulados GBM.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(
        simulated_returns,
        bins=80,
        density=True,
        alpha=0.55,
        label="Retornos simulados GBM",
    )

    plt.hist(
        real_returns,
        bins=40,
        density=True,
        alpha=0.65,
        label="Retornos reales test",
    )

    plt.xlabel("Retorno logarítmico diario")
    plt.ylabel("Densidad")
    plt.title("Distribución de retornos reales vs retornos simulados GBM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_returns_boxplot(
    real_returns: np.ndarray,
    simulated_returns: np.ndarray,
    output_path: Path,
) -> None:
    """
    Genera un boxplot comparativo de retornos reales y simulados.

    Para que el gráfico sea legible, se usa una muestra de retornos simulados
    del mismo tamaño que el conjunto real.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    sample_size = len(real_returns)
    simulated_sample = rng.choice(
        simulated_returns,
        size=sample_size,
        replace=False,
    )

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [real_returns, simulated_sample],
        tick_labels=["Reales test", "GBM simulado"],
        showfliers=True,
    )

    plt.ylabel("Retorno logarítmico diario")
    plt.title("Boxplot de retornos reales vs retornos simulados GBM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cumulative_returns(
    dates: pd.Series,
    real_returns: np.ndarray,
    simulated_mean_returns: np.ndarray,
    output_path: Path,
) -> None:
    """
    Compara el retorno acumulado real con el retorno acumulado medio simulado.
    """
    real_cumulative = np.exp(np.cumsum(real_returns)) - 1
    simulated_cumulative = np.exp(np.cumsum(simulated_mean_returns)) - 1

    # Los retornos tienen un dato menos que los precios.
    dates_returns = dates.iloc[1:]

    plt.figure(figsize=(12, 6))

    plt.plot(
        dates_returns,
        real_cumulative,
        label="Retorno acumulado real",
        linewidth=2,
    )

    plt.plot(
        dates_returns,
        simulated_cumulative,
        label="Retorno acumulado medio GBM",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Fecha")
    plt.ylabel("Retorno acumulado")
    plt.title("Retorno acumulado real vs retorno acumulado medio GBM")
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

    mu_daily, sigma_daily = estimate_gbm_parameters(calibration_returns)

    # Simulación GBM
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

    # Retornos reales
    real_returns = compute_log_returns(test_df["price"]).to_numpy()

    # Retornos simulados
    simulated_returns = compute_simulated_returns(paths)

    # Retorno medio simulado
    simulated_returns_by_day = np.log(paths[:, 1:] / paths[:, :-1])
    simulated_mean_returns = simulated_returns_by_day.mean(axis=0)

    # Estadísticos
    real_stats = calculate_distribution_statistics(real_returns)
    simulated_stats = calculate_distribution_statistics(simulated_returns)

    # Información de calibración
    real_stats_extended = {
        "n_observations": len(real_returns),
        "mu_daily_calibrated": mu_daily,
        "sigma_daily_calibrated": sigma_daily,
        **real_stats,
    }

    simulated_stats_extended = {
        "n_observations": len(simulated_returns),
        "mu_daily_calibrated": mu_daily,
        "sigma_daily_calibrated": sigma_daily,
        **simulated_stats,
    }

    # Rutas de salida
    metrics_path = TABLES_DIR / "gbm_returns_comparison_metrics.csv"
    distribution_plot_path = FIGURES_DIR / "gbm_vs_real_returns_distribution.png"
    boxplot_path = FIGURES_DIR / "gbm_vs_real_returns_boxplot.png"
    cumulative_plot_path = FIGURES_DIR / "gbm_vs_real_cumulative_returns.png"

    # Guardar métricas
    save_comparison_metrics(
        real_stats=real_stats_extended,
        simulated_stats=simulated_stats_extended,
        output_path=metrics_path,
    )

    # Guardar gráficos
    plot_returns_distribution(
        real_returns=real_returns,
        simulated_returns=simulated_returns,
        output_path=distribution_plot_path,
    )

    plot_returns_boxplot(
        real_returns=real_returns,
        simulated_returns=simulated_returns,
        output_path=boxplot_path,
    )

    plot_cumulative_returns(
        dates=test_df["date"],
        real_returns=real_returns,
        simulated_mean_returns=simulated_mean_returns,
        output_path=cumulative_plot_path,
    )

    # Resumen final
    print("Comparación de retornos reales vs retornos GBM completada.")
    print(f"Datos de calibración: {len(calibration_df)} precios")
    print(f"Datos de test: {len(test_df)} precios")
    print(f"Retornos reales test: {len(real_returns)}")
    print(f"Retornos simulados GBM: {len(simulated_returns)}")
    print()
    print(f"mu_daily calibrado: {mu_daily:.8f}")
    print(f"sigma_daily calibrado: {sigma_daily:.8f}")
    print(f"mu anualizado: {mu_daily * 252:.4%}")
    print(f"sigma anualizado: {sigma_daily * np.sqrt(252):.4%}")
    print()
    print("Estadísticos principales:")
    print(f"Media retornos reales: {real_stats['mean']:.8f}")
    print(f"Media retornos GBM:    {simulated_stats['mean']:.8f}")
    print(f"Std retornos reales:   {real_stats['std']:.8f}")
    print(f"Std retornos GBM:      {simulated_stats['std']:.8f}")
    print(f"Asimetría real:        {real_stats['skewness']:.8f}")
    print(f"Asimetría GBM:         {simulated_stats['skewness']:.8f}")
    print(f"Curtosis real:         {real_stats['excess_kurtosis']:.8f}")
    print(f"Curtosis GBM:          {simulated_stats['excess_kurtosis']:.8f}")
    print(f"Mínimo real:           {real_stats['min']:.8f}")
    print(f"Mínimo GBM:            {simulated_stats['min']:.8f}")
    print(f"Máximo real:           {real_stats['max']:.8f}")
    print(f"Máximo GBM:            {simulated_stats['max']:.8f}")
    print()
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Histograma guardado en: {distribution_plot_path}")
    print(f"Boxplot guardado en: {boxplot_path}")
    print(f"Retorno acumulado guardado en: {cumulative_plot_path}")


if __name__ == "__main__":
    main()
