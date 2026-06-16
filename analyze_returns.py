from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1.- ABRIMOS LAS RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_returns.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "01_exploratory"
TABLES_DIR = OUTPUT_DIR / "tables" / "01_statistics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2.- CARGA Y VALIDACIÓN DE DATOS
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """
    Cargamos la serie del precio del oro con retornos logarítmicos.
    """

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)

    # Normalización básica de nombres de columnas.
    # Ejemplo: Fecha -> fecha, Precio -> precio, Return -> return.
    df.columns = [col.strip().lower() for col in df.columns]

    # Renombramos columnas al formato interno estándar -> inglés y sin tildes.
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


# ============================================================
# 3.- ESTADÍSTICA DESCRIPTIVA
# ============================================================

def compute_statistics(returns: pd.Series) -> pd.DataFrame:
    """
    Calcula estadísticos básicos de los retornos logarítmicos.

    Se utilizan métodos integrados de pandas para calcular los estadísticos
    descriptivos directamente sobre la serie de retornos. Esto permite mantener
    el análisis compacto, reproducible y adecuado para una serie temporal real.

    Notas:
    - returns.var() calcula la varianza muestral por defecto.
    - returns.std() calcula la desviación estándar muestral por defecto.
    - returns.kurtosis() entrega el exceso de curtosis; para una distribución
      normal ideal, este valor es aproximadamente 0.
    """

    stats = {
        "n": returns.count(),            # número de retornos válidos
        "mean": returns.mean(),          # media empírica de los retornos
        "variance": returns.var(),       # varianza muestral
        "std": returns.std(),            # desviación estándar muestral
        "skewness": returns.skew(),      # asimetría de la distribución
        "kurtosis": returns.kurtosis(),  # exceso de curtosis
        "min": returns.min(),            # retorno mínimo observado
        "max": returns.max(),            # retorno máximo observado
    }

    return pd.DataFrame.from_dict(stats, orient="index", columns=["value"])


# ============================================================
# 4.- VISUALIZACIONES
# ============================================================

def plot_price_series(df: pd.DataFrame) -> None:
    """
    Grafica la serie temporal del precio del oro.

    La unidad utilizada para el precio es US$/ozt, es decir,
    dólares estadounidenses por onza troy.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["price"])
    plt.xlabel("Fecha")
    plt.ylabel("Precio del oro [US$/ozt]")
    plt.title("Serie temporal del precio del oro en US$/ozt")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_series.png", dpi=300)
    plt.close()


def plot_returns_series(df: pd.DataFrame) -> None:
    """
    Grafica la serie temporal de retornos logarítmicos.

    Los retornos logarítmicos son adimensionales, ya que se calculan
    como el logaritmo del cociente entre dos precios con la misma unidad.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["log_return"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fecha")
    plt.ylabel("Retorno logarítmico [adimensional]")
    plt.title("Serie temporal de retornos logarítmicos diarios")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "log_returns_series.png", dpi=300)
    plt.close()


def plot_returns_histogram(returns: pd.Series) -> None:
    """
    Grafica el histograma de los retornos logarítmicos.

    Se usa density=True para representar una densidad de probabilidad,
    no una frecuencia absoluta.
    """

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=50, density=True, alpha=0.7)
    plt.xlabel("Retorno logarítmico [adimensional]")
    plt.ylabel("Densidad de probabilidad")
    plt.title("Histograma de retornos logarítmicos diarios")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "log_returns_histogram.png", dpi=300)
    plt.close()


def plot_returns_normal_fit(returns: pd.Series) -> None:
    """
    Compara el histograma de retornos con una distribución normal ajustada.

    La normal ajustada se construye usando la media y la desviación estándar
    empíricas de los retornos. Esta comparación permite evaluar visualmente
    qué tan compatible es la distribución observada con el supuesto gaussiano
    utilizado en modelos difusivos simples como el GBM.
    """

    mu = returns.mean()
    sigma = returns.std()

    # np.linspace se utiliza para construir una malla de valores donde evaluar
    # la densidad normal ajustada.
    x = np.linspace(returns.min(), returns.max(), 500)

    # Densidad de probabilidad normal:
    # f(x) = 1/(sigma*sqrt(2*pi)) * exp(-0.5*((x-mu)/sigma)^2)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=50, density=True, alpha=0.7, label="Retornos reales")
    plt.plot(x, normal_pdf, linewidth=2, label="Normal ajustada")
    plt.xlabel("Retorno logarítmico [adimensional]")
    plt.ylabel("Densidad de probabilidad")
    plt.title("Retornos logarítmicos diarios vs distribución normal ajustada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "log_returns_normal_fit.png", dpi=300)
    plt.close()


def plot_autocorrelation(returns: pd.Series, max_lag: int = 30) -> None:
    """
    Calcula y grafica la autocorrelación de los retornos.

    Se utiliza el método integrado returns.autocorr(lag=lag) de pandas.
    Cada retardo representa una diferencia temporal medida en días de la serie.
    La autocorrelación es adimensional.
    """

    autocorr_values = [returns.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    lags = np.arange(1, max_lag + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(lags, autocorr_values)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Retardo [días]")
    plt.ylabel("Autocorrelación [adimensional]")
    plt.title("Autocorrelación de retornos logarítmicos diarios")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "returns_autocorrelation.png", dpi=300)
    plt.close()


# ============================================================
# 5.- EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    df = load_data(DATA_PATH)
    returns = df["log_return"]

    # ============================================================
    # Estadísticos descriptivos de los retornos
    # ============================================================

    stats_df = compute_statistics(returns)
    stats_path = TABLES_DIR / "returns_statistics.csv"
    stats_df.to_csv(stats_path)

    # ============================================================
    # Parámetros preliminares para el modelo GBM
    # ============================================================

    # Se utiliza una anualización estándar de 252 días hábiles.
    # Esta anualización se usa solo como escala comparativa, no como
    # predicción financiera.
    trading_days = 252

    mu_daily = returns.mean() + 0.5 * returns.var()
    sigma_daily = returns.std()

    gbm_params = pd.DataFrame({
        "parameter": [
            "mu_daily",
            "sigma_daily",
            "mu_annualized",
            "sigma_annualized"
        ],
        "value": [
            mu_daily,
            sigma_daily,
            mu_daily * trading_days,
            sigma_daily * np.sqrt(trading_days)
        ]
    })

    gbm_params_path = TABLES_DIR / "gbm_parameters.csv"
    gbm_params.to_csv(gbm_params_path, index=False)

    # ============================================================
    # Generación de gráficos
    # ============================================================

    plot_price_series(df)
    plot_returns_series(df)
    plot_returns_histogram(returns)
    plot_returns_normal_fit(returns)
    plot_autocorrelation(returns)

    # ============================================================
    # Mensajes de salida
    # ============================================================

    print("Análisis estadístico de retornos completado.")
    print(f"Número de datos analizados: {len(df)}")
    print(f"Media de retornos: {returns.mean():.8f}")
    print(f"Varianza: {returns.var():.8f}")
    print(f"Desviación estándar: {returns.std():.8f}")
    print(f"Curtosis: {returns.kurtosis():.8f}")
    print(f"Asimetría: {returns.skew():.8f}")
    print(f"Mínimo retorno: {returns.min():.8f}")
    print(f"Máximo retorno: {returns.max():.8f}")
    print(f"Estadísticos guardados en: {stats_path}")
    print(f"Parámetros GBM guardados en: {gbm_params_path}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()