# ============================================================
# Comparación ML base: Random Forest vs modelos simples
# Proyecto FIS205 - Modelamiento estocástico del precio del oro
# ============================================================
#
# Este script agrega una extensión exploratoria de Machine Learning
# al proyecto principal de GBM y Monte Carlo.
#
# La idea NO es construir una estrategia de trading ni afirmar capacidad
# predictiva financiera fuerte. El objetivo es comparar un modelo
# supervisado simple con modelos base:
#
#   1. Random Walk one-step
#   2. GBM expected return
#   3. Random Forest
#
# La evaluación es one-step ahead:
#
#   Se predice el retorno logarítmico del día siguiente usando
#   información histórica disponible hasta el día actual.
#
# Esto es distinto a una simulación libre de 252 días como en el GBM
# Monte Carlo. Por eso, esta sección debe interpretarse como una
# extensión computacional comparativa.


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuración general
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_clean.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures" / "04_ml_comparison"
TABLES_DIR = BASE_DIR / "outputs" / "tables" / "04_ml_comparison"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 252
RANDOM_SEED = 42


# ============================================================
# Funciones auxiliares
# ============================================================

def load_price_data(path):
    """
    Carga la serie limpia de precios del oro.
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
        raise ValueError("No se encontró columna de fecha.")

    if price_col is None:
        raise ValueError("No se encontró columna de precio.")

    data = data[[date_col, price_col]].copy()
    data.columns = ["date", "price"]

    data["date"] = pd.to_datetime(data["date"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna()
    data = data.sort_values("date").reset_index(drop=True)

    return data


def create_features(data):
    """
    Construye variables predictoras usando retornos logarítmicos pasados.

    La variable objetivo es el retorno del día siguiente:

        target_return = r(t+1)

    Las variables predictoras usan información disponible hasta t.
    """

    df = data.copy()

    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    # Retornos rezagados
    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    # Medias móviles de retornos
    df["return_mean_5"] = df["log_return"].rolling(window=5).mean()
    df["return_mean_20"] = df["log_return"].rolling(window=20).mean()

    # Volatilidades móviles
    df["return_std_5"] = df["log_return"].rolling(window=5).std()
    df["return_std_20"] = df["log_return"].rolling(window=20).std()

    # Retornos acumulados recientes
    df["return_sum_5"] = df["log_return"].rolling(window=5).sum()
    df["return_sum_20"] = df["log_return"].rolling(window=20).sum()

    # Objetivo: retorno del día siguiente
    df["target_return"] = df["log_return"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_4",
        "return_lag_5",
        "return_mean_5",
        "return_mean_20",
        "return_std_5",
        "return_std_20",
        "return_sum_5",
        "return_sum_20",
    ]

    return df, feature_cols


def estimate_gbm_expected_return(train_prices):
    """
    Estima el retorno logarítmico esperado del GBM.

    Para GBM:

        E[r] = mu - 0.5 sigma^2

    Al estimar desde datos, esto equivale simplemente a la media de los
    retornos logarítmicos del tramo de entrenamiento.
    """

    log_returns = np.log(train_prices[1:] / train_prices[:-1])

    expected_return = np.mean(log_returns)

    mu_daily = np.mean(log_returns) + 0.5 * np.var(log_returns, ddof=1)
    sigma_daily = np.std(log_returns, ddof=1)

    return expected_return, mu_daily, sigma_daily


def reconstruct_one_step_prices(previous_real_prices, predicted_returns):
    """
    Reconstruye precios predichos one-step:

        S_pred(t+1) = S_real(t) * exp(r_pred(t+1))

    Aquí se usa el precio real del día anterior porque la evaluación es
    de un paso adelante, no una trayectoria libre.
    """

    predicted_prices = previous_real_prices * np.exp(predicted_returns)

    return predicted_prices


def compute_metrics(real_returns, predicted_returns, real_prices, predicted_prices):
    """
    Calcula métricas tanto en retornos como en precios.
    """

    mae_return = mean_absolute_error(real_returns, predicted_returns)
    rmse_return = np.sqrt(mean_squared_error(real_returns, predicted_returns))

    mae_price = mean_absolute_error(real_prices, predicted_prices)
    rmse_price = np.sqrt(mean_squared_error(real_prices, predicted_prices))
    mape_price = np.mean(np.abs((predicted_prices - real_prices) / real_prices)) * 100

    return {
        "MAE_return": mae_return,
        "RMSE_return": rmse_return,
        "MAE_price": mae_price,
        "RMSE_price": rmse_price,
        "MAPE_price_percent": mape_price,
    }


# ============================================================
# Programa principal
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Cargar datos y construir features
    # --------------------------------------------------------

    price_data = load_price_data(DATA_PATH)
    ml_data, feature_cols = create_features(price_data)

    if len(ml_data) <= TEST_SIZE:
        raise ValueError("No hay suficientes datos para separar 252 observaciones de test.")

    # --------------------------------------------------------
    # 2. Separar train/test de forma temporal
    # --------------------------------------------------------

    train_data = ml_data.iloc[:-TEST_SIZE].copy()
    test_data = ml_data.iloc[-TEST_SIZE:].copy()

    X_train = train_data[feature_cols]
    y_train = train_data["target_return"]

    X_test = test_data[feature_cols]
    y_test = test_data["target_return"].to_numpy()

    test_dates = test_data["date"].to_numpy()

    # Precio real del día t y del día t+1
    previous_real_prices = test_data["price"].to_numpy()
    real_next_prices = previous_real_prices * np.exp(y_test)

    # --------------------------------------------------------
    # 3. Modelo base: Random Walk one-step
    # --------------------------------------------------------
    #
    # Random Walk one-step equivale a predecir retorno cero:
    #
    #   r_pred(t+1) = 0
    #
    # Por tanto:
    #
    #   S_pred(t+1) = S_real(t)

    rw_pred_returns = np.zeros_like(y_test)
    rw_pred_prices = reconstruct_one_step_prices(previous_real_prices, rw_pred_returns)

    # --------------------------------------------------------
    # 4. Modelo base: GBM expected return
    # --------------------------------------------------------

    train_prices_for_gbm = train_data["price"].to_numpy()

    gbm_expected_return, mu_daily, sigma_daily = estimate_gbm_expected_return(
        train_prices_for_gbm
    )

    gbm_pred_returns = np.full_like(y_test, fill_value=gbm_expected_return)
    gbm_pred_prices = reconstruct_one_step_prices(previous_real_prices, gbm_pred_returns)

    # --------------------------------------------------------
    # 5. Modelo ML: Random Forest
    # --------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    rf_pred_returns = rf_model.predict(X_test)
    rf_pred_prices = reconstruct_one_step_prices(previous_real_prices, rf_pred_returns)

    # --------------------------------------------------------
    # 6. Calcular métricas
    # --------------------------------------------------------

    metrics = []

    metrics.append(
        {
            "model": "Random_Walk_one_step",
            **compute_metrics(
                real_returns=y_test,
                predicted_returns=rw_pred_returns,
                real_prices=real_next_prices,
                predicted_prices=rw_pred_prices,
            ),
        }
    )

    metrics.append(
        {
            "model": "GBM_expected_return",
            **compute_metrics(
                real_returns=y_test,
                predicted_returns=gbm_pred_returns,
                real_prices=real_next_prices,
                predicted_prices=gbm_pred_prices,
            ),
        }
    )

    metrics.append(
        {
            "model": "Random_Forest",
            **compute_metrics(
                real_returns=y_test,
                predicted_returns=rf_pred_returns,
                real_prices=real_next_prices,
                predicted_prices=rf_pred_prices,
            ),
        }
    )

    metrics_df = pd.DataFrame(metrics)

    metrics_path = TABLES_DIR / "ml_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------
    # 7. Guardar predicciones
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "date": test_dates,
            "previous_real_price": previous_real_prices,
            "real_next_price": real_next_prices,
            "real_return": y_test,
            "rw_pred_return": rw_pred_returns,
            "gbm_pred_return": gbm_pred_returns,
            "rf_pred_return": rf_pred_returns,
            "rw_pred_price": rw_pred_prices,
            "gbm_pred_price": gbm_pred_prices,
            "rf_pred_price": rf_pred_prices,
        }
    )

    predictions_path = TABLES_DIR / "ml_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # --------------------------------------------------------
    # 8. Gráfico de precios one-step
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        test_dates,
        real_next_prices,
        label="Precio real t+1",
        linewidth=2.2,
    )

    ax.plot(
        test_dates,
        rw_pred_prices,
        label="Random Walk one-step",
        linewidth=1.8,
        linestyle=":",
    )

    ax.plot(
        test_dates,
        gbm_pred_prices,
        label="GBM expected return",
        linewidth=1.8,
        linestyle="--",
    )

    ax.plot(
        test_dates,
        rf_pred_prices,
        label="Random Forest",
        linewidth=1.8,
        linestyle="-.",
    )

    ax.set_title("Comparación one-step de modelos")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio del oro [US$/ozt]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    price_figure_path = FIGURES_DIR / "ml_price_comparison.png"
    plt.savefig(price_figure_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 9. Gráfico de retornos reales vs predichos
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        test_dates,
        y_test,
        label="Retorno real",
        linewidth=1.8,
    )

    ax.plot(
        test_dates,
        rf_pred_returns,
        label="Retorno predicho Random Forest",
        linewidth=1.8,
        linestyle="--",
    )

    ax.axhline(0, linewidth=1.0)

    ax.set_title("Retornos reales vs retornos predichos por Random Forest")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno logarítmico")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    returns_figure_path = FIGURES_DIR / "ml_return_predictions.png"
    plt.savefig(returns_figure_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 10. Importancia de variables
    # --------------------------------------------------------

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    importance_path = TABLES_DIR / "ml_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    ax.invert_yaxis()
    ax.set_title("Importancia de variables en Random Forest")
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("Variable")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    importance_figure_path = FIGURES_DIR / "ml_feature_importance.png"
    plt.savefig(importance_figure_path, dpi=300)
    plt.close()

    # --------------------------------------------------------
    # 11. Mostrar resumen
    # --------------------------------------------------------

    print("Comparación ML base completada.")
    print()
    print(f"Datos disponibles para ML: {len(ml_data)}")
    print(f"Train: {len(train_data)} observaciones")
    print(f"Test: {len(test_data)} observaciones")
    print()
    print(f"Fecha inicial test: {test_data['date'].iloc[0].date()}")
    print(f"Fecha final test: {test_data['date'].iloc[-1].date()}")
    print()
    print(f"GBM expected return diario: {gbm_expected_return:.8f}")
    print(f"GBM mu diario: {mu_daily:.8f}")
    print(f"GBM sigma diario: {sigma_daily:.8f}")
    print()
    print("Métricas:")
    print(metrics_df.to_string(index=False))
    print()
    print("Importancia de variables:")
    print(importance_df.to_string(index=False))
    print()
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Predicciones guardadas en: {predictions_path}")
    print(f"Importancia guardada en: {importance_path}")
    print()
    print(f"Gráfico precios guardado en: {price_figure_path}")
    print(f"Gráfico retornos guardado en: {returns_figure_path}")
    print(f"Gráfico importancia guardado en: {importance_figure_path}")


if __name__ == "__main__":
    main()
