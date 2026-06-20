"""
Sensibilidad temporal del modelo Random Forest.

Se entrena el mismo modelo con distintas ventanas históricas y se evalúa
sobre un tramo de validación fijo usando predicción one-step.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "gold_prices_clean.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures" / "04_ml_comparison"
TABLES_DIR = BASE_DIR / "outputs" / "tables" / "04_ml_comparison"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

START_YEARS = [2009, 2018, 2024]
TEST_SIZE = 252
RANDOM_SEED = 42


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
        "Gold_Price", "gold_price",
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

    El objetivo es predecir:

        r(t+1)

    usando información disponible hasta el tiempo t.
    """

    df = data.copy()

    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    df["return_mean_5"] = df["log_return"].rolling(window=5).mean()
    df["return_mean_20"] = df["log_return"].rolling(window=20).mean()

    df["return_std_5"] = df["log_return"].rolling(window=5).std()
    df["return_std_20"] = df["log_return"].rolling(window=20).std()

    df["return_sum_5"] = df["log_return"].rolling(window=5).sum()
    df["return_sum_20"] = df["log_return"].rolling(window=20).sum()

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


def reconstruct_one_step_prices(previous_real_prices, predicted_returns):
    """
    Reconstruye precios one-step:

        S_pred(t+1) = S_real(t) exp(r_pred(t+1))
    """

    return previous_real_prices * np.exp(predicted_returns)


def compute_metrics(real_returns, predicted_returns, real_prices, predicted_prices):
    """
    Calcula métricas de error para retornos y precios.
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


def train_random_forest(X_train, y_train):
    """
    Entrena un Random Forest Regressor con hiperparámetros simples
    y conservadores para evitar sobreajuste excesivo.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def main():
    # Datos y variables

    price_data = load_price_data(DATA_PATH)
    ml_data, feature_cols = create_features(price_data)

    if len(ml_data) <= TEST_SIZE:
        raise ValueError("No hay suficientes datos para separar 252 observaciones de test.")

    # Tramo de validación

    train_full = ml_data.iloc[:-TEST_SIZE].copy()
    test_data = ml_data.iloc[-TEST_SIZE:].copy()

    X_test = test_data[feature_cols]
    y_test = test_data["target_return"].to_numpy()

    test_dates = test_data["date"].to_numpy()

    previous_real_prices = test_data["price"].to_numpy()
    real_next_prices = previous_real_prices * np.exp(y_test)

    # Random Walk one-step

    rw_pred_returns = np.zeros_like(y_test)
    rw_pred_prices = reconstruct_one_step_prices(previous_real_prices, rw_pred_returns)

    rw_metrics = compute_metrics(
        real_returns=y_test,
        predicted_returns=rw_pred_returns,
        real_prices=real_next_prices,
        predicted_prices=rw_pred_prices,
    )

    results = [
        {
            "model": "Random_Walk_one_step",
            "start_year": "baseline",
            "n_train_observations": len(train_full),
            **rw_metrics,
        }
    ]

    prediction_curves = {
        "Precio real t+1": real_next_prices,
        "Random Walk one-step": rw_pred_prices,
    }

    feature_importances = []

    # Entrenamiento por ventanas

    for start_year in START_YEARS:
        train_window = train_full[train_full["date"].dt.year >= start_year].copy()

        if len(train_window) < 100:
            print(f"Ventana {start_year}: omitida por tener pocos datos.")
            continue

        X_train = train_window[feature_cols]
        y_train = train_window["target_return"]

        rf_model = train_random_forest(X_train, y_train)

        rf_pred_returns = rf_model.predict(X_test)
        rf_pred_prices = reconstruct_one_step_prices(previous_real_prices, rf_pred_returns)

        metrics = compute_metrics(
            real_returns=y_test,
            predicted_returns=rf_pred_returns,
            real_prices=real_next_prices,
            predicted_prices=rf_pred_prices,
        )

        results.append(
            {
                "model": "Random_Forest",
                "start_year": start_year,
                "n_train_observations": len(train_window),
                "train_start_date": train_window["date"].iloc[0].date(),
                "train_end_date": train_window["date"].iloc[-1].date(),
                **metrics,
            }
        )

        prediction_curves[f"RF desde {start_year}"] = rf_pred_prices

        for feature, importance in zip(feature_cols, rf_model.feature_importances_):
            feature_importances.append(
                {
                    "start_year": start_year,
                    "feature": feature,
                    "importance": importance,
                }
            )

    results_df = pd.DataFrame(results)
    importance_df = pd.DataFrame(feature_importances)

    # Guardar resultados

    metrics_path = TABLES_DIR / "ml_window_sensitivity_metrics.csv"
    importance_path = TABLES_DIR / "ml_window_sensitivity_feature_importance.csv"

    results_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    # Gráfico de precios

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        test_dates,
        real_next_prices,
        label="Precio real t+1",
        linewidth=2.3,
    )

    ax.plot(
        test_dates,
        rw_pred_prices,
        label="Random Walk one-step",
        linewidth=1.8,
        linestyle=":",
    )

    line_styles = {
        2009: "--",
        2018: "-.",
        2024: (0, (5, 2)),
    }

    for start_year in START_YEARS:
        label = f"RF desde {start_year}"
        if label in prediction_curves:
            ax.plot(
                test_dates,
                prediction_curves[label],
                label=label,
                linewidth=1.8,
                linestyle=line_styles.get(start_year, "--"),
            )

    ax.set_title("Sensibilidad temporal del Random Forest")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio del oro [US$/ozt]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    price_figure_path = FIGURES_DIR / "ml_window_sensitivity_price.png"
    plt.savefig(price_figure_path, dpi=300)
    plt.close()

    # Gráfico de error

    rf_only = results_df[results_df["model"] == "Random_Forest"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        rf_only["start_year"].astype(str),
        rf_only["MAPE_price_percent"],
    )

    ax.set_title("MAPE one-step del Random Forest según ventana histórica")
    ax.set_xlabel("Año inicial de entrenamiento")
    ax.set_ylabel("MAPE [%]")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    error_figure_path = FIGURES_DIR / "ml_window_sensitivity_mape.png"
    plt.savefig(error_figure_path, dpi=300)
    plt.close()

    # Resumen final

    print("Sensibilidad temporal del Random Forest completada.")
    print()
    print(f"Datos disponibles para ML: {len(ml_data)}")
    print(f"Test fijo: {len(test_data)} observaciones")
    print(f"Fecha inicial test: {test_data['date'].iloc[0].date()}")
    print(f"Fecha final test: {test_data['date'].iloc[-1].date()}")
    print()
    print("Métricas:")
    print(
        results_df[
            [
                "model",
                "start_year",
                "n_train_observations",
                "MAE_return",
                "RMSE_return",
                "MAE_price",
                "RMSE_price",
                "MAPE_price_percent",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Tabla de métricas guardada en: {metrics_path}")
    print(f"Importancia de variables guardada en: {importance_path}")
    print()
    print(f"Gráfico de precios guardado en: {price_figure_path}")
    print(f"Gráfico de MAPE guardado en: {error_figure_path}")


if __name__ == "__main__":
    main()
