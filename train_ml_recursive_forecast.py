"""
Pronóstico recursivo con Random Forest.

El modelo parte desde el precio inicial del tramo de validación y
genera una trayectoria completa usando sus propios retornos predichos.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


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


def create_training_features(data):
    """
    Crea una base supervisada para entrenar Random Forest.

    Variables de entrada:
    - retornos rezagados
    - medias móviles
    - volatilidades móviles
    - retornos acumulados

    Variable objetivo:
    - retorno logarítmico del día siguiente
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


def build_feature_vector(recent_returns):
    """
    Construye un vector de características usando los retornos recientes
    disponibles en la trayectoria actual.

    recent_returns contiene retornos reales históricos antes del test y,
    luego, retornos predichos por el propio modelo.
    """

    recent_returns = np.array(recent_returns, dtype=float)

    if len(recent_returns) < 20:
        raise ValueError("Se necesitan al menos 20 retornos recientes.")

    feature_values = {
        "return_lag_1": recent_returns[-1],
        "return_lag_2": recent_returns[-2],
        "return_lag_3": recent_returns[-3],
        "return_lag_4": recent_returns[-4],
        "return_lag_5": recent_returns[-5],
        "return_mean_5": np.mean(recent_returns[-5:]),
        "return_mean_20": np.mean(recent_returns[-20:]),
        "return_std_5": np.std(recent_returns[-5:], ddof=1),
        "return_std_20": np.std(recent_returns[-20:], ddof=1),
        "return_sum_5": np.sum(recent_returns[-5:]),
        "return_sum_20": np.sum(recent_returns[-20:]),
    }

    return feature_values


def train_random_forest(X_train, y_train):
    """
    Entrena un Random Forest Regressor con hiperparámetros conservadores.
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


def recursive_forecast(model, feature_cols, initial_returns, S0, n_steps):
    """
    Genera una trayectoria recursiva.

    En cada paso:
    - se construyen features desde los retornos recientes
    - se predice el retorno siguiente
    - se actualiza el precio predicho
    - se agrega el retorno predicho a la historia
    """

    predicted_prices = [S0]
    predicted_returns = []

    recent_returns = list(initial_returns)
    current_price = S0

    for _ in range(n_steps):
        feature_dict = build_feature_vector(recent_returns)
        X_next = pd.DataFrame([feature_dict])[feature_cols]

        r_pred = model.predict(X_next)[0]

        next_price = current_price * np.exp(r_pred)

        predicted_returns.append(r_pred)
        predicted_prices.append(next_price)

        recent_returns.append(r_pred)
        current_price = next_price

    return np.array(predicted_prices), np.array(predicted_returns)


def compute_trajectory_metrics(real_prices, predicted_prices):
    """
    Calcula métricas de error para trayectorias completas.
    """

    errors = predicted_prices - real_prices

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / real_prices)) * 100

    final_abs_error = np.abs(predicted_prices[-1] - real_prices[-1])
    final_pct_error = final_abs_error / real_prices[-1] * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "final_absolute_error": final_abs_error,
        "final_percentage_error": final_pct_error,
    }


def main():
    # Datos y tramo de validación

    price_data = load_price_data(DATA_PATH)

    if len(price_data) <= TEST_SIZE:
        raise ValueError("La serie es demasiado corta para usar 252 días de test.")

    train_full = price_data.iloc[:-TEST_SIZE].copy()
    test_data = price_data.iloc[-TEST_SIZE:].copy()

    test_dates = test_data["date"].to_numpy()
    test_prices = test_data["price"].to_numpy()

    test_start_date = test_data["date"].iloc[0]

    S0 = test_prices[0]
    n_steps = len(test_prices) - 1

    # Random Walk constante

    rw_constant_path = np.full_like(test_prices, fill_value=S0, dtype=float)

    results = [
        {
            "model": "Random_Walk_constant",
            "start_year": "baseline",
            "n_train_prices": len(train_full),
            **compute_trajectory_metrics(test_prices, rw_constant_path),
        }
    ]

    predicted_paths = {
        "Random Walk constante": rw_constant_path,
    }

    # Entrenamiento por ventanas

    for start_year in START_YEARS:
        train_window = train_full[train_full["date"].dt.year >= start_year].copy()

        if len(train_window) < 120:
            print(f"Ventana {start_year}: omitida por tener pocos datos.")
            continue

        ml_train, feature_cols = create_training_features(train_window)

        if len(ml_train) < 100:
            print(f"Ventana {start_year}: omitida por tener pocos datos útiles.")
            continue

        X_train = ml_train[feature_cols]
        y_train = ml_train["target_return"]

        model = train_random_forest(X_train, y_train)

        # Para iniciar la trayectoria recursiva se usan los últimos
        # retornos reales disponibles antes y hasta S0.
        history_for_initial = price_data[
            (price_data["date"].dt.year >= start_year)
            & (price_data["date"] <= test_start_date)
        ].copy()

        history_prices = history_for_initial["price"].to_numpy()
        initial_returns = np.log(history_prices[1:] / history_prices[:-1])

        if len(initial_returns) < 20:
            print(f"Ventana {start_year}: omitida por no tener retornos iniciales suficientes.")
            continue

        rf_path, rf_returns = recursive_forecast(
            model=model,
            feature_cols=feature_cols,
            initial_returns=initial_returns,
            S0=S0,
            n_steps=n_steps,
        )

        predicted_paths[f"RF recursivo desde {start_year}"] = rf_path

        metrics = compute_trajectory_metrics(test_prices, rf_path)

        results.append(
            {
                "model": "Random_Forest_recursive",
                "start_year": start_year,
                "n_train_prices": len(train_window),
                "n_train_samples": len(ml_train),
                "train_start_date": train_window["date"].iloc[0].date(),
                "train_end_date": train_window["date"].iloc[-1].date(),
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    # Guardar métricas

    metrics_path = TABLES_DIR / "ml_recursive_forecast_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    # Guardar trayectorias

    paths_df = pd.DataFrame(
        {
            "date": test_dates,
            "real_price": test_prices,
        }
    )

    for label, path in predicted_paths.items():
        clean_label = (
            label.lower()
            .replace(" ", "_")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        paths_df[clean_label] = path

    paths_path = TABLES_DIR / "ml_recursive_forecast_paths.csv"
    paths_df.to_csv(paths_path, index=False)

    # Gráfico de trayectorias

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        test_dates,
        test_prices,
        label="Precio real",
        linewidth=2.4,
    )

    ax.plot(
        test_dates,
        rw_constant_path,
        label="Random Walk constante",
        linewidth=2.0,
        linestyle=":",
    )

    line_styles = {
        2009: "--",
        2018: "-.",
        2024: (0, (5, 2)),
    }

    for start_year in START_YEARS:
        label = f"RF recursivo desde {start_year}"
        if label in predicted_paths:
            ax.plot(
                test_dates,
                predicted_paths[label],
                label=label,
                linewidth=2.0,
                linestyle=line_styles.get(start_year, "--"),
            )

    ax.set_title("Pronóstico recursivo con Random Forest")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio del oro [US$/ozt]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "ml_recursive_forecast_price.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Gráfico de error

    rf_only = results_df[results_df["model"] == "Random_Forest_recursive"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        rf_only["start_year"].astype(str),
        rf_only["MAPE_percent"],
    )

    ax.set_title("MAPE del Random Forest recursivo según ventana histórica")
    ax.set_xlabel("Año inicial de entrenamiento")
    ax.set_ylabel("MAPE [%]")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    error_figure_path = FIGURES_DIR / "ml_recursive_forecast_mape.png"
    plt.savefig(error_figure_path, dpi=300)
    plt.close()

    # Resumen final

    print("Pronóstico recursivo con Random Forest completado.")
    print()
    print(f"Test fijo: últimos {TEST_SIZE} precios")
    print(f"Fecha inicial test: {test_data['date'].iloc[0].date()}")
    print(f"Fecha final test: {test_data['date'].iloc[-1].date()}")
    print(f"Precio inicial S0: {S0:.4f} [US$/ozt]")
    print()
    print("Métricas de trayectoria:")
    print(
        results_df[
            [
                "model",
                "start_year",
                "n_train_prices",
                "MAE",
                "RMSE",
                "MAPE_percent",
                "final_percentage_error",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Trayectorias guardadas en: {paths_path}")
    print(f"Gráfico de trayectorias guardado en: {figure_path}")
    print(f"Gráfico de MAPE guardado en: {error_figure_path}")


if __name__ == "__main__":
    main()
