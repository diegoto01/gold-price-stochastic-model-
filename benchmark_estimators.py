import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "gold_returns.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables" / "04_ml_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
N_ESTIMATORS_LIST = [300, 1000, 10000]
TEST_DAYS = 252


def create_features(df):
    df = df.copy()

    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["Return"].shift(lag)

    for window in [5, 10, 20]:
        df[f"return_mean_{window}"] = df["Return"].rolling(window).mean()
        df[f"return_std_{window}"] = df["Return"].rolling(window).std()
        df[f"return_sum_{window}"] = df["Return"].rolling(window).sum()

    df["target_return"] = df["Return"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_price_prediction(y_true_return, y_pred_return, price_t, price_next):
    pred_price = price_t * np.exp(y_pred_return)

    mae_price = mean_absolute_error(price_next, pred_price)
    rmse_price = np.sqrt(mean_squared_error(price_next, pred_price))
    mape_price = np.mean(np.abs((price_next - pred_price) / price_next)) * 100

    rmse_return = np.sqrt(mean_squared_error(y_true_return, y_pred_return))
    mae_return = mean_absolute_error(y_true_return, y_pred_return)

    return {
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mape_price_percent": mape_price,
        "mae_return": mae_return,
        "rmse_return": rmse_return,
    }


def run_benchmark():
    df = pd.read_csv(DATA_PATH)
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df = create_features(df)

    feature_cols = [
        col for col in df.columns
        if col.startswith("return_lag_")
        or col.startswith("return_mean_")
        or col.startswith("return_std_")
        or col.startswith("return_sum_")
    ]

    train_df = df.iloc[:-TEST_DAYS].copy()
    test_df = df.iloc[-TEST_DAYS:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_return"]

    price_t = test_df["Precio"].to_numpy()
    price_next = test_df["Precio"].shift(-1).to_numpy()

    valid_mask = ~np.isnan(price_next)

    X_test = X_test.iloc[valid_mask]
    y_test = y_test.iloc[valid_mask]
    price_t = price_t[valid_mask]
    price_next = price_next[valid_mask]

    results = []

    for n_estimators in N_ESTIMATORS_LIST:
        print(f"\nRandom Forest con {n_estimators} árboles")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        metrics = evaluate_price_prediction(
            y_true_return=y_test,
            y_pred_return=y_pred,
            price_t=price_t,
            price_next=price_next,
        )

        row = {
            "model": "Random Forest",
            "n_estimators": n_estimators,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
            "total_time_seconds": train_time + predict_time,
            **metrics,
        }

        results.append(row)

        print(
            f"MAPE: {row['mape_price_percent']:.4f}% | "
            f"RMSE precio: {row['rmse_price']:.4f} | "
            f"Tiempo total: {row['total_time_seconds']:.2f} s"
        )

    if XGBOOST_AVAILABLE:
        for n_estimators in N_ESTIMATORS_LIST:
            print(f"\nXGBoost con {n_estimators} árboles")

            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time = time.perf_counter() - t1

            metrics = evaluate_price_prediction(
                y_true_return=y_test,
                y_pred_return=y_pred,
                price_t=price_t,
                price_next=price_next,
            )

            row = {
                "model": "XGBoost",
                "n_estimators": n_estimators,
                "train_time_seconds": train_time,
                "predict_time_seconds": predict_time,
                "total_time_seconds": train_time + predict_time,
                **metrics,
            }

            results.append(row)

            print(
                f"MAPE: {row['mape_price_percent']:.4f}% | "
                f"RMSE precio: {row['rmse_price']:.4f} | "
                f"Tiempo total: {row['total_time_seconds']:.2f} s"
            )

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "estimator_sensitivity.csv"
    results_df.to_csv(output_path, index=False)

    print("\nResumen:")
    print(results_df)
    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    run_benchmark()
