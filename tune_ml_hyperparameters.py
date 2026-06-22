import time
from itertools import product
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
VALIDATION_DAYS = 252
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

    return df.dropna().reset_index(drop=True)


def evaluate_prediction(y_true_return, y_pred_return, price_t, price_next):
    pred_price = price_t * np.exp(y_pred_return)

    mae_price = mean_absolute_error(price_next, pred_price)
    rmse_price = np.sqrt(mean_squared_error(price_next, pred_price))
    mape_price = np.mean(np.abs((price_next - pred_price) / price_next)) * 100

    mae_return = mean_absolute_error(y_true_return, y_pred_return)
    rmse_return = np.sqrt(mean_squared_error(y_true_return, y_pred_return))

    return {
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mape_price_percent": mape_price,
        "mae_return": mae_return,
        "rmse_return": rmse_return,
    }


def prepare_xy(df, feature_cols):
    X = df[feature_cols]
    y = df["target_return"]

    price_t = df["Precio"].to_numpy()
    price_next = df["Precio"].shift(-1).to_numpy()

    valid_mask = ~np.isnan(price_next)

    return (
        X.iloc[valid_mask],
        y.iloc[valid_mask],
        price_t[valid_mask],
        price_next[valid_mask],
    )


def run_random_forest_grid(X_train, y_train, X_val, y_val, price_t_val, price_next_val):
    results = []

    grid = {
        "n_estimators": [300, 1000],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", 1.0],
    }

    keys = list(grid.keys())

    for values in product(*grid.values()):
        params = dict(zip(keys, values))

        print(f"\nRandom Forest | {params}")

        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_val)
        predict_time = time.perf_counter() - t1

        metrics = evaluate_prediction(
            y_true_return=y_val,
            y_pred_return=y_pred,
            price_t=price_t_val,
            price_next=price_next_val,
        )

        row = {
            "model": "Random Forest",
            **params,
            "learning_rate": np.nan,
            "subsample": np.nan,
            "colsample_bytree": np.nan,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
            "total_time_seconds": train_time + predict_time,
            **metrics,
        }

        results.append(row)

        print(
            f"Validación | MAPE: {row['mape_price_percent']:.4f}% | "
            f"RMSE precio: {row['rmse_price']:.4f} | "
            f"Tiempo: {row['total_time_seconds']:.2f} s"
        )

    return results


def run_xgboost_grid(X_train, y_train, X_val, y_val, price_t_val, price_next_val):
    results = []

    if not XGBOOST_AVAILABLE:
        print("\nXGBoost no está instalado. Se omite esta parte.")
        return results

    grid = {
        "n_estimators": [300, 1000],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    keys = list(grid.keys())

    for values in product(*grid.values()):
        params = dict(zip(keys, values))

        print(f"\nXGBoost | {params}")

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_val)
        predict_time = time.perf_counter() - t1

        metrics = evaluate_prediction(
            y_true_return=y_val,
            y_pred_return=y_pred,
            price_t=price_t_val,
            price_next=price_next_val,
        )

        row = {
            "model": "XGBoost",
            "min_samples_leaf": np.nan,
            "max_features": np.nan,
            **params,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
            "total_time_seconds": train_time + predict_time,
            **metrics,
        }

        results.append(row)

        print(
            f"Validación | MAPE: {row['mape_price_percent']:.4f}% | "
            f"RMSE precio: {row['rmse_price']:.4f} | "
            f"Tiempo: {row['total_time_seconds']:.2f} s"
        )

    return results


def evaluate_best_models(results_df, train_val_df, test_df, feature_cols):
    test_results = []

    X_train_val = train_val_df[feature_cols]
    y_train_val = train_val_df["target_return"]

    X_test, y_test, price_t_test, price_next_test = prepare_xy(test_df, feature_cols)

    for model_name in results_df["model"].unique():
        best_row = (
            results_df[results_df["model"] == model_name]
            .sort_values("mape_price_percent")
            .iloc[0]
        )

        print(f"\nMejor modelo en validación: {model_name}")
        print(best_row)

        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=int(best_row["n_estimators"]),
                max_depth=None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
                min_samples_leaf=int(best_row["min_samples_leaf"]),
                max_features=best_row["max_features"],
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )

        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=int(best_row["n_estimators"]),
                learning_rate=float(best_row["learning_rate"]),
                max_depth=int(best_row["max_depth"]),
                subsample=float(best_row["subsample"]),
                colsample_bytree=float(best_row["colsample_bytree"]),
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )

        else:
            continue

        t0 = time.perf_counter()
        model.fit(X_train_val, y_train_val)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        metrics = evaluate_prediction(
            y_true_return=y_test,
            y_pred_return=y_pred,
            price_t=price_t_test,
            price_next=price_next_test,
        )

        row = {
            "model": model_name,
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": best_row["max_depth"],
            "min_samples_leaf": best_row["min_samples_leaf"],
            "max_features": best_row["max_features"],
            "learning_rate": best_row["learning_rate"],
            "subsample": best_row["subsample"],
            "colsample_bytree": best_row["colsample_bytree"],
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
            "total_time_seconds": train_time + predict_time,
            **metrics,
        }

        test_results.append(row)

        print(
            f"Test final | MAPE: {row['mape_price_percent']:.4f}% | "
            f"RMSE precio: {row['rmse_price']:.4f} | "
            f"Tiempo: {row['total_time_seconds']:.2f} s"
        )

    return pd.DataFrame(test_results)


def main():
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

    test_df = df.iloc[-TEST_DAYS:].copy()
    validation_df = df.iloc[-(TEST_DAYS + VALIDATION_DAYS):-TEST_DAYS].copy()
    train_df = df.iloc[:-(TEST_DAYS + VALIDATION_DAYS)].copy()
    train_val_df = df.iloc[:-TEST_DAYS].copy()

    print("Tamaños de los conjuntos:")
    print(f"Entrenamiento: {len(train_df)} datos")
    print(f"Validación:    {len(validation_df)} datos")
    print(f"Test final:    {len(test_df)} datos")

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]

    X_val, y_val, price_t_val, price_next_val = prepare_xy(validation_df, feature_cols)

    results = []
    results.extend(
        run_random_forest_grid(
            X_train, y_train,
            X_val, y_val,
            price_t_val, price_next_val,
        )
    )
    results.extend(
        run_xgboost_grid(
            X_train, y_train,
            X_val, y_val,
            price_t_val, price_next_val,
        )
    )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["model", "mape_price_percent"]).reset_index(drop=True)

    validation_output = OUTPUT_DIR / "hyperparameter_tuning_validation.csv"
    results_df.to_csv(validation_output, index=False)

    test_results_df = evaluate_best_models(
        results_df=results_df,
        train_val_df=train_val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    test_output = OUTPUT_DIR / "hyperparameter_tuning_test.csv"
    test_results_df.to_csv(test_output, index=False)

    print("\nMejores resultados de validación:")
    print(results_df.groupby("model").head(5))

    print("\nResultados finales en test:")
    print(test_results_df)

    print(f"\nResultados de validación guardados en: {validation_output}")
    print(f"Resultados de test guardados en: {test_output}")


if __name__ == "__main__":
    main()
