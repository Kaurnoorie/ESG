"""Model training helpers used by the trend and relationship analyses."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def _default_models() -> dict:
    return {
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbosity=0,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_time_series_models(
    years: np.ndarray,
    values: np.ndarray,
    future_years: list[int],
) -> dict:
    """Train forecasting models using year as the single feature."""

    X = np.asarray(years).reshape(-1, 1)
    y = np.asarray(values)
    X_future = np.asarray(future_years).reshape(-1, 1)

    results = {}
    for model_name, model in _default_models().items():
        model.fit(X, y)
        training_predictions = model.predict(X)
        future_predictions = model.predict(X_future)

        results[model_name] = {
            "model": model,
            "predictions_train": training_predictions,
            "predictions_future": future_predictions,
            **_regression_metrics(y, training_predictions),
        }

    return results


def train_relationship_models(esg_scores: np.ndarray, values: np.ndarray) -> dict:
    """Train models that predict a financial KPI from ESG score."""

    X = np.asarray(esg_scores).reshape(-1, 1)
    y = np.asarray(values)

    results = {}
    for model_name, model in _default_models().items():
        model.fit(X, y)
        predictions = model.predict(X)

        results[model_name] = {
            "model": model,
            "predictions": predictions,
            **_regression_metrics(y, predictions),
        }

    return results


def build_prediction_dataframe(future_years: list[int], model_results: dict) -> pd.DataFrame:
    """Create the wide predictions table used by the original notebook."""

    predictions = pd.DataFrame({"Year": future_years})
    for metric_name, results in model_results.items():
        for model_name, result in results.items():
            column = f"{metric_name}_{model_name.replace(' ', '_')}"
            predictions[column] = result["predictions_future"]

    return predictions


def build_metrics_dataframe(model_results: dict) -> pd.DataFrame:
    """Flatten model performance metrics into a CSV-friendly table."""

    rows = []
    for metric_name, results in model_results.items():
        for model_name, result in results.items():
            rows.append(
                {
                    "Metric": metric_name.replace("_", " "),
                    "Model": model_name,
                    "RMSE": result["rmse"],
                    "MAE": result["mae"],
                    "R2": result["r2"],
                }
            )

    return pd.DataFrame(rows)


def best_model_name(results: dict) -> str:
    """Return the model with the highest training R2."""

    return max(results.keys(), key=lambda name: results[name]["r2"])
