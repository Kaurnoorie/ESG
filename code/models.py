"""Model training helpers used by the trend and relationship analyses."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def _time_series_models() -> dict:
    from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
    }


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

    # Chronological train-test split (test on the last 3 years)
    test_size = 3
    if len(X) > test_size + 2:
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y

    results = {}
    for model_name, _ in _time_series_models().items():
        # Train and evaluate on split to get honest metrics
        model_eval = _time_series_models()[model_name]
        model_eval.fit(X_train, y_train)
        test_predictions = model_eval.predict(X_test)
        metrics = _regression_metrics(y_test, test_predictions)

        # Train full model for returning actual future predictions and plotting
        model_full = _time_series_models()[model_name]
        model_full.fit(X, y)
        training_predictions = model_full.predict(X)
        future_predictions = model_full.predict(X_future)

        results[model_name] = {
            "model": model_full,
            "predictions_train": training_predictions,
            "predictions_future": future_predictions,
            **metrics,
        }

    return results


def train_relationship_models(esg_scores: np.ndarray, values: np.ndarray) -> dict:
    """Train models that predict a financial KPI from ESG score."""

    X = np.asarray(esg_scores).reshape(-1, 1)
    y = np.asarray(values)

    # Random train-test split for relationship evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for model_name, _ in _default_models().items():
        # Train and evaluate on split to get honest metrics
        model_eval = _default_models()[model_name]
        model_eval.fit(X_train, y_train)
        test_predictions = model_eval.predict(X_test)
        metrics = _regression_metrics(y_test, test_predictions)

        # Train full model for returning overall predictions for plotting
        model_full = _default_models()[model_name]
        model_full.fit(X, y)
        predictions = model_full.predict(X)

        results[model_name] = {
            "model": model_full,
            "predictions": predictions,
            **metrics,
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
