"""Run historical trend analysis and 2026-2030 model forecasts."""

import warnings

import pandas as pd

from config import FUTURE_YEARS, OUTPUT_DIR
from data_processing import ensure_output_dir, load_banking_dataset
from models import build_metrics_dataframe, build_prediction_dataframe, train_time_series_models
from trend_plots import plot_detailed_trends, plot_forecast_dashboard, plot_statistics_summary

warnings.filterwarnings("ignore")


def run_trend_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the trend analysis and write CSV/PNG outputs."""

    ensure_output_dir()
    dataset = load_banking_dataset()
    historical_df = dataset.historical

    model_results = {
        "ESG_Score": train_time_series_models(historical_df["Year"], historical_df["ESG_Score"], FUTURE_YEARS),
        "Pretax_ROA": train_time_series_models(historical_df["Year"], historical_df["Pretax_ROA"], FUTURE_YEARS),
        "Pretax_ROE": train_time_series_models(historical_df["Year"], historical_df["Pretax_ROE"], FUTURE_YEARS),
    }

    predictions_df = build_prediction_dataframe(FUTURE_YEARS, model_results)
    metrics_df = build_metrics_dataframe(model_results)

    historical_df.to_csv(OUTPUT_DIR / "banking_historical_data_2011_2025.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "banking_predictions_2026_2030.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "banking_model_performance.csv", index=False)

    plot_forecast_dashboard(historical_df, model_results, FUTURE_YEARS)
    plot_detailed_trends(historical_df, model_results, FUTURE_YEARS)
    plot_statistics_summary(historical_df, model_results)

    return historical_df, predictions_df, metrics_df


def print_summary(historical_df: pd.DataFrame, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    print("=" * 80)
    print("BANKING SECTOR: ESG SCORE, ROE & ROA TREND ANALYSIS")
    print("Historical Trends (2011-2025) & ML Predictions (2026-2030)")
    print("=" * 80)
    print("\nHistorical data:")
    print(historical_df.to_string(index=False))
    print("\nModel performance:")
    print(metrics_df.to_string(index=False))
    print("\nPredictions:")
    print(predictions_df.to_string(index=False))

    model_results_by_metric = {
        metric: metrics_df[metrics_df["Metric"] == metric.replace("_", " ")]
        for metric in ["ESG_Score", "Pretax_ROA", "Pretax_ROE"]
    }

    print("\nKey findings:")
    for metric, metric_rows in model_results_by_metric.items():
        historical_values = historical_df[metric]
        best_row = metric_rows.sort_values("R2", ascending=False).iloc[0]
        best_model = best_row["Model"]
        forecast_column = f"{metric}_{best_model.replace(' ', '_')}"
        print(f"\n{metric.replace('_', ' ')}:")
        print(f"  2011: {historical_values.iloc[0]:.2f}")
        print(f"  2025: {historical_values.iloc[-1]:.2f}")
        print(f"  Change: {historical_values.iloc[-1] - historical_values.iloc[0]:+.2f}")
        print(f"  Best model: {best_model} (R2 = {best_row['R2']:.4f})")
        print(f"  2030 forecast: {predictions_df[forecast_column].iloc[-1]:.2f}")

    print("\nFiles generated in outputs/:")
    print("  banking_esg_roa_roe_analysis.png")
    print("  banking_detailed_trends.png")
    print("  banking_statistics_summary.png")
    print("  banking_historical_data_2011_2025.csv")
    print("  banking_predictions_2026_2030.csv")
    print("  banking_model_performance.csv")


def main() -> None:
    historical_df, predictions_df, metrics_df = run_trend_analysis()
    print_summary(historical_df, predictions_df, metrics_df)


if __name__ == "__main__":
    main()
