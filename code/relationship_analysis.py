"""Analyze relationships between ESG scores and financial KPIs."""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from config import OUTPUT_DIR
from data_processing import ensure_output_dir, load_banking_dataset
from models import train_relationship_models
from relationship_plots import plot_detailed_relationship, plot_relationship_dashboard
from trend_analysis import run_trend_analysis

warnings.filterwarnings("ignore")


def _load_or_create_trend_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_path = OUTPUT_DIR / "banking_historical_data_2011_2025.csv"
    predictions_path = OUTPUT_DIR / "banking_predictions_2026_2030.csv"

    if not historical_path.exists() or not predictions_path.exists():
        historical_df, predictions_df, _ = run_trend_analysis()
        return historical_df, predictions_df

    return pd.read_csv(historical_path), pd.read_csv(predictions_path)


def _correlation_test(x, y) -> tuple[float, float]:
    return stats.pearsonr(x, y)


def _rolling_correlations(combined_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    rows = []
    for index in range(window, len(combined_df) + 1):
        window_data = combined_df.iloc[index - window:index]
        rows.append(
            {
                "Year": window_data["Year"].iloc[-1],
                "ESG_ROA_Correlation": window_data["ESG_Score"].corr(window_data["Pretax_ROA"]),
                "ESG_ROE_Correlation": window_data["ESG_Score"].corr(window_data["Pretax_ROE"]),
            }
        )

    return pd.DataFrame(rows)


def run_relationship_analysis() -> dict:
    """Run relationship analysis and write CSV/PNG outputs."""

    ensure_output_dir()
    historical_df, predictions_df = _load_or_create_trend_outputs()

    # Load panel dataset for relationship analysis to avoid data destruction
    dataset = load_banking_dataset()
    panel_df = dataset.panel

    future_df = pd.DataFrame(
        {
            "Year": predictions_df["Year"],
            "ESG_Score": predictions_df["ESG_Score_Linear_Regression"],
            "Pretax_ROA": predictions_df["Pretax_ROA_Linear_Regression"],
            "Pretax_ROE": predictions_df["Pretax_ROE_Linear_Regression"],
        }
    )
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)

    # Use panel data for computing correlation properly instead of aggregated data
    corr_esg_roa = panel_df["ESG_Score"].corr(panel_df["Pretax_ROA"])
    corr_esg_roe = panel_df["ESG_Score"].corr(panel_df["Pretax_ROE"])
    corr_roa_roe = panel_df["Pretax_ROA"].corr(panel_df["Pretax_ROE"])
    _, p_roa = _correlation_test(panel_df["ESG_Score"], panel_df["Pretax_ROA"])
    _, p_roe = _correlation_test(panel_df["ESG_Score"], panel_df["Pretax_ROE"])

    X_esg = panel_df["ESG_Score"].values.reshape(-1, 1)
    y_roa = panel_df["Pretax_ROA"].values
    y_roe = panel_df["Pretax_ROE"].values

    model_roa_linear = LinearRegression().fit(X_esg, y_roa)
    y_roa_pred = model_roa_linear.predict(X_esg)
    r2_roa_linear = r2_score(y_roa, y_roa_pred)

    model_roe_linear = LinearRegression().fit(X_esg, y_roe)
    y_roe_pred = model_roe_linear.predict(X_esg)
    r2_roe_linear = r2_score(y_roe, y_roe_pred)

    roa_models = train_relationship_models(panel_df["ESG_Score"], y_roa)
    roe_models = train_relationship_models(panel_df["ESG_Score"], y_roe)

    rolling_df = _rolling_correlations(historical_df)

    correlation_results = pd.DataFrame(
        {
            "Relationship": ["ESG vs ROA", "ESG vs ROE", "ROA vs ROE"],
            "Correlation": [corr_esg_roa, corr_esg_roe, corr_roa_roe],
            "P_Value": [p_roa, p_roe, np.nan],
            "Significant": [p_roa < 0.05, p_roe < 0.05, np.nan],
            "R_Squared": [r2_roa_linear, r2_roe_linear, np.nan],
            "Slope": [model_roa_linear.coef_[0], model_roe_linear.coef_[0], np.nan],
            "Intercept": [model_roa_linear.intercept_, model_roe_linear.intercept_, np.nan],
        }
    )

    correlation_results.to_csv(OUTPUT_DIR / "banking_correlation_results.csv", index=False)
    rolling_df.to_csv(OUTPUT_DIR / "banking_rolling_correlations.csv", index=False)

    context = {
        "historical_df": historical_df,
        "panel_df": panel_df,
        "predictions_df": predictions_df,
        "future_df": future_df,
        "combined_df": combined_df,
        "rolling_df": rolling_df,
        "corr_esg_roa": corr_esg_roa,
        "corr_esg_roe": corr_esg_roe,
        "corr_roa_roe": corr_roa_roe,
        "p_roa": p_roa,
        "p_roe": p_roe,
        "model_roa_linear": model_roa_linear,
        "model_roe_linear": model_roe_linear,
        "y_roa": y_roa,
        "y_roe": y_roe,
        "y_roa_pred": y_roa_pred,
        "y_roe_pred": y_roe_pred,
        "r2_roa_linear": r2_roa_linear,
        "r2_roe_linear": r2_roe_linear,
        "roa_models": roa_models,
        "roe_models": roe_models,
    }

    plot_relationship_dashboard(context)
    plot_detailed_relationship(context)
    return context


def print_summary(context: dict) -> None:
    print("=" * 80)
    print("BANKING SECTOR: ESG & FINANCIAL KPIs RELATIONSHIP ANALYSIS")
    print("=" * 80)
    print(f"\nESG vs ROA correlation: {context['corr_esg_roa']:.4f} (p={context['p_roa']:.4f})")
    print(f"ESG vs ROE correlation: {context['corr_esg_roe']:.4f} (p={context['p_roe']:.4f})")
    print(f"ROA vs ROE correlation: {context['corr_roa_roe']:.4f}")
    print("\nRegression equations:")
    print(f"  ROA = {context['model_roa_linear'].coef_[0]:.6f} x ESG + {context['model_roa_linear'].intercept_:.6f}")
    print(f"  ROE = {context['model_roe_linear'].coef_[0]:.6f} x ESG + {context['model_roe_linear'].intercept_:.6f}")
    print("\nFiles generated in outputs/:")
    print("  banking_esg_relationship_analysis.png")
    print("  banking_detailed_relationship.png")
    print("  banking_correlation_results.csv")
    print("  banking_rolling_correlations.csv")


def main() -> None:
    context = run_relationship_analysis()
    print_summary(context)


if __name__ == "__main__":
    main()
