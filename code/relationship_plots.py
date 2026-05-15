"""Visualizations for ESG-to-financial KPI relationship analysis."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def _normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 100


def plot_relationship_dashboard(context: dict) -> None:
    """Create the 12-panel relationship dashboard from the final notebook section."""

    historical_df = context["historical_df"]
    combined_df = context["combined_df"]
    future_df = context["future_df"]
    rolling_df = context["rolling_df"]
    future_esg = context["future_esg"]
    roa_models = context["roa_models"]
    roe_models = context["roe_models"]
    model_roa = context["model_roa_linear"]
    model_roe = context["model_roe_linear"]

    corr_esg_roa = context["corr_esg_roa"]
    corr_esg_roe = context["corr_esg_roe"]
    p_roa = context["p_roa"]
    p_roe = context["p_roe"]
    r2_roa = context["r2_roa_linear"]
    r2_roe = context["r2_roe_linear"]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(22, 16))

    esg_range = np.linspace(historical_df["ESG_Score"].min() - 2, historical_df["ESG_Score"].max() + 2, 100).reshape(-1, 1)

    ax1 = plt.subplot(3, 4, 1)
    scatter1 = ax1.scatter(historical_df["ESG_Score"], historical_df["Pretax_ROA"], c=historical_df["Year"], cmap="viridis", s=120, alpha=0.8, edgecolors="black", linewidth=1.5)
    ax1.plot(esg_range, model_roa.predict(esg_range), "r--", linewidth=2.5, label=f"Linear Fit (R2={r2_roa:.3f})")
    ax1.scatter(future_df["ESG_Score"], roa_models["XGBoost"]["model"].predict(future_esg), c="red", s=150, marker="s", edgecolors="black", linewidth=2, label="Predicted (2026-2030)")
    ax1.set_xlabel("ESG Score", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Pretax ROA (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"ESG vs ROA: Correlation = {corr_esg_roa:.3f}", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Year")

    ax2 = plt.subplot(3, 4, 2)
    scatter2 = ax2.scatter(historical_df["ESG_Score"], historical_df["Pretax_ROE"], c=historical_df["Year"], cmap="plasma", s=120, alpha=0.8, edgecolors="black", linewidth=1.5)
    ax2.plot(esg_range, model_roe.predict(esg_range), "r--", linewidth=2.5, label=f"Linear Fit (R2={r2_roe:.3f})")
    ax2.scatter(future_df["ESG_Score"], roe_models["XGBoost"]["model"].predict(future_esg), c="red", s=150, marker="s", edgecolors="black", linewidth=2, label="Predicted (2026-2030)")
    ax2.set_xlabel("ESG Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pretax ROE (%)", fontsize=11, fontweight="bold")
    ax2.set_title(f"ESG vs ROE: Correlation = {corr_esg_roe:.3f}", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Year")

    ax3 = plt.subplot(3, 4, 3)
    corr_matrix = historical_df[["ESG_Score", "Pretax_ROA", "Pretax_ROE"]].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax3)
    ax3.set_title("Correlation Matrix (2011-2025)", fontsize=12, fontweight="bold")
    ax3.set_xticklabels(["ESG", "ROA", "ROE"], rotation=0)
    ax3.set_yticklabels(["ESG", "ROA", "ROE"], rotation=0)

    ax4 = plt.subplot(3, 4, 4)
    ax4.axis("off")
    summary_text = f"""
RELATIONSHIP SUMMARY

HISTORICAL (2011-2025):
ESG vs ROA: r = {corr_esg_roa:7.3f}
  P-value:    {p_roa:7.4f}
  Significant: {'Yes' if p_roa < 0.05 else 'No'}

ESG vs ROE: r = {corr_esg_roe:7.3f}
  P-value:    {p_roe:7.4f}
  Significant: {'Yes' if p_roe < 0.05 else 'No'}

REGRESSION EQUATIONS:
ROA = {model_roa.coef_[0]:.4f} x ESG + {model_roa.intercept_:.2f}
ROE = {model_roe.coef_[0]:.4f} x ESG + {model_roe.intercept_:.2f}
"""
    ax4.text(0.05, 0.95, summary_text, fontsize=10, family="monospace", va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax5 = plt.subplot(3, 4, 5)
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(combined_df["Year"], combined_df["ESG_Score"], "o-", linewidth=2.5, markersize=8, color="#9B59B6", label="ESG Score")
    line2 = ax5_twin.plot(combined_df["Year"], combined_df["Pretax_ROA"], "s-", linewidth=2.5, markersize=8, color="#E67E22", label="Pretax ROA (%)")
    ax5.axvline(x=2025, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax5.axvspan(2025, 2030, alpha=0.1, color="orange")
    ax5.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax5.set_ylabel("ESG Score", fontsize=11, fontweight="bold", color="#9B59B6")
    ax5_twin.set_ylabel("Pretax ROA (%)", fontsize=11, fontweight="bold", color="#E67E22")
    ax5.set_title("ESG Score & ROA Timeline (2011-2030)", fontsize=12, fontweight="bold")
    ax5.legend(line1 + line2, [line.get_label() for line in line1 + line2], loc="upper left", fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 4, 6)
    ax6_twin = ax6.twinx()
    line3 = ax6.plot(combined_df["Year"], combined_df["ESG_Score"], "o-", linewidth=2.5, markersize=8, color="#9B59B6", label="ESG Score")
    line4 = ax6_twin.plot(combined_df["Year"], combined_df["Pretax_ROE"], "^-", linewidth=2.5, markersize=8, color="#1ABC9C", label="Pretax ROE (%)")
    ax6.axvline(x=2025, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax6.axvspan(2025, 2030, alpha=0.1, color="orange")
    ax6.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax6.set_ylabel("ESG Score", fontsize=11, fontweight="bold", color="#9B59B6")
    ax6_twin.set_ylabel("Pretax ROE (%)", fontsize=11, fontweight="bold", color="#1ABC9C")
    ax6.set_title("ESG Score & ROE Timeline (2011-2030)", fontsize=12, fontweight="bold")
    ax6.legend(line3 + line4, [line.get_label() for line in line3 + line4], loc="upper left", fontsize=9)
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(rolling_df["Year"], rolling_df["ESG_ROA_Correlation"], "o-", linewidth=2.5, markersize=9, label="ESG-ROA", color="#E67E22")
    ax7.plot(rolling_df["Year"], rolling_df["ESG_ROE_Correlation"], "s-", linewidth=2.5, markersize=9, label="ESG-ROE", color="#1ABC9C")
    ax7.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
    ax7.axvline(x=2025, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax7.axhline(y=corr_esg_roa, color="#E67E22", linestyle=":", linewidth=2, alpha=0.5, label=f"Overall ESG-ROA ({corr_esg_roa:.3f})")
    ax7.axhline(y=corr_esg_roe, color="#1ABC9C", linestyle=":", linewidth=2, alpha=0.5, label=f"Overall ESG-ROE ({corr_esg_roe:.3f})")
    ax7.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax7.set_ylabel("5-Year Rolling Correlation", fontsize=11, fontweight="bold")
    ax7.set_title("Correlation Evolution Over Time", fontsize=12, fontweight="bold")
    ax7.legend(loc="best", fontsize=8)
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 4, 8)
    models = ["XGBoost", "Random Forest", "Decision Tree"]
    x = np.arange(len(models))
    width = 0.35
    ax8.bar(x - width / 2, [roa_models[name]["r2"] for name in models], width, label="ESG->ROA", color="#E67E22", alpha=0.8)
    ax8.bar(x + width / 2, [roe_models[name]["r2"] for name in models], width, label="ESG->ROE", color="#1ABC9C", alpha=0.8)
    ax8.set_ylabel("R2 Score", fontsize=11, fontweight="bold")
    ax8.set_title("Relationship Model Accuracy", fontsize=12, fontweight="bold")
    ax8.set_xticks(x)
    ax8.set_xticklabels(models, fontsize=10)
    ax8.legend(loc="upper right", fontsize=10)
    ax8.grid(True, alpha=0.3, axis="y")

    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(combined_df["Year"], _normalize(combined_df["ESG_Score"]), "o-", linewidth=2.5, markersize=8, label="ESG Score", color="#9B59B6")
    ax9.plot(combined_df["Year"], _normalize(combined_df["Pretax_ROA"]), "s-", linewidth=2.5, markersize=8, label="Pretax ROA", color="#E67E22")
    ax9.plot(combined_df["Year"], _normalize(combined_df["Pretax_ROE"]), "^-", linewidth=2.5, markersize=8, label="Pretax ROE", color="#1ABC9C")
    ax9.axvline(x=2025, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax9.axvspan(2025, 2030, alpha=0.1, color="orange")
    ax9.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax9.set_ylabel("Normalized Value (0-100)", fontsize=11, fontweight="bold")
    ax9.set_title("Normalized Trends Comparison", fontsize=12, fontweight="bold")
    ax9.legend(loc="best", fontsize=10)
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(3, 4, 10)
    years = historical_df["Year"].iloc[1:]
    ax10.plot(years, historical_df["ESG_Score"].pct_change().iloc[1:] * 100, "o-", linewidth=2, markersize=7, label="ESG Score", color="#9B59B6")
    ax10.plot(years, historical_df["Pretax_ROA"].pct_change().iloc[1:] * 100, "s-", linewidth=2, markersize=7, label="Pretax ROA", color="#E67E22")
    ax10.plot(years, historical_df["Pretax_ROE"].pct_change().iloc[1:] * 100, "^-", linewidth=2, markersize=7, label="Pretax ROE", color="#1ABC9C")
    ax10.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax10.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax10.set_ylabel("Year-over-Year Change (%)", fontsize=11, fontweight="bold")
    ax10.set_title("Annual Growth Rates (2011-2025)", fontsize=12, fontweight="bold")
    ax10.legend(loc="best", fontsize=9)
    ax10.grid(True, alpha=0.3)

    ax11 = plt.subplot(3, 4, 11)
    roa_std = historical_df["Pretax_ROA"].std()
    future_roa_xgb = roa_models["XGBoost"]["model"].predict(future_esg)
    ax11.fill_between(future_df["Year"], future_roa_xgb - roa_std, future_roa_xgb + roa_std, alpha=0.3, color="#E67E22", label="Confidence Band")
    ax11.plot(future_df["Year"], future_roa_xgb, "o-", linewidth=3, markersize=10, color="#E67E22", label="XGBoost Prediction")
    ax11.plot(historical_df["Year"].iloc[-3:], historical_df["Pretax_ROA"].iloc[-3:], "s--", linewidth=2, markersize=8, color="gray", alpha=0.7, label="Recent Historical")
    ax11.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax11.set_ylabel("Pretax ROA (%)", fontsize=11, fontweight="bold")
    ax11.set_title("Future ROA Prediction (2026-2030)", fontsize=12, fontweight="bold")
    ax11.legend(loc="best", fontsize=9)
    ax11.grid(True, alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    ax12.axis("off")
    insights = f"""
KEY INSIGHTS

1. HISTORICAL RELATIONSHIP:
   ESG-ROA: {corr_esg_roa:.3f}
   ESG-ROE: {corr_esg_roe:.3f}

2. TRADE-OFF:
   1pt ESG changes ROA by {model_roa.coef_[0]:.3f}%
   1pt ESG changes ROE by {model_roe.coef_[0]:.3f}%

3. FUTURE OUTLOOK:
   ESG stabilizes near {future_df['ESG_Score'].mean():.2f}
   ROA stabilizes near {future_roa_xgb.mean():.2f}%
"""
    ax12.text(0.05, 0.95, insights, fontsize=10, family="monospace", va="top", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4))

    plt.suptitle("Banking Sector: ESG & Financial KPIs Comprehensive Relationship Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_DIR / "banking_esg_relationship_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_detailed_relationship(context: dict) -> None:
    """Create the smaller residual and correlation comparison figure."""

    historical_df = context["historical_df"]
    combined_corr_roa = context["combined_corr_roa"]
    combined_corr_roe = context["combined_corr_roe"]
    corr_esg_roa = context["corr_esg_roa"]
    corr_esg_roe = context["corr_esg_roe"]
    y_roa = context["y_roa"]
    y_roe = context["y_roe"]
    y_roa_pred = context["y_roa_pred"]
    y_roe_pred = context["y_roe_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.scatter(y_roa_pred, y_roa - y_roa_pred, c=historical_df["Year"], cmap="viridis", s=120, alpha=0.8, edgecolors="black")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted ROA (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Residuals", fontsize=11, fontweight="bold")
    ax.set_title("Residual Analysis: ESG -> ROA Prediction", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_roe_pred, y_roe - y_roe_pred, c=historical_df["Year"], cmap="plasma", s=120, alpha=0.8, edgecolors="black")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted ROE (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Residuals", fontsize=11, fontweight="bold")
    ax.set_title("Residual Analysis: ESG -> ROE Prediction", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    categories = ["ESG vs\nROA", "ESG vs\nROE"]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, [corr_esg_roa, corr_esg_roe], width, label="Historical (2011-25)", color="#3498DB", alpha=0.8)
    ax.bar(x + width / 2, [combined_corr_roa, combined_corr_roe], width, label="With Future (2011-30)", color="#E74C3C", alpha=0.8)
    ax.set_ylabel("Correlation Coefficient", fontsize=11, fontweight="bold")
    ax.set_title("Correlation Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        "RELATIONSHIP EVOLUTION\n\n"
        "Historical period shows the observed ESG-profitability relationship.\n"
        "Forecast years stabilize ESG, ROA, and ROE around a new equilibrium.\n\n"
        "Use the generated CSV outputs for exact coefficients and predictions.",
        fontsize=11,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    plt.suptitle("Banking Sector: Detailed ESG-Financial Relationship Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_DIR / "banking_detailed_relationship.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
