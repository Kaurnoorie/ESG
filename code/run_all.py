"""Run the full refactored ESG analysis pipeline."""

from relationship_analysis import run_relationship_analysis, print_summary as print_relationship_summary
from trend_analysis import run_trend_analysis, print_summary as print_trend_summary


def main() -> None:
    historical_df, predictions_df, metrics_df = run_trend_analysis()
    print_trend_summary(historical_df, predictions_df, metrics_df)

    relationship_context = run_relationship_analysis()
    print_relationship_summary(relationship_context)


if __name__ == "__main__":
    main()
