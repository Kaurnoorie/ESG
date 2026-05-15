# Refactored ESG Analysis Code

This directory contains a smaller, modular replacement for the notebook-export style `book_1.py`.

## Files

- `config.py`: shared paths, column ranges, output locations, and constants.
- `data_processing.py`: Excel loading, FY label conversion, data cleanup, and historical aggregation.
- `models.py`: model training and metric helpers for forecasting and relationship models.
- `trend_plots.py`: visualizations for historical trends and 2026-2030 forecasts.
- `relationship_plots.py`: visualizations for ESG-to-financial KPI relationships.
- `trend_analysis.py`: runnable trend analysis script.
- `relationship_analysis.py`: runnable relationship analysis script.
- `explore_dataset.py`: lightweight dataset inspection script.
- `run_all.py`: runs both main analyses.

## Usage

Install `uv` if needed: https://docs.astral.sh/uv/getting-started/installation/

From the project root, use `uv run` to execute any script:

```powershell
uv run python code\trend_analysis.py
uv run python code\relationship_analysis.py
uv run python code\run_all.py
```

For a quick dataset inspection:

```powershell
uv run python code\explore_dataset.py
```

Generated CSV and image outputs are written to the existing `outputs/` directory.
