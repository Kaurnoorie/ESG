"""Data loading and preprocessing utilities for the ESG banking dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BASE_YEAR,
    COMPANY_COLUMNS,
    DATA_FILE,
    ESG_COLUMNS,
    OUTPUT_DIR,
    ROA_COLUMNS,
    ROE_COLUMNS,
)


@dataclass(frozen=True)
class BankingDataset:
    """Processed banking panel data used by the analysis scripts."""

    historical: pd.DataFrame
    panel: pd.DataFrame
    company_info: pd.DataFrame
    years: list[int]
    esg_data: np.ndarray
    roa_data: np.ndarray
    roe_data: np.ndarray


def ensure_output_dir() -> None:
    """Create the outputs directory used by the analysis scripts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataset(path=DATA_FILE) -> pd.DataFrame:
    """Load the Excel workbook without promoting rows to headers."""

    return pd.read_excel(path, header=None)


def fiscal_year_to_year(label, base_year: int = BASE_YEAR) -> int | None:
    """Convert FY labels such as FY0 and FY-1 into calendar years."""

    if pd.isna(label):
        return None

    label_text = str(label).strip().upper()
    if label_text == "FY0":
        return base_year

    if label_text.startswith("FY-"):
        try:
            return base_year - int(label_text.replace("FY-", ""))
        except ValueError:
            return None

    return None


def _chronological_year_indices(year_labels: list) -> tuple[list[int], list[int]]:
    years = [fiscal_year_to_year(label) for label in year_labels]
    valid_pairs = [(index, year) for index, year in enumerate(years) if year is not None]
    valid_pairs.sort(key=lambda item: item[1])
    return [index for index, _ in valid_pairs], [year for _, year in valid_pairs]


def _numeric_metric_matrix(data_rows: pd.DataFrame, column_slice: slice, indices: list[int]) -> np.ndarray:
    metric_data = data_rows.iloc[:, column_slice].apply(pd.to_numeric, errors="coerce").values
    return metric_data[:, indices]


def load_banking_dataset(path=DATA_FILE) -> BankingDataset:
    """Load, clean, and aggregate the banking ESG/ROA/ROE dataset."""

    df_raw = load_raw_dataset(path)
    year_labels = df_raw.iloc[1, ESG_COLUMNS].tolist()
    chronological_indices, years = _chronological_year_indices(year_labels)

    data_rows = df_raw.iloc[2:, :].reset_index(drop=True)
    company_info = data_rows.iloc[:, 0:5].copy()
    company_info.columns = COMPANY_COLUMNS

    esg_data = _numeric_metric_matrix(data_rows, ESG_COLUMNS, chronological_indices)
    roa_data = _numeric_metric_matrix(data_rows, ROA_COLUMNS, chronological_indices)
    roe_data = _numeric_metric_matrix(data_rows, ROE_COLUMNS, chronological_indices)

    historical = pd.DataFrame(
        {
            "Year": years,
            "ESG_Score": np.nanmean(esg_data, axis=0),
            "Pretax_ROA": np.nanmean(roa_data, axis=0) * 100,
            "Pretax_ROE": np.nanmean(roe_data, axis=0) * 100,
        }
    )

    panel_rows = []
    for i, company in company_info.iterrows():
        for j, year in enumerate(years):
            esg = esg_data[i, j]
            roa = roa_data[i, j]
            roe = roe_data[i, j]
            if not np.isnan(esg) and not np.isnan(roa) and not np.isnan(roe):
                row = company.to_dict()
                row["Year"] = year
                row["ESG_Score"] = esg
                row["Pretax_ROA"] = roa * 100
                row["Pretax_ROE"] = roe * 100
                panel_rows.append(row)
    
    panel = pd.DataFrame(panel_rows)

    return BankingDataset(
        historical=historical,
        panel=panel,
        company_info=company_info,
        years=years,
        esg_data=esg_data,
        roa_data=roa_data,
        roe_data=roe_data,
    )
