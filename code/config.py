"""Shared configuration for the banking ESG analyses."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Banking (ESG, RoE, RoA)-Final Clean Data.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"

BASE_YEAR = 2025
FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

COMPANY_COLUMNS = ["RIC", "Company_Name", "Country_HQ", "Country_Inc", "Industry"]

ESG_COLUMNS = slice(5, 20)
ROA_COLUMNS = slice(20, 35)
ROE_COLUMNS = slice(35, 50)

METRICS = [
    ("ESG_Score", "ESG Score"),
    ("Pretax_ROA", "Pretax ROA (%)"),
    ("Pretax_ROE", "Pretax ROE (%)"),
]
