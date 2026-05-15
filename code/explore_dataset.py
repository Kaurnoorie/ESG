"""Small dataset inspection script replacing the first exploratory notebook cells."""

import pandas as pd

from config import DATA_FILE
from data_processing import load_raw_dataset


def main() -> None:
    df = pd.read_excel(DATA_FILE)
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())

    raw_df = load_raw_dataset()
    print("\nFirst 3 raw rows:")
    print(raw_df.iloc[:3, :10])
    print("\nRaw columns 20-30:")
    print(raw_df.iloc[:3, 20:30])
    print("\nRaw columns 35-45:")
    print(raw_df.iloc[:3, 35:45])


if __name__ == "__main__":
    main()
