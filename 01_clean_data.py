import os
from pathlib import Path

import pandas as pd

# 1. Create a virtual environment (called .venv)


def load_raw_data(input_path: str) -> pd.DataFrame:
    """
    Load the raw data using pandas.
    Supports CSV by default.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    col_strategies: dict | None = None,
) -> pd.DataFrame:
    """
    Handle missing values in a configurable way.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to clean.

    col_strategies : dict, optional
        A dictionary that specifies how to handle missing values for
        particular columns. Keys are column names, values are "strategy
        configs" with at least a `"strategy"` key.

        General format:
            col_strategies = {
                "<column_name>": {
                    "strategy": "<strategy_name>",
                    # optional extra keys depending on strategy
                },
                ...
            }

        Available strategies
        --------------------
        1. "drop_rows"
           - Description: Drop all rows where this column is NaN.
           - Config format:
                 {"strategy": "drop_rows"}
           - Example:
                 col_strategies = {
                     "some_numeric": {"strategy": "drop_rows"}
                 }

        2. "fill_with_value"
           - Description: Fill NaN in this column with a specific value.
           - Config format:
                 {"strategy": "fill_with_value", "value": <value>}
           - Example:
                 col_strategies = {
                     "country": {"strategy": "fill_with_value", "value": "Unknown"}
                 }

        3. "unknown_category"
           - Description: For categorical columns, treat missing values as a
             separate category called "Unknown".
           - Config format:
                 {"strategy": "unknown_category"}
           - Example:
                 col_strategies = {
                     "telescope_name": {"strategy": "unknown_category"}
                 }

        Default behavior (if a column is not in col_strategies)
        -------------------------------------------------------
        - Numeric columns (int, float): fill NaN with the column median.
        - Non-numeric columns (object, string, category): fill NaN with
          the column mode (most frequent value), if it exists.

        Usage examples
        --------------
        Example 1: Specific rules for two columns, defaults for others
            col_strategies = {
                "telescope_name": {"strategy": "unknown_category"},
                "exposure_value": {"strategy": "drop_rows"},
            }
            df = handle_missing_values(df, col_strategies)

        Example 2: Only use defaults (median/mode)
            df = handle_missing_values(df)

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing values handled according to the
        specified strategies and defaults.
    """
    if col_strategies is None:
        col_strategies = {}

    # 1) Column-specific strategies
    for col, cfg in col_strategies.items():
        if col not in df.columns:
            continue

        strat = cfg.get("strategy")

        if strat == "drop_rows":
            df = df[df[col].notna()]
        elif strat == "fill_with_value":
            value = cfg.get("value")
            df[col] = df[col].fillna(value)
        elif strat == "unknown_category":
            df[col] = df[col].fillna("Unknown")

    # 2) Default strategies for remaining missing values

    # Numeric: median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col in col_strategies:
            continue
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Non-numeric: mode
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    for col in non_numeric_cols:
        if col in col_strategies:
            continue
        mode_series = df[col].mode()
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    df = df.drop_duplicates()
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - Strip whitespace
    - Lowercase
    - Replace spaces with underscores
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def save_clean_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned data to data/cleaned/ as CSV (no index column).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    raw_data_path = "/Users/kaiwilliams/Documents/Documents/Projects/Messy-Data-ML/Data/cumulative.csv"
    cleaned_data_path = "/Users/kaiwilliams/Documents/Documents/Projects/Messy-Data-ML/Data/Cleaned/clean_data.csv"

    df = load_raw_data(raw_data_path)

    col_strategies = {
        "kepler_name": {"strategy": "unknown_category"},
        "koi_score": {"strategy": "drop_rows"},
    }

    df = handle_missing_values(df, col_strategies=col_strategies)
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    save_clean_data(df, cleaned_data_path)

    print(f"Cleaned data saved to: {cleaned_data_path}")


if __name__ == "__main__":
    main()
