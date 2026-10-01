"""
Data loader and schema profiling utilities for Conversational Data Analysis.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def load_csv(file_or_path, **kwargs) -> pd.DataFrame:
    """Load a CSV file from a file path or file-like object with automatic delimiter and date handling."""
    try:
        df = pd.read_csv(file_or_path, **kwargs)
    except UnicodeDecodeError:
        # Retry with latin-1 encoding if utf-8 fails
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, encoding="latin1", **kwargs)

    # Attempt to convert date-like object columns to datetime
    for col in df.columns:
        if df[col].dtype == "object":
            # Check if column name or sample suggests date
            col_lower = str(col).lower()
            if any(term in col_lower for term in ["date", "time", "timestamp", "year", "month"]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive metadata and statistical overview of the dataframe."""
    total_rows, total_cols = df.shape
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    missing_cells = int(df.isna().sum().sum())
    total_cells = total_rows * total_cols
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    col_details = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        sample_vals = df[col].dropna().head(3).tolist()

        col_meta = {
            "name": col,
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": round((null_count / total_rows * 100), 1) if total_rows > 0 else 0,
            "unique_count": n_unique,
            "sample_values": sample_vals,
        }
        col_details.append(col_meta)

    return {
        "rows": total_rows,
        "columns": total_cols,
        "memory_mb": memory_usage_mb,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "numeric_cols": numeric_cols,
        "datetime_cols": datetime_cols,
        "categorical_cols": categorical_cols,
        "columns_meta": col_details,
    }


def get_schema_prompt_context(df: pd.DataFrame, table_name: str = "dataset") -> str:
    """Format dataframe schema and representative data for injection into LLM prompts."""
    summary = get_dataset_summary(df)
    lines = [
        f"TABLE NAME: {table_name}",
        f"TOTAL ROWS: {summary['rows']}, TOTAL COLUMNS: {summary['columns']}",
        "COLUMNS SPECIFICATION:",
    ]

    for col in summary["columns_meta"]:
        name = col["name"]
        dtype = col["dtype"]
        nulls = col["null_count"]
        samples = col["sample_values"]
        
        # Check if numeric to add min/max
        if name in summary["numeric_cols"]:
            col_min = round(float(df[name].min()), 2) if not df[name].dropna().empty else None
            col_max = round(float(df[name].max()), 2) if not df[name].dropna().empty else None
            lines.append(
                f"- \"{name}\" ({dtype}) | Nulls: {nulls} | Range: [{col_min} to {col_max}] | Samples: {samples}"
            )
        else:
            lines.append(
                f"- \"{name}\" ({dtype}) | Nulls: {nulls} | Unique: {col['unique_count']} | Samples: {samples}"
            )

    return "\n".join(lines)
