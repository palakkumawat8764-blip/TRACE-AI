"""Parser for CSV metric/telemetry files into normalized events and numeric DataFrames."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.models import EVENT_COLUMNS

TIMESTAMP_CANDIDATES = ("timestamp", "time", "datetime", "date", "created_at", "@timestamp")
SERVICE_CANDIDATES = ("service", "host", "component", "node", "source")


def _detect_column(columns: pd.Index, preferred_names: tuple[str, ...], substring: str) -> str | None:
    lower_map = {str(col).strip().lower(): str(col) for col in columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]
    for lower_col, orig_col in lower_map.items():
        if substring in lower_col:
            return orig_col
    return None


def parse_csv_metrics(
    source: Path | str | BytesIO | Any,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Parse a CSV file into (events_df, metrics_df, warnings)."""
    name = source_name or (Path(source).name if isinstance(source, (str, Path)) else getattr(source, "name", "uploaded.csv"))
    warnings: list[str] = []

    if not isinstance(source, (str, Path)) and hasattr(source, "seek"):
        source.seek(0)

    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        warnings.append(f"{name}: CSV file is empty.")
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(), warnings
    except Exception as exc:
        warnings.append(f"{name}: failed to parse CSV ({exc}).")
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(), warnings

    if df.empty:
        warnings.append(f"{name}: CSV has headers but 0 rows.")
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(), warnings

    ts_col = _detect_column(df.columns, TIMESTAMP_CANDIDATES, "time")
    if ts_col is None:
        warnings.append(f"{name}: no timestamp column found; timeline placement unavailable.")
        df["timestamp"] = pd.NaT
    else:
        parsed_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        invalid_count = int(parsed_ts.isna().sum())
        if invalid_count > 0:
            warnings.append(f"{name}: {invalid_count} row(s) had unparseable timestamps.")
        df["timestamp"] = parsed_ts

    service_col = _detect_column(df.columns, SERVICE_CANDIDATES, "service")
    if service_col is not None and service_col != "service":
        df["service"] = df[service_col].astype(str)
    elif "service" not in df.columns:
        df["service"] = Path(name).stem

    df["source_file"] = name

    # Identify numeric metric columns for downstream ML anomaly detection
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in {"timestamp"}
    ]
    if not numeric_cols:
        warnings.append(f"{name}: no numeric metric columns detected.")

    metric_keep_cols = ["timestamp", "source_file", "service"] + numeric_cols
    metrics_df = df[metric_keep_cols].copy()

    # Build concise event rows so CSV telemetry appears alongside logs on the unified timeline
    event_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        metric_summary = ", ".join(f"{col}={row[col]}" for col in numeric_cols) if numeric_cols else "no numeric metrics"
        event_records.append(
            {
                "timestamp": row["timestamp"],
                "source_file": name,
                "source_type": "csv",
                "service": str(row["service"]),
                "severity": "METRIC",
                "message": metric_summary,
                "raw_line": metric_summary,
            }
        )

    events_df = pd.DataFrame(event_records, columns=EVENT_COLUMNS)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
    return events_df, metrics_df, warnings
