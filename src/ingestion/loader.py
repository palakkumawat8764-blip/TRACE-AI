"""Unified multi-file ingestion entry point for CSV, TXT/LOG, and PDF sources."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.csv_parser import parse_csv_metrics
from src.ingestion.models import EVENT_COLUMNS, DocumentPage, IngestionResult
from src.ingestion.pdf_parser import parse_pdf_document
from src.ingestion.text_parser import parse_text_log

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".log", ".pdf"}


def _resolve_name_and_source(item: Path | str | tuple[str, Any] | Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), item[1]
    if isinstance(item, (str, Path)):
        path = Path(item)
        return path.name, path
    name = getattr(item, "name", "unknown")
    return str(name), item


def ingest_files(sources: Iterable[Path | str | tuple[str, Any] | Any]) -> IngestionResult:
    """Ingest a collection of CSV, TXT, LOG, and PDF files into an IngestionResult."""
    event_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    manual_pages: list[DocumentPage] = []
    warnings: list[str] = []

    for item in sources:
        name, source = _resolve_name_and_source(item)
        ext = Path(name).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            warnings.append(f"{name}: unsupported file type '{ext or 'none'}' (skipped).")
            continue

        if isinstance(source, Path) and not source.exists():
            warnings.append(f"{name}: file not found at '{source}'.")
            continue

        if ext == ".csv":
            ev_df, met_df, file_warnings = parse_csv_metrics(source, source_name=name)
            if not ev_df.empty:
                event_frames.append(ev_df)
            if not met_df.empty:
                metric_frames.append(met_df)
            warnings.extend(file_warnings)

        elif ext in {".txt", ".log"}:
            ev_df, file_warnings = parse_text_log(
                source, source_name=name, source_type=ext.lstrip(".")
            )
            if not ev_df.empty:
                event_frames.append(ev_df)
            warnings.extend(file_warnings)

        elif ext == ".pdf":
            ev_df, pages, file_warnings = parse_pdf_document(source, source_name=name)
            if not ev_df.empty:
                event_frames.append(ev_df)
            manual_pages.extend(pages)
            warnings.extend(file_warnings)

    if event_frames:
        events_df = pd.concat(event_frames, ignore_index=True)
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
        events_df = events_df.sort_values(
            by="timestamp", ascending=True, na_position="last", kind="mergesort"
        ).reset_index(drop=True)
    else:
        events_df = pd.DataFrame(columns=EVENT_COLUMNS)

    if metric_frames:
        metrics_df = pd.concat(metric_frames, ignore_index=True)
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"], utc=True, errors="coerce")
        metrics_df = metrics_df.sort_values(
            by="timestamp", ascending=True, na_position="last", kind="mergesort"
        ).reset_index(drop=True)
    else:
        metrics_df = pd.DataFrame()

    return IngestionResult(
        events_df=events_df,
        metrics_df=metrics_df,
        manual_pages=manual_pages,
        warnings=warnings,
    )
