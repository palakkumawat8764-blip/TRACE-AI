"""Parser for PDF incident reports and technical manuals using pypdf."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from src.ingestion.models import EVENT_COLUMNS, DocumentPage
from src.ingestion.text_parser import TIMESTAMP_PATTERN, extract_line_metadata


def parse_pdf_document(
    source: Path | str | BytesIO | Any,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, list[DocumentPage], list[str]]:
    """Extract RAG pages and any dated events from a PDF file.

    Returns (events_df, manual_pages, warnings).
    """
    name = source_name or (Path(source).name if isinstance(source, (str, Path)) else getattr(source, "name", "uploaded.pdf"))
    warnings: list[str] = []
    pages: list[DocumentPage] = []
    event_records: list[dict[str, Any]] = []

    try:
        if isinstance(source, (str, Path)):
            reader = PdfReader(str(source))
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()
            reader = PdfReader(BytesIO(data) if isinstance(data, bytes) else source)
    except Exception as exc:
        warnings.append(f"{name}: failed to read PDF ({exc}).")
        return pd.DataFrame(columns=EVENT_COLUMNS), pages, warnings

    if not reader.pages:
        warnings.append(f"{name}: PDF contains 0 pages.")
        return pd.DataFrame(columns=EVENT_COLUMNS), pages, warnings

    default_service = Path(name).stem
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            warnings.append(f"{name} (page {page_idx}): no extractable text found.")
            continue

        pages.append(DocumentPage(source_file=name, page_number=page_idx, text=text))

        # Capture any timestamped lines (e.g., from exported PDF incident reports)
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or not TIMESTAMP_PATTERN.search(stripped):
                continue
            ts_value, severity, service, message = extract_line_metadata(
                stripped, default_service=default_service
            )
            if not pd.isna(ts_value):
                event_records.append(
                    {
                        "timestamp": ts_value,
                        "source_file": name,
                        "source_type": "pdf",
                        "service": service,
                        "severity": severity,
                        "message": f"[p.{page_idx}] {message}",
                        "raw_line": stripped,
                    }
                )

    events_df = pd.DataFrame(event_records, columns=EVENT_COLUMNS)
    if not events_df.empty:
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
    return events_df, pages, warnings
