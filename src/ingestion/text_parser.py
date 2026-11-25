"""Parser for TXT and LOG incident files with timestamp, severity, and service extraction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.ingestion.models import EVENT_COLUMNS

# Matches ISO-8601 or "YYYY-MM-DD HH:MM:SS" (optional fractional seconds and UTC/offset)
TIMESTAMP_PATTERN = re.compile(
    r"(?P<ts>\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s*UTC|Z|[+-]\d{2}:?\d{2})?)"
)

SEVERITY_PATTERN = re.compile(
    r"\b(?P<sev>CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|INFO|DEBUG)\b",
    re.IGNORECASE,
)

# Matches common log prefixes such as: "[INFO] api-gateway: message" or "api-gateway: message"
SERVICE_PATTERN = re.compile(
    r"(?:\[(?:CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|INFO|DEBUG)\]\s*)?(?P<svc>[a-zA-Z0-9_.-]+):\s+",
    re.IGNORECASE,
)

SEVERITY_NORMALIZATION = {
    "FATAL": "CRITICAL",
    "CRITICAL": "CRITICAL",
    "ERR": "ERROR",
    "ERROR": "ERROR",
    "WARNING": "WARN",
    "WARN": "WARN",
    "INFO": "INFO",
    "DEBUG": "DEBUG",
}


def read_text_lines(source: Path | str | Any) -> list[str]:
    """Read lines from a filesystem path or an uploaded binary/text file-like object."""
    if isinstance(source, (str, Path)):
        return Path(source).read_text(encoding="utf-8", errors="replace").splitlines()

    if hasattr(source, "seek"):
        source.seek(0)
    raw = source.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    return text.splitlines()


def extract_line_metadata(
    line: str, default_service: str = "general"
) -> tuple[pd.Timestamp | pd.NaT, str, str, str]:
    """Extract (utc_timestamp, severity, service, cleaned_message) from a single text line."""
    ts_value: pd.Timestamp | pd.NaT = pd.NaT
    remaining = line.strip()

    ts_match = TIMESTAMP_PATTERN.search(remaining)
    if ts_match:
        raw_ts = ts_match.group("ts").strip()
        parsed = pd.to_datetime(raw_ts.replace("/", "-"), utc=True, errors="coerce")
        if not pd.isna(parsed):
            ts_value = parsed
            remaining = (remaining[: ts_match.start()] + remaining[ts_match.end() :]).strip()
            remaining = remaining.lstrip("-: ").strip()

    severity = "INFO"
    sev_match = SEVERITY_PATTERN.search(remaining)
    if sev_match:
        severity = SEVERITY_NORMALIZATION.get(sev_match.group("sev").upper(), "INFO")

    service = default_service
    svc_match = SERVICE_PATTERN.match(remaining)
    if svc_match:
        candidate = svc_match.group("svc")
        if candidate.upper() not in SEVERITY_NORMALIZATION:
            service = candidate
            remaining = remaining[svc_match.end() :].strip()
    elif sev_match:
        # Strip standalone "[INFO]" prefix even when no "service:" label follows
        cleaned = re.sub(
            r"^\[(?:CRITICAL|FATAL|ERROR|ERR|WARNING|WARN|INFO|DEBUG)\]\s*",
            "",
            remaining,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            remaining = cleaned

    return ts_value, severity, service, (remaining or line.strip())


def parse_text_log(
    source: Path | str | BytesIO | Any,
    source_name: str | None = None,
    source_type: str = "log",
) -> tuple[pd.DataFrame, list[str]]:
    """Parse a `.log` or `.txt` source into a normalized events DataFrame and warnings."""
    name = source_name or (Path(source).name if isinstance(source, (str, Path)) else getattr(source, "name", "uploaded.log"))
    default_service = Path(name).stem
    warnings: list[str] = []

    lines = read_text_lines(source)
    if not any(line.strip() for line in lines):
        warnings.append(f"{name}: file is empty.")
        return pd.DataFrame(columns=EVENT_COLUMNS), warnings

    records: list[dict[str, Any]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        ts_value, severity, service, message = extract_line_metadata(
            stripped, default_service=default_service
        )
        records.append(
            {
                "timestamp": ts_value,
                "source_file": name,
                "source_type": source_type,
                "service": service,
                "severity": severity,
                "message": message,
                "raw_line": stripped,
            }
        )

    events_df = pd.DataFrame(records, columns=EVENT_COLUMNS)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
    return events_df, warnings
