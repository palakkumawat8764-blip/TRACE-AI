"""Data models for normalized ingestion outputs across CSV, TXT/LOG, and PDF files."""

from dataclasses import dataclass, field
import pandas as pd

EVENT_COLUMNS = [
    "timestamp",
    "source_file",
    "source_type",
    "service",
    "severity",
    "message",
    "raw_line",
]


@dataclass(frozen=True)
class DocumentPage:
    """One extracted page from a PDF (or text document) for RAG indexing."""

    source_file: str
    page_number: int
    text: str


@dataclass
class IngestionResult:
    """Combined structured output produced by ingesting multiple incident files."""

    events_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EVENT_COLUMNS))
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    manual_pages: list[DocumentPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def timestamped_events_df(self) -> pd.DataFrame:
        """Return only events that have a valid parsed UTC timestamp."""
        if self.events_df.empty:
            return self.events_df
        return self.events_df.dropna(subset=["timestamp"]).reset_index(drop=True)
