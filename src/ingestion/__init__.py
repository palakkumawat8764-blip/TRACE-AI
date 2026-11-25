"""Multi-file ingestion package for CSV, TXT/LOG, and PDF sources."""

from src.ingestion.csv_parser import parse_csv_metrics
from src.ingestion.loader import SUPPORTED_EXTENSIONS, ingest_files
from src.ingestion.models import EVENT_COLUMNS, DocumentPage, IngestionResult
from src.ingestion.pdf_parser import parse_pdf_document
from src.ingestion.text_parser import parse_text_log

__all__ = [
    "DocumentPage",
    "EVENT_COLUMNS",
    "IngestionResult",
    "SUPPORTED_EXTENSIONS",
    "ingest_files",
    "parse_csv_metrics",
    "parse_pdf_document",
    "parse_text_log",
]
