"""Unit tests for Milestone 2 multi-file ingestion (CSV, LOG, TXT, PDF)."""

from io import BytesIO
from pathlib import Path
import unittest

from src.config import SAMPLE_DATA_DIR
from src.ingestion import EVENT_COLUMNS, ingest_files


class InMemoryUpload(BytesIO):
    """Mimics Streamlit's UploadedFile with a .name attribute."""

    def __init__(self, name: str, data: bytes) -> None:
        super().__init__(data)
        self.name = name


class TestIngestionModule(unittest.TestCase):
    """Verify parsing, normalization, and edge-case handling."""

    def test_ingest_sample_data_bundle(self) -> None:
        files = [
            SAMPLE_DATA_DIR / "server_metrics.csv",
            SAMPLE_DATA_DIR / "app_events.log",
            SAMPLE_DATA_DIR / "incident_summary_ticket.txt",
            SAMPLE_DATA_DIR / "technical_manual.pdf",
        ]
        result = ingest_files(files)

        self.assertEqual(result.warnings, [])
        self.assertEqual(list(result.events_df.columns), EVENT_COLUMNS)

        # 30 CSV metric rows + 20 log lines + 8 non-empty ticket lines (4 timestamped + 4 header lines)
        self.assertEqual(len(result.events_df), 58)
        self.assertEqual(len(result.timestamped_events_df), 54)

        # Metrics DataFrame checks
        self.assertEqual(len(result.metrics_df), 30)
        for col in ("cpu_percent", "memory_percent", "db_pool_active", "latency_ms", "error_rate"):
            self.assertIn(col, result.metrics_df.columns)

        # Verify PDF manual pages were extracted
        self.assertEqual(len(result.manual_pages), 3)
        self.assertEqual([p.page_number for p in result.manual_pages], [1, 2, 3])
        self.assertIn("ConnectionPoolExhausted", result.manual_pages[0].text)
        self.assertIn("504 Gateway Timeout", result.manual_pages[1].text)
        self.assertIn("nightly_analytics_sync", result.manual_pages[2].text)

        # Verify chronological sorting of timestamped events
        ts = result.timestamped_events_df["timestamp"]
        self.assertTrue(ts.is_monotonic_increasing)

        # Verify severity parsing in logs
        severities = set(result.events_df["severity"].unique())
        self.assertTrue({"METRIC", "INFO", "WARN", "ERROR", "CRITICAL"}.issubset(severities))

    def test_ingest_in_memory_uploaded_files(self) -> None:
        csv_upload = InMemoryUpload(
            "metrics.csv",
            b"timestamp,service,cpu_percent\n2026-11-20T14:00:00Z,web,42.5\n",
        )
        log_upload = InMemoryUpload(
            "service.log",
            b"2026-11-20 14:01:00 UTC [ERROR] web: Connection refused\n",
        )
        pdf_bytes = (SAMPLE_DATA_DIR / "technical_manual.pdf").read_bytes()
        pdf_upload = InMemoryUpload("runbook.pdf", pdf_bytes)

        result = ingest_files([csv_upload, log_upload, pdf_upload])

        self.assertEqual(result.warnings, [])
        self.assertEqual(len(result.metrics_df), 1)
        self.assertEqual(len(result.timestamped_events_df), 2)
        self.assertEqual(len(result.manual_pages), 3)
        self.assertEqual(result.events_df.iloc[1]["severity"], "ERROR")
        self.assertEqual(result.events_df.iloc[1]["service"], "web")

    def test_edge_cases_warn_gracefully(self) -> None:
        empty_log = InMemoryUpload("empty.log", b"   \n\n")
        bad_ext = InMemoryUpload("image.png", b"PNGDATA")
        no_ts_csv = InMemoryUpload("no_ts.csv", b"service,cpu\napi,25\n")
        missing_path = Path("does_not_exist.csv")

        result = ingest_files([empty_log, bad_ext, no_ts_csv, missing_path])

        self.assertEqual(len(result.warnings), 4)
        self.assertEqual(len(result.metrics_df), 1)
        self.assertTrue(result.timestamped_events_df.empty)


if __name__ == "__main__":
    unittest.main()
