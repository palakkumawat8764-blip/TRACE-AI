"""End-to-end integration tests covering the complete TRACE AI pipeline and dashboard."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import unittest

import pandas as pd
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from src.analysis import analyze_incident
from src.config import PROJECT_ROOT, SAMPLE_DATA_DIR, load_config
from src.ingestion import ingest_files
from src.rag import build_manual_index
from src.reporting import generate_investigation_report
from src.ui import export_report_to_csv, export_report_to_pdf


class UploadedMemoryFile(BytesIO):
    """In-memory upload mock mimicking Streamlit UploadedFile."""

    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name


class TestEndToEndTRACEAI(unittest.TestCase):
    """Verify every stage from multi-file upload through report export & interactive UI."""

    def test_full_pipeline_via_uploaded_sample_files(self) -> None:
        uploads = [
            UploadedMemoryFile(p.name, p.read_bytes())
            for p in sorted(Path(SAMPLE_DATA_DIR).iterdir())
        ]
        cfg = load_config()

        # 1. Ingestion (CSV, LOG, TXT, PDF)
        ingestion = ingest_files(uploads)
        self.assertEqual(ingestion.warnings, [])
        self.assertEqual(len(ingestion.metrics_df), 30)
        self.assertEqual(len(ingestion.manual_pages), 3)
        self.assertEqual(len(ingestion.timestamped_events_df), 54)

        # 2 & 3. Anomaly detection, cross-document correlation & timeline reconstruction
        analysis = analyze_incident(ingestion, config=cfg)
        self.assertGreaterEqual(analysis.summary_stats["total_anomalies"], 10)
        self.assertGreaterEqual(analysis.summary_stats["cross_document_clusters"], 1)
        self.assertGreaterEqual(len(analysis.phases), 4)

        # 4. Local FAISS + scikit-learn RAG search
        rag = build_manual_index(ingestion.manual_pages, config=cfg)
        self.assertTrue(rag.is_ready)
        rag_hits = rag.search("ConnectionPoolExhausted 504 Gateway Timeout")
        self.assertGreaterEqual(len(rag_hits), 2)

        # 5. Evidence-based root cause report (mock mode)
        report = generate_investigation_report(ingestion, analysis, rag_index=rag, config=cfg)
        self.assertIn("mock", report.mode_used.lower())
        self.assertEqual(report.severity, "CRITICAL")
        self.assertGreaterEqual(len(report.confirmed_observations), 5)
        self.assertGreaterEqual(len(report.ranked_root_causes), 2)

        # 6. PDF & CSV downloads round-trip cleanly
        pdf_bytes = export_report_to_pdf(report)
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        self.assertGreaterEqual(len(pdf_reader.pages), 2)

        csv_df = pd.read_csv(BytesIO(export_report_to_csv(report)))
        self.assertIn("Root Cause Hypothesis", set(csv_df["section"]))
        self.assertIn("Confirmed Observation", set(csv_df["section"]))

    def test_alternative_cpu_oom_scenario_generalization(self) -> None:
        """Ensure pipeline generalizes beyond sample_data to a compute/OOM incident."""
        csv_data = (
            "timestamp,service,cpu_percent,memory_percent\n"
            "2026-11-21T10:00:00Z,worker-node,22.0,41.0\n"
            "2026-11-21T10:01:00Z,worker-node,23.5,42.0\n"
            "2026-11-21T10:02:00Z,worker-node,21.8,41.5\n"
            "2026-11-21T10:03:00Z,worker-node,24.0,43.0\n"
            "2026-11-21T10:04:00Z,worker-node,22.5,42.2\n"
            "2026-11-21T10:05:00Z,worker-node,96.4,98.5\n"
            "2026-11-21T10:06:00Z,worker-node,98.1,99.2\n"
            "2026-11-21T10:07:00Z,worker-node,25.0,44.0\n"
        ).encode("utf-8")
        log_data = (
            "2026-11-21 10:05:10 UTC [ERROR] worker-node: OOMKilled container due to memory pressure\n"
            "2026-11-21 10:06:02 UTC [CRITICAL] worker-node: Pod eviction triggered on high memory\n"
        ).encode("utf-8")

        ingestion = ingest_files(
            [
                UploadedMemoryFile("k8s_metrics.csv", csv_data),
                UploadedMemoryFile("k8s_events.log", log_data),
            ]
        )
        analysis = analyze_incident(ingestion)
        report = generate_investigation_report(ingestion, analysis)

        self.assertGreaterEqual(len(analysis.anomalies), 2)
        self.assertTrue(
            any("Compute (CPU) or Memory Saturation" in rc.title for rc in report.ranked_root_causes)
        )

    def test_gemini_mode_without_valid_key_falls_back_safely(self) -> None:
        cfg = replace(load_config(), llm_mode="gemini", gemini_api_key="invalid-test-key")
        ingestion = ingest_files(sorted(Path(SAMPLE_DATA_DIR).iterdir()))
        analysis = analyze_incident(ingestion, config=cfg)
        report = generate_investigation_report(ingestion, analysis, config=cfg)

        self.assertIn("fallback", report.mode_used.lower())
        self.assertGreaterEqual(len(report.ranked_root_causes), 1)

    def test_streamlit_dashboard_interactive_controls(self) -> None:
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Interact with the RAG search box and anomaly-only filter checkbox
        if at.text_input:
            at.text_input[0].set_value("504 Gateway Timeout")
        if at.checkbox:
            at.checkbox[0].set_value(True)
        at.run()

        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.tabs), 5)


if __name__ == "__main__":
    unittest.main()
