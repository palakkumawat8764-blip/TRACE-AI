"""Unit & headless Streamlit smoke tests for Milestone 6 Dashboard & Reports."""

from io import BytesIO
from pathlib import Path
import py_compile
import unittest

import pandas as pd
import plotly.graph_objects as go
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from src.analysis import analyze_incident
from src.config import PROJECT_ROOT, SAMPLE_DATA_DIR, load_config
from src.ingestion import ingest_files
from src.rag import build_manual_index
from src.reporting import generate_investigation_report
from src.ui import (
    build_metrics_anomaly_chart,
    build_root_cause_confidence_chart,
    build_timeline_scatter_chart,
    export_report_to_csv,
    export_report_to_pdf,
)


class TestUIModule(unittest.TestCase):
    """Verify Plotly charts, PDF/CSV report exports, and headless Streamlit app execution."""

    @classmethod
    def setUpClass(cls) -> None:
        files = sorted(Path(SAMPLE_DATA_DIR).iterdir())
        cls.ingestion = ingest_files(files)
        cls.analysis = analyze_incident(cls.ingestion, config=load_config())
        cls.rag = build_manual_index(cls.ingestion.manual_pages)
        cls.report = generate_investigation_report(cls.ingestion, cls.analysis, rag_index=cls.rag)

    def test_plotly_charts_on_sample_and_empty_inputs(self) -> None:
        fig_metric = build_metrics_anomaly_chart(self.analysis.metrics_scored_df, "latency_ms")
        fig_timeline = build_timeline_scatter_chart(self.analysis.timeline_df)
        fig_causes = build_root_cause_confidence_chart(self.report.ranked_root_causes)

        self.assertIsInstance(fig_metric, go.Figure)
        self.assertIsInstance(fig_timeline, go.Figure)
        self.assertIsInstance(fig_causes, go.Figure)
        self.assertGreaterEqual(len(fig_metric.data), 2)
        self.assertGreaterEqual(len(fig_timeline.data), 2)
        self.assertEqual(len(fig_causes.data), 1)

        # Empty input safety
        self.assertIsInstance(build_metrics_anomaly_chart(pd.DataFrame()), go.Figure)
        self.assertIsInstance(build_timeline_scatter_chart(pd.DataFrame()), go.Figure)
        self.assertIsInstance(build_root_cause_confidence_chart([]), go.Figure)

    def test_pdf_and_csv_report_exports(self) -> None:
        pdf_bytes = export_report_to_pdf(self.report)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))
        reader = PdfReader(BytesIO(pdf_bytes))
        self.assertGreaterEqual(len(reader.pages), 1)
        extracted_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        self.assertIn("CONFIRMED OBSERVATIONS", extracted_text)
        self.assertIn("RANKED POSSIBLE ROOT CAUSES", extracted_text)
        self.assertIn("technical_manual.pdf", extracted_text)

        csv_bytes = export_report_to_csv(self.report)
        csv_df = pd.read_csv(BytesIO(csv_bytes))
        sections = set(csv_df["section"].unique())
        self.assertTrue(
            {
                "Executive Summary",
                "Confirmed Observation",
                "Timeline Phase",
                "Root Cause Hypothesis",
                "RAG Manual Evidence",
                "Recommendation",
            }.issubset(sections)
        )

    def test_app_py_compiles_and_runs_headlessly(self) -> None:
        app_path = PROJECT_ROOT / "app.py"
        py_compile.compile(str(app_path), doraise=True)

        at = AppTest.from_file(str(app_path), default_timeout=20)
        at.run()

        self.assertEqual(len(at.exception), 0, msg=f"Streamlit exceptions: {at.exception}")
        self.assertGreaterEqual(len(at.tabs), 5)


if __name__ == "__main__":
    unittest.main()
