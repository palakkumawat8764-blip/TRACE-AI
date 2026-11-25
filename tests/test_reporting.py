"""Unit tests for Milestone 5 evidence-based root cause analysis & reporting."""

import unittest

from src.analysis import analyze_incident
from src.config import SAMPLE_DATA_DIR, load_config
from src.ingestion import IngestionResult, ingest_files
from src.rag import build_manual_index
from src.reporting import generate_investigation_report


class TestReportingModule(unittest.TestCase):
    """Verify grounded root cause ranking, RAG evidence, timestamps, and report formatting."""

    @classmethod
    def setUpClass(cls) -> None:
        files = [
            SAMPLE_DATA_DIR / "server_metrics.csv",
            SAMPLE_DATA_DIR / "app_events.log",
            SAMPLE_DATA_DIR / "incident_summary_ticket.txt",
            SAMPLE_DATA_DIR / "technical_manual.pdf",
        ]
        cls.config = load_config()
        cls.ingestion = ingest_files(files)
        cls.analysis = analyze_incident(cls.ingestion, config=cls.config)
        cls.rag_index = build_manual_index(cls.ingestion.manual_pages, config=cls.config)
        cls.report = generate_investigation_report(
            cls.ingestion,
            cls.analysis,
            rag_index=cls.rag_index,
            config=cls.config,
        )

    def test_report_runs_in_local_mock_mode(self) -> None:
        self.assertIn("mock", self.report.mode_used.lower())
        self.assertEqual(self.report.severity, "CRITICAL")

    def test_confirmed_observations_include_timestamps_and_filenames(self) -> None:
        self.assertGreaterEqual(len(self.report.confirmed_observations), 4)
        sources = {ev.source_file for ev in self.report.confirmed_observations}
        self.assertTrue({"server_metrics.csv", "app_events.log"}.issubset(sources))
        for ev in self.report.confirmed_observations:
            self.assertNotEqual(ev.timestamp, "")
            self.assertIn("2026-11-20", ev.timestamp)

    def test_uses_reconstructed_timeline_and_rag_evidence(self) -> None:
        self.assertGreaterEqual(len(self.report.timeline_highlights), 3)
        phase_names = [p["phase"] for p in self.report.timeline_highlights]
        self.assertIn("Peak Incident", phase_names)

        self.assertGreaterEqual(len(self.report.manual_evidence), 2)
        citations = {hit.citation for hit in self.report.manual_evidence}
        self.assertIn("technical_manual.pdf (p. 1)", citations)

    def test_ranked_root_causes_confidence_and_recommendations(self) -> None:
        self.assertGreaterEqual(len(self.report.ranked_root_causes), 2)
        ranks = [rc.rank for rc in self.report.ranked_root_causes]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

        scores = [rc.confidence_score for rc in self.report.ranked_root_causes]
        self.assertEqual(scores, sorted(scores, reverse=True))

        top_titles = " | ".join(rc.title for rc in self.report.ranked_root_causes[:3])
        self.assertIn("Database Connection Pool Exhaustion", top_titles)

        for rc in self.report.ranked_root_causes:
            self.assertIn(rc.confidence, {"HIGH", "MEDIUM", "LOW"})
            self.assertGreater(len(rc.supporting_evidence), 0)
            self.assertGreater(len(rc.recommendations), 0)

        self.assertGreaterEqual(len(self.report.recommendations), 3)

    def test_distinguishes_confirmed_observations_from_possible_causes(self) -> None:
        md = self.report.markdown_report
        self.assertIn("Confirmed Observations (Directly Observed Facts)", md)
        self.assertIn("Ranked Possible Root Causes (Hypotheses", md)
        self.assertIn("Technical Manual Evidence (FAISS RAG Citations)", md)

        payload = self.report.to_dict()
        self.assertIn("confirmed_observations", payload)
        self.assertIn("ranked_root_causes", payload)

    def test_empty_incident_generates_safe_report(self) -> None:
        empty_ing = IngestionResult()
        empty_ana = analyze_incident(empty_ing, config=self.config)
        empty_report = generate_investigation_report(empty_ing, empty_ana, config=self.config)
        self.assertEqual(empty_report.severity, "NONE")
        self.assertEqual(empty_report.ranked_root_causes, [])
        self.assertIn("No Anomalies Detected", empty_report.incident_title)


if __name__ == "__main__":
    unittest.main()
