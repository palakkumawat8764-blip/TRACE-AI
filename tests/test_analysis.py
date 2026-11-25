"""Unit tests for Milestone 3 anomaly detection, correlation, and timeline engine."""

import unittest

from src.analysis import analyze_incident
from src.config import SAMPLE_DATA_DIR, load_config
from src.ingestion import IngestionResult, ingest_files


class TestAnalysisModule(unittest.TestCase):
    """Verify ML anomaly detection, log burst detection, correlation, and timeline phases."""

    def test_analyze_sample_incident_end_to_end(self) -> None:
        files = [
            SAMPLE_DATA_DIR / "server_metrics.csv",
            SAMPLE_DATA_DIR / "app_events.log",
            SAMPLE_DATA_DIR / "incident_summary_ticket.txt",
            SAMPLE_DATA_DIR / "technical_manual.pdf",
        ]
        ingestion = ingest_files(files)
        result = analyze_incident(ingestion, config=load_config())

        # Hybrid detector should flag the incident window (14:26-14:37) and exclude baseline/recovery
        metric_anoms = [a for a in result.anomalies if a.source_type == "csv"]
        self.assertGreaterEqual(len(metric_anoms), 3)
        for finding in metric_anoms:
            self.assertGreaterEqual(finding.timestamp.minute, 26)
            self.assertLessEqual(finding.timestamp.minute, 37)

        peak_metric = max(metric_anoms, key=lambda item: item.anomaly_score)
        self.assertGreaterEqual(peak_metric.timestamp.minute, 30)
        self.assertLessEqual(peak_metric.timestamp.minute, 34)

        # Log burst detector should flag ERROR/CRITICAL bursts from app_events.log
        log_anoms = [a for a in result.anomalies if a.source_type == "log"]
        self.assertGreaterEqual(len(log_anoms), 2)

        # Cross-document correlation should link csv, log, and txt together in one incident cluster
        self.assertGreaterEqual(len(result.clusters), 1)
        primary_cluster = result.clusters[0]
        self.assertTrue({"csv", "log", "txt"}.issubset(set(primary_cluster.source_types)))
        self.assertIn("api-gateway", primary_cluster.services)

        # Verify timeline DataFrame annotations and reconstructed phases
        self.assertEqual(len(result.timeline_df), len(ingestion.events_df))
        self.assertIn("is_anomaly", result.timeline_df.columns)
        self.assertIn("cluster_id", result.timeline_df.columns)
        self.assertTrue(bool(result.timeline_df["is_anomaly"].any()))

        phase_names = [p.phase_name for p in result.phases]
        self.assertIn("Pre-Incident Baseline", phase_names)
        self.assertIn("Peak Incident", phase_names)
        self.assertIn("Mitigation & Recovery", phase_names)

    def test_empty_ingestion_is_safe(self) -> None:
        empty = IngestionResult()
        result = analyze_incident(empty, config=load_config())

        self.assertEqual(result.anomalies, [])
        self.assertEqual(result.clusters, [])
        self.assertEqual(result.phases, [])
        self.assertEqual(result.summary_stats["total_anomalies"], 0)


if __name__ == "__main__":
    unittest.main()
