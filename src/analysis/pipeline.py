"""End-to-end Milestone 3 incident analysis pipeline."""

from __future__ import annotations

from typing import Any

from src.analysis.anomaly_detector import detect_log_anomalies, detect_metric_anomalies
from src.analysis.correlator import correlate_anomalies, reconstruct_timeline
from src.analysis.models import AnalysisResult
from src.config import AppConfig, load_config
from src.ingestion.models import IngestionResult


def analyze_incident(
    ingestion: IngestionResult,
    config: AppConfig | None = None,
) -> AnalysisResult:
    """Run metric + log anomaly detection, cross-document correlation, and timeline reconstruction."""
    cfg = config or load_config()

    metrics_scored_df, metric_anomalies = detect_metric_anomalies(
        ingestion.metrics_df,
        contamination=cfg.anomaly_contamination,
    )
    log_anomalies = detect_log_anomalies(ingestion.events_df)

    all_anomalies = sorted(
        metric_anomalies + log_anomalies,
        key=lambda item: item.timestamp,
    )

    clusters = correlate_anomalies(
        all_anomalies,
        ingestion.events_df,
        window_seconds=cfg.correlation_window_seconds,
    )

    timeline_df, phases = reconstruct_timeline(
        ingestion.events_df,
        all_anomalies,
        clusters,
        window_seconds=cfg.correlation_window_seconds,
    )

    cross_doc_clusters = sum(1 for c in clusters if len(c.source_types) >= 2)
    peak_anomaly = max(all_anomalies, key=lambda item: item.anomaly_score) if all_anomalies else None

    summary_stats: dict[str, Any] = {
        "total_events": int(len(ingestion.events_df)),
        "timestamped_events": int(len(ingestion.timestamped_events_df)),
        "metric_anomalies": int(len(metric_anomalies)),
        "log_anomalies": int(len(log_anomalies)),
        "total_anomalies": int(len(all_anomalies)),
        "correlated_clusters": int(len(clusters)),
        "cross_document_clusters": int(cross_doc_clusters),
        "peak_timestamp": peak_anomaly.timestamp.isoformat() if peak_anomaly else None,
        "peak_severity": peak_anomaly.severity if peak_anomaly else "NONE",
    }

    return AnalysisResult(
        metrics_scored_df=metrics_scored_df,
        anomalies=all_anomalies,
        clusters=clusters,
        timeline_df=timeline_df,
        phases=phases,
        summary_stats=summary_stats,
    )
