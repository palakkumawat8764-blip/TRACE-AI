"""Data models for Milestone 3 anomaly detection, correlation, and timeline outputs."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AnomalyFinding:
    """A single metric or log-burst anomaly detected during investigation."""

    timestamp: pd.Timestamp
    source_file: str
    source_type: str
    service: str
    anomaly_score: float
    severity: str
    summary: str
    drivers: list[str] = field(default_factory=list)


@dataclass
class CorrelatedCluster:
    """Group of anomalies and surrounding cross-document events within a time window."""

    cluster_id: int
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: pd.Timestamp
    services: list[str]
    source_files: list[str]
    source_types: list[str]
    anomalies: list[AnomalyFinding]
    related_events: pd.DataFrame
    summary: str


@dataclass(frozen=True)
class TimelinePhase:
    """High-level incident phase reconstructed from correlated events."""

    phase_name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    event_count: int
    key_events: list[str]


@dataclass
class AnalysisResult:
    """Complete output of Milestone 3 analysis pipeline."""

    metrics_scored_df: pd.DataFrame
    anomalies: list[AnomalyFinding]
    clusters: list[CorrelatedCluster]
    timeline_df: pd.DataFrame
    phases: list[TimelinePhase]
    summary_stats: dict[str, Any]
