"""Deterministic rule-based root cause ranking and evidence extraction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.models import AnalysisResult
from src.rag.models import RetrievedChunk
from src.reporting.models import EvidenceItem, RootCauseCandidate


@dataclass(frozen=True)
class _DiagnosticPattern:
    key: str
    title: str
    log_keywords: tuple[str, ...]
    metric_keywords: tuple[str, ...]
    manual_keywords: tuple[str, ...]
    rationale_template: str
    recommendations: tuple[str, ...]


DIAGNOSTIC_PATTERNS: tuple[_DiagnosticPattern, ...] = (
    _DiagnosticPattern(
        key="db_pool_exhaustion",
        title="Database Connection Pool Exhaustion",
        log_keywords=(
            "connectionpoolexhausted",
            "db_pool",
            "db-proxy",
            "idle-in-transaction",
            "waiting for db",
            "active connections",
            "orders_db",
        ),
        metric_keywords=("db_pool_active", "db_pool", "connections"),
        manual_keywords=("connection pool", "connectionpoolexhausted", "idle-in-transaction", "db-proxy"),
        rationale_template=(
            "Observed connection pool saturation and queued database requests coincide with the "
            "spike in downstream latency and errors. This suggests pool exhaustion is a primary "
            "mechanism behind the service degradation."
        ),
        recommendations=(
            "Enforce statement and idle-in-transaction timeouts on all database clients and workers.",
            "Identify and terminate or throttle queries holding idle-in-transaction connections during peak load.",
            "Verify active pool utilization drops below 30% of max capacity after remediation.",
        ),
    ),
    _DiagnosticPattern(
        key="batch_job_contention",
        title="Unthrottled Batch / Background Job Resource Contention",
        log_keywords=(
            "batch",
            "cron",
            "nightly_analytics_sync",
            "analytics-worker",
            "bulk read",
            "manually triggered",
            "without connection timeout",
        ),
        metric_keywords=("db_pool_active", "cpu_percent", "memory_percent"),
        manual_keywords=("batch", "nightly_analytics_sync", "read-replica", "01:00-04:00"),
        rationale_template=(
            "Logs and ticket notes record a heavy batch job running without timeout limits shortly "
            "before telemetry began degrading, making unthrottled batch execution a likely trigger."
        ),
        recommendations=(
            "Restrict heavy batch/analytics jobs to off-peak maintenance windows with SRE approval.",
            "Route analytics or bulk-read workloads to a dedicated read-replica with capped max_connections.",
            "Add mandatory query/cursor timeouts before re-enabling the batch job.",
        ),
    ),
    _DiagnosticPattern(
        key="gateway_timeout_cascade",
        title="Upstream Timeout & API Gateway Thread Saturation Cascade",
        log_keywords=(
            "504",
            "gateway timeout",
            "circuit breaker",
            "upstream",
            "timeout",
            "sla",
        ),
        metric_keywords=("latency_ms", "error_rate"),
        manual_keywords=("504 gateway timeout", "api-gateway", "circuit breaker", "rolling restart"),
        rationale_template=(
            "Sustained latency spikes and HTTP 504 errors indicate worker threads blocked waiting on "
            "upstream dependencies, pushing error rates past circuit-breaker thresholds."
        ),
        recommendations=(
            "Relieve upstream database or dependency contention before cycling gateway workers.",
            "Perform a rolling restart of blocked API gateway worker pools if threads remain wedged.",
            "Confirm latency returns below normal baseline (< 150 ms) and circuit breakers close.",
        ),
    ),
    _DiagnosticPattern(
        key="compute_memory_pressure",
        title="Compute (CPU) or Memory Saturation",
        log_keywords=("oom", "oomkilled", "memory", "cpu", "throttl", "evict"),
        metric_keywords=("cpu_percent", "memory_percent", "cpu", "memory"),
        manual_keywords=("cpu_percent", "memory_percent", "oomkilled", "memory pressure"),
        rationale_template=(
            "Telemetry shows elevated CPU or memory utilization during the anomaly window, which "
            "may reflect resource starvation or thread contention."
        ),
        recommendations=(
            "Inspect per-process CPU and heap/memory profiles during peak load.",
            "Verify container resource limits and horizontal autoscaling thresholds.",
        ),
    ),
)


def _format_ts(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "N/A"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def extract_confirmed_observations(
    analysis: AnalysisResult,
    max_items: int = 12,
) -> list[EvidenceItem]:
    """Extract strictly observed facts (metrics, errors, warnings) with timestamps & filenames."""
    observations: list[EvidenceItem] = []

    # 1. Include top metric anomalies (peak first, then earliest trigger)
    metric_anoms = [a for a in analysis.anomalies if a.source_type == "csv"]
    if metric_anoms:
        earliest_metric = min(metric_anoms, key=lambda a: a.timestamp)
        peak_metric = max(metric_anoms, key=lambda a: a.anomaly_score)
        for item, tag in (
            (earliest_metric, "First Metric Anomaly"),
            (peak_metric, "Peak Metric Anomaly"),
        ):
            if item is peak_metric and item.timestamp == earliest_metric.timestamp and tag == "Peak Metric Anomaly":
                continue
            observations.append(
                EvidenceItem(
                    timestamp=_format_ts(item.timestamp),
                    source_file=item.source_file,
                    category=tag,
                    detail=f"{item.service}: {item.summary} (severity={item.severity}, score={item.anomaly_score})",
                )
            )

    # 2. Include key timestamped non-CSV events (WARN, ERROR, CRITICAL, or trigger INFO in ticket/scheduler)
    if not analysis.timeline_df.empty:
        ts_df = analysis.timeline_df.dropna(subset=["timestamp"]).copy()
        non_csv = ts_df[ts_df["source_type"] != "csv"]
        important = non_csv[
            non_csv["severity"].isin({"CRITICAL", "ERROR", "WARN"})
            | non_csv["source_type"].isin({"txt", "pdf"})
            | non_csv["message"].str.contains("triggered|restarted|reclaimed|closed", case=False, na=False)
        ]
        for _, row in important.head(max_items - len(observations)).iterrows():
            observations.append(
                EvidenceItem(
                    timestamp=_format_ts(row["timestamp"]),
                    source_file=str(row["source_file"]),
                    category=f"Observed {row['severity']} Event",
                    detail=f"{row['service']}: {row['message']}",
                )
            )

    observations.sort(key=lambda ev: ev.timestamp)
    return observations[:max_items]


def rank_root_causes(
    analysis: AnalysisResult,
    manual_hits: list[RetrievedChunk],
) -> list[RootCauseCandidate]:
    """Score and rank possible root causes from correlated events, metrics, and RAG citations."""
    if not analysis.anomalies and analysis.timeline_df.empty:
        return []

    ts_df = (
        analysis.timeline_df.dropna(subset=["timestamp"]).copy()
        if not analysis.timeline_df.empty
        else pd.DataFrame()
    )
    peak_ts = pd.Timestamp(analysis.summary_stats["peak_timestamp"]) if analysis.summary_stats.get("peak_timestamp") else None
    metric_drivers = {
        driver.lower()
        for finding in analysis.anomalies
        for driver in finding.drivers
    }

    candidates: list[tuple[float, RootCauseCandidate]] = []

    for pattern in DIAGNOSTIC_PATTERNS:
        matched_evidence: list[EvidenceItem] = []
        matched_log_hits = 0

        if not ts_df.empty:
            for _, row in ts_df.iterrows():
                if row["source_type"] == "csv":
                    continue
                haystack = f"{row['service']} {row['message']}".lower()
                if any(kw in haystack for kw in pattern.log_keywords):
                    matched_log_hits += 1
                    if len(matched_evidence) < 4:
                        matched_evidence.append(
                            EvidenceItem(
                                timestamp=_format_ts(row["timestamp"]),
                                source_file=str(row["source_file"]),
                                category=f"Correlated {row['severity']}",
                                detail=f"{row['service']}: {row['message']}",
                            )
                        )

        matched_metric_drivers = [
            drv for drv in metric_drivers if any(mk in drv for mk in pattern.metric_keywords)
        ]
        if matched_metric_drivers:
            for finding in analysis.anomalies:
                if finding.source_type == "csv" and any(d in matched_metric_drivers for d in [x.lower() for x in finding.drivers]):
                    matched_evidence.append(
                        EvidenceItem(
                            timestamp=_format_ts(finding.timestamp),
                            source_file=finding.source_file,
                            category="Metric Anomaly",
                            detail=f"{finding.service}: {finding.summary}",
                        )
                    )
                    break

        matched_manual_hits = [
            hit for hit in manual_hits
            if any(mk in hit.chunk.text.lower() for mk in pattern.manual_keywords)
        ]
        for hit in matched_manual_hits[:2]:
            matched_evidence.append(
                EvidenceItem(
                    timestamp="Reference Manual",
                    source_file=hit.citation,
                    category="RAG Manual Match",
                    detail=hit.chunk.text.splitlines()[0][:140],
                )
            )

        if matched_log_hits == 0 and not matched_metric_drivers:
            continue

        # Score evidence signals into a bounded [0.15, 0.95] confidence score
        score = 0.0
        score += min(0.45, 0.14 * matched_log_hits)
        score += min(0.25, 0.12 * len(matched_metric_drivers))
        score += min(0.20, 0.15 * len(matched_manual_hits))

        # Temporal precedence bonus: signal observed before the incident peak
        if peak_ts is not None and not ts_df.empty:
            earliest_match = None
            for _, row in ts_df.iterrows():
                if row["source_type"] == "csv":
                    continue
                haystack = f"{row['service']} {row['message']}".lower()
                if any(kw in haystack for kw in pattern.log_keywords):
                    earliest_match = pd.Timestamp(row["timestamp"])
                    break
            if earliest_match is not None and earliest_match < peak_ts:
                score += 0.10

        confidence_score = round(min(0.95, max(0.15, score)), 2)
        if confidence_score < 0.25:
            continue

        candidates.append(
            (
                confidence_score,
                RootCauseCandidate(
                    rank=0,
                    title=pattern.title,
                    confidence=_confidence_label(confidence_score),
                    confidence_score=confidence_score,
                    rationale=pattern.rationale_template,
                    supporting_evidence=matched_evidence,
                    recommendations=list(pattern.recommendations),
                ),
            )
        )

    # Generic fallback if anomalies exist but none of the built-in signatures matched
    if not candidates and analysis.anomalies:
        top_anom = max(analysis.anomalies, key=lambda a: a.anomaly_score)
        candidates.append(
            (
                0.45,
                RootCauseCandidate(
                    rank=0,
                    title=f"Unclassified Anomaly Spike on {top_anom.service}",
                    confidence="MEDIUM",
                    confidence_score=0.45,
                    rationale=(
                        "Significant statistical/log deviations were observed without matching a "
                        "known runbook signature; manual inspection of correlated events is advised."
                    ),
                    supporting_evidence=[
                        EvidenceItem(
                            timestamp=_format_ts(top_anom.timestamp),
                            source_file=top_anom.source_file,
                            category="Peak Anomaly",
                            detail=top_anom.summary,
                        )
                    ],
                    recommendations=[
                        "Inspect correlated logs around the peak anomaly timestamp.",
                        "Compare affected metric drivers against service SLO thresholds.",
                    ],
                ),
            )
        )

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    ranked: list[RootCauseCandidate] = []
    for idx, (_, candidate) in enumerate(candidates, start=1):
        ranked.append(
            RootCauseCandidate(
                rank=idx,
                title=candidate.title,
                confidence=candidate.confidence,
                confidence_score=candidate.confidence_score,
                rationale=candidate.rationale,
                supporting_evidence=candidate.supporting_evidence,
                recommendations=candidate.recommendations,
            )
        )
    return ranked
