"""Cross-document event correlation and automatic incident timeline reconstruction."""

from __future__ import annotations

import pandas as pd

from src.analysis.models import AnomalyFinding, CorrelatedCluster, TimelinePhase


def correlate_anomalies(
    anomalies: list[AnomalyFinding],
    events_df: pd.DataFrame,
    window_seconds: int = 180,
) -> list[CorrelatedCluster]:
    """Group nearby anomalies and pull correlated events across CSV, LOG, TXT, and PDF files."""
    if not anomalies:
        return []

    sorted_anomalies = sorted(anomalies, key=lambda item: item.timestamp)
    gap_limit = pd.Timedelta(seconds=max(1, int(window_seconds)))

    groups: list[list[AnomalyFinding]] = [[sorted_anomalies[0]]]
    for finding in sorted_anomalies[1:]:
        if finding.timestamp - groups[-1][-1].timestamp <= gap_limit:
            groups[-1].append(finding)
        else:
            groups.append([finding])

    timestamped_events = (
        events_df.dropna(subset=["timestamp"]).copy() if not events_df.empty else events_df.copy()
    )

    clusters: list[CorrelatedCluster] = []
    for cluster_idx, group in enumerate(groups, start=1):
        start_time = min(item.timestamp for item in group)
        end_time = max(item.timestamp for item in group)
        peak_finding = max(group, key=lambda item: item.anomaly_score)

        if timestamped_events.empty:
            related = timestamped_events.copy()
        else:
            mask = (timestamped_events["timestamp"] >= start_time - gap_limit) & (
                timestamped_events["timestamp"] <= end_time + gap_limit
            )
            related = timestamped_events.loc[mask].sort_values("timestamp").reset_index(drop=True)

        services = sorted(
            {
                svc.strip()
                for item in group
                for svc in item.service.split(",")
                if svc.strip() and svc.strip() != "unknown"
            }
            | (set(related["service"].astype(str).unique()) if not related.empty else set())
        )
        source_files = sorted(related["source_file"].astype(str).unique().tolist()) if not related.empty else sorted(
            {item.source_file for item in group}
        )
        source_types = sorted(related["source_type"].astype(str).unique().tolist()) if not related.empty else sorted(
            {item.source_type for item in group}
        )

        summary = (
            f"Cluster #{cluster_idx}: {len(group)} anomaly signal(s) peaking at "
            f"{peak_finding.timestamp.strftime('%H:%M:%S UTC')} across "
            f"{len(source_files)} file(s) ({', '.join(source_types)})"
        )

        clusters.append(
            CorrelatedCluster(
                cluster_id=cluster_idx,
                start_time=start_time,
                peak_time=peak_finding.timestamp,
                end_time=end_time,
                services=services,
                source_files=source_files,
                source_types=source_types,
                anomalies=group,
                related_events=related,
                summary=summary,
            )
        )

    return clusters


def _select_key_events(phase_df: pd.DataFrame, limit: int = 5) -> list[str]:
    if phase_df.empty:
        return []

    priority_order = {"CRITICAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "METRIC": 4, "DEBUG": 5}
    ranked = phase_df.copy()
    ranked["_prio"] = ranked["severity"].map(priority_order).fillna(6)
    # Prefer non-metric narrative/log lines alongside anomalous metrics
    ranked["_anom_boost"] = (~ranked["is_anomaly"]).astype(int)
    ranked = ranked.sort_values(by=["_anom_boost", "_prio", "timestamp"]).head(limit)
    ranked = ranked.sort_values(by="timestamp")

    formatted: list[str] = []
    for _, row in ranked.iterrows():
        ts_str = pd.Timestamp(row["timestamp"]).strftime("%H:%M:%S UTC")
        formatted.append(
            f"[{ts_str}] ({row['source_file']} | {row['severity']}) {row['service']}: {row['message']}"
        )
    return formatted


def reconstruct_timeline(
    events_df: pd.DataFrame,
    anomalies: list[AnomalyFinding],
    clusters: list[CorrelatedCluster],
    window_seconds: int = 180,
) -> tuple[pd.DataFrame, list[TimelinePhase]]:
    """Annotate the unified timeline with anomaly/cluster flags and segment it into phases."""
    if events_df.empty:
        return events_df.copy(), []

    timeline = events_df.copy()
    timeline["is_anomaly"] = False
    timeline["cluster_id"] = pd.Series([pd.NA] * len(timeline), dtype="Int64")

    if not anomalies:
        ts_only = timeline.dropna(subset=["timestamp"])
        if ts_only.empty:
            return timeline, []
        return timeline, [
            TimelinePhase(
                phase_name="Normal Operations (No Anomalies Detected)",
                start_time=pd.Timestamp(ts_only["timestamp"].min()),
                end_time=pd.Timestamp(ts_only["timestamp"].max()),
                event_count=len(ts_only),
                key_events=_select_key_events(ts_only),
            )
        ]

    # Mark events matching metric anomaly timestamps or ERROR/CRITICAL log bursts
    metric_anom_ts = {item.timestamp for item in anomalies if item.source_type == "csv"}
    timeline["is_anomaly"] = (
        ((timeline["source_type"] == "csv") & (timeline["timestamp"].isin(metric_anom_ts)))
        | ((timeline["source_type"] != "csv") & (timeline["severity"].isin({"ERROR", "CRITICAL"})))
    )

    gap = pd.Timedelta(seconds=max(1, int(window_seconds)))
    for cluster in clusters:
        in_cluster = (
            (~timeline["timestamp"].isna())
            & (timeline["timestamp"] >= cluster.start_time - gap)
            & (timeline["timestamp"] <= cluster.end_time + gap)
        )
        timeline.loc[in_cluster & timeline["cluster_id"].isna(), "cluster_id"] = cluster.cluster_id

    ts_df = timeline.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if ts_df.empty:
        return timeline, []

    first_anom = min(item.timestamp for item in anomalies)
    last_anom = max(item.timestamp for item in anomalies)
    peak_anom = max(anomalies, key=lambda item: item.anomaly_score).timestamp

    trigger_start = first_anom - gap
    peak_start = max(first_anom, peak_anom - pd.Timedelta(seconds=90))
    recovery_end = last_anom + gap

    phase_masks = [
        ("Pre-Incident Baseline", ts_df["timestamp"] < trigger_start),
        ("Trigger & Escalation", (ts_df["timestamp"] >= trigger_start) & (ts_df["timestamp"] < peak_start)),
        ("Peak Incident", (ts_df["timestamp"] >= peak_start) & (ts_df["timestamp"] <= last_anom)),
        ("Mitigation & Recovery", (ts_df["timestamp"] > last_anom) & (ts_df["timestamp"] <= recovery_end)),
        ("Post-Recovery Stability", ts_df["timestamp"] > recovery_end),
    ]

    phases: list[TimelinePhase] = []
    for name, mask in phase_masks:
        subset = ts_df.loc[mask]
        if subset.empty:
            continue
        phases.append(
            TimelinePhase(
                phase_name=name,
                start_time=pd.Timestamp(subset["timestamp"].min()),
                end_time=pd.Timestamp(subset["timestamp"].max()),
                event_count=len(subset),
                key_events=_select_key_events(subset),
            )
        )

    return timeline, phases
