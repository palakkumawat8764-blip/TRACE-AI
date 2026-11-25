"""Anomaly detection for numeric CSV telemetry (Isolation Forest) and log error bursts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.analysis.models import AnomalyFinding


def _robust_z_scores(series: pd.Series) -> pd.Series:
    """Compute robust Z-scores using median/MAD, falling back to mean/std when MAD is zero."""
    clean = pd.to_numeric(series, errors="coerce").fillna(series.median())
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad > 1e-9:
        return 0.6745 * (clean - median) / mad

    std = float(clean.std(ddof=0))
    if std > 1e-9:
        return (clean - float(clean.mean())) / std
    return pd.Series(np.zeros(len(clean)), index=series.index)


def _score_to_severity(score: float) -> str:
    if score >= 0.15:
        return "CRITICAL"
    if score >= 0.06:
        return "HIGH"
    return "MEDIUM"


def detect_metric_anomalies(
    metrics_df: pd.DataFrame,
    contamination: float = 0.08,
) -> tuple[pd.DataFrame, list[AnomalyFinding]]:
    """Detect multivariate anomalies in numeric metrics using StandardScaler + IsolationForest.

    Returns (scored_metrics_df, list_of_AnomalyFinding).
    """
    if metrics_df.empty:
        return metrics_df.copy(), []

    scored_df = metrics_df.copy()
    numeric_cols = [
        col
        for col in scored_df.select_dtypes(include="number").columns
        if col not in {"is_anomaly", "anomaly_score"}
    ]

    if not numeric_cols or len(scored_df) < 5:
        scored_df["is_anomaly"] = False
        scored_df["anomaly_score"] = 0.0
        scored_df["top_driver"] = ""
        return scored_df, []

    feature_matrix = scored_df[numeric_cols].copy()
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    bounded_contamination = min(max(float(contamination), 0.01), 0.45)
    model = IsolationForest(
        contamination=bounded_contamination,
        random_state=42,
        n_estimators=150,
    )
    preds = model.fit_predict(scaled_features)
    # decision_function returns negative values for anomalies; negate so higher = more anomalous
    raw_scores = -model.decision_function(scaled_features)

    z_frame = pd.DataFrame(
        {col: _robust_z_scores(feature_matrix[col]) for col in numeric_cols},
        index=scored_df.index,
    )
    max_abs_z = z_frame.abs().max(axis=1)

    # Combine Isolation Forest (density isolation) with extreme robust Z-score guard
    # so sustained outage plateaus (multiple consecutive peak minutes) are also flagged.
    is_anomaly = (preds == -1) | (max_abs_z >= 6.0)
    combined_scores = np.maximum(raw_scores, 0.0) + 0.04 * np.log1p(max_abs_z)

    scored_df["is_anomaly"] = is_anomaly
    scored_df["anomaly_score"] = np.round(combined_scores, 4)

    top_drivers: list[str] = []
    findings: list[AnomalyFinding] = []

    for idx, row in scored_df.iterrows():
        row_z = z_frame.loc[idx].sort_values(key=lambda s: s.abs(), ascending=False)
        top_cols = [col for col in row_z.index[:3] if abs(float(row_z[col])) >= 1.0]
        if not top_cols:
            top_cols = list(row_z.index[:2])
        top_drivers.append(top_cols[0] if top_cols else "")

        if bool(row["is_anomaly"]) and not pd.isna(row.get("timestamp")):
            driver_parts = [
                f"{col}={row[col]} (z={row_z[col]:+.1f})"
                for col in top_cols
            ]
            score_val = float(row["anomaly_score"])
            findings.append(
                AnomalyFinding(
                    timestamp=pd.Timestamp(row["timestamp"]),
                    source_file=str(row.get("source_file", "metrics.csv")),
                    source_type="csv",
                    service=str(row.get("service", "unknown")),
                    anomaly_score=score_val,
                    severity=_score_to_severity(score_val),
                    summary="; ".join(driver_parts),
                    drivers=top_cols,
                )
            )

    scored_df["top_driver"] = top_drivers
    findings.sort(key=lambda item: item.timestamp)
    return scored_df, findings


def detect_log_anomalies(
    events_df: pd.DataFrame,
    bucket_freq: str = "1min",
) -> list[AnomalyFinding]:
    """Detect error/critical bursts across non-metric log and ticket events."""
    if events_df.empty:
        return []

    log_events = events_df[
        (events_df["source_type"] != "csv") & (~events_df["timestamp"].isna())
    ].copy()
    if log_events.empty:
        return []

    severity_weights = {"CRITICAL": 3, "ERROR": 2, "WARN": 1}
    log_events["weight"] = log_events["severity"].map(severity_weights).fillna(0).astype(int)
    log_events["is_err"] = log_events["severity"].isin({"ERROR", "CRITICAL"}).astype(int)
    log_events["bucket"] = log_events["timestamp"].dt.floor(bucket_freq)

    grouped = (
        log_events.groupby("bucket", sort=True)
        .agg(
            err_count=("is_err", "sum"),
            weight_sum=("weight", "sum"),
            total_count=("message", "count"),
        )
        .reset_index()
    )

    if grouped.empty:
        return []

    findings: list[AnomalyFinding] = []
    for _, b_row in grouped.iterrows():
        err_count = int(b_row["err_count"])
        weight_sum = int(b_row["weight_sum"])
        if err_count < 1 or weight_sum < 3:
            continue

        bucket_events = log_events[
            (log_events["bucket"] == b_row["bucket"])
            & (log_events["severity"].isin({"ERROR", "CRITICAL", "WARN"}))
        ]
        services = sorted(bucket_events["service"].astype(str).unique().tolist())
        files = sorted(bucket_events["source_file"].astype(str).unique().tolist())
        top_msgs = bucket_events["message"].astype(str).head(2).tolist()

        # Scale log burst score onto a comparable 0.05-0.30+ range
        burst_score = round(min(0.40, 0.04 * weight_sum), 4)
        has_critical = bool((bucket_events["severity"] == "CRITICAL").any())
        severity = "CRITICAL" if (has_critical or err_count >= 2) else "HIGH"

        findings.append(
            AnomalyFinding(
                timestamp=pd.Timestamp(b_row["bucket"]),
                source_file=", ".join(files),
                source_type="log",
                service=", ".join(services) if services else "unknown",
                anomaly_score=burst_score,
                severity=severity,
                summary=f"{err_count} error/critical log(s) in {bucket_freq} window: " + " | ".join(top_msgs),
                drivers=services,
            )
        )

    findings.sort(key=lambda item: item.timestamp)
    return findings
