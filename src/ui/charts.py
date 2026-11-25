"""Interactive dark-themed Plotly chart builders for TRACE AI."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.config import DARK_THEME
from src.reporting.models import RootCauseCandidate

SEVERITY_COLORS = {
    "CRITICAL": DARK_THEME["accent_red"],
    "ERROR": "#FB7185",
    "WARN": DARK_THEME["accent_amber"],
    "INFO": DARK_THEME["accent_cyan"],
    "METRIC": "#64748B",
    "DEBUG": "#475569",
}


def _apply_dark_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=DARK_THEME["text_primary"], size=16)),
        paper_bgcolor=DARK_THEME["bg_secondary"],
        plot_bgcolor=DARK_THEME["bg_primary"],
        font=dict(color=DARK_THEME["text_primary"]),
        height=height,
        margin=dict(l=48, r=24, t=52, b=42),
        legend=dict(
            bgcolor="rgba(19,26,42,0.7)",
            bordercolor=DARK_THEME["border"],
            borderwidth=1,
        ),
        xaxis=dict(gridcolor=DARK_THEME["border"], zerolinecolor=DARK_THEME["border"]),
        yaxis=dict(gridcolor=DARK_THEME["border"], zerolinecolor=DARK_THEME["border"]),
    )
    return fig


def build_metrics_anomaly_chart(
    metrics_scored_df: pd.DataFrame,
    metric_column: str | None = None,
) -> go.Figure:
    """Plot a telemetry metric over time with detected anomalies highlighted."""
    fig = go.Figure()
    if metrics_scored_df.empty or "timestamp" not in metrics_scored_df.columns:
        return _apply_dark_layout(fig, "Telemetry Anomaly Detection (No Metric Data)")

    numeric_cols = [
        c
        for c in metrics_scored_df.select_dtypes(include="number").columns
        if c not in {"is_anomaly", "anomaly_score"}
    ]
    if not numeric_cols:
        return _apply_dark_layout(fig, "Telemetry Anomaly Detection (No Numeric Columns)")

    col = metric_column if metric_column in numeric_cols else numeric_cols[0]
    df = metrics_scored_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(color=DARK_THEME["accent_cyan"], width=2.4),
            marker=dict(size=5, color=DARK_THEME["accent_cyan"]),
        )
    )

    if "is_anomaly" in df.columns:
        anoms = df[df["is_anomaly"] == True]  # noqa: E712
        if not anoms.empty:
            drivers = anoms["top_driver"] if "top_driver" in anoms.columns else ""
            scores = anoms["anomaly_score"] if "anomaly_score" in anoms.columns else 0.0
            fig.add_trace(
                go.Scatter(
                    x=anoms["timestamp"],
                    y=anoms[col],
                    mode="markers",
                    name="Detected Anomaly",
                    marker=dict(
                        size=12,
                        color=DARK_THEME["accent_red"],
                        symbol="diamond",
                        line=dict(color="#FFFFFF", width=1.2),
                    ),
                    customdata=list(zip(scores, drivers)),
                    hovertemplate=(
                        "<b>Anomaly</b><br>Time: %{x}<br>"
                        f"{col}: %{{y}}<br>"
                        "Score: %{customdata[0]}<br>"
                        "Top driver: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

    return _apply_dark_layout(fig, f"Telemetry Trend & Anomalies — {col}")


def build_timeline_scatter_chart(timeline_df: pd.DataFrame) -> go.Figure:
    """Plot cross-document events along UTC time grouped by source_type and severity."""
    fig = go.Figure()
    if timeline_df.empty or "timestamp" not in timeline_df.columns:
        return _apply_dark_layout(fig, "Cross-Document Incident Timeline (No Events)")

    df = timeline_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return _apply_dark_layout(fig, "Cross-Document Incident Timeline (No Timestamped Events)")

    for sev, group in df.groupby("severity", sort=False):
        color = SEVERITY_COLORS.get(str(sev), DARK_THEME["text_muted"])
        sizes = [13 if bool(a) else 8 for a in group.get("is_anomaly", False)]
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["source_type"].astype(str).str.upper(),
                mode="markers",
                name=str(sev),
                marker=dict(color=color, size=sizes, opacity=0.9),
                customdata=list(
                    zip(
                        group["source_file"].astype(str),
                        group["service"].astype(str),
                        group["message"].astype(str).str.slice(0, 110),
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Time: %{x}<br>"
                    f"Severity: {sev}<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_yaxes(title_text="Source Type")
    return _apply_dark_layout(fig, "Cross-Document Event Correlation Timeline", height=360)


def build_root_cause_confidence_chart(
    ranked_root_causes: list[RootCauseCandidate],
) -> go.Figure:
    """Horizontal bar chart comparing ranked root-cause confidence percentages."""
    fig = go.Figure()
    if not ranked_root_causes:
        return _apply_dark_layout(fig, "Ranked Root Cause Hypotheses (None)", height=260)

    ordered = list(reversed(ranked_root_causes))
    labels = [f"#{rc.rank} {rc.title}" for rc in ordered]
    pcts = [round(rc.confidence_score * 100, 1) for rc in ordered]
    colors = [
        DARK_THEME["accent_red"]
        if rc.confidence == "HIGH"
        else DARK_THEME["accent_amber"]
        if rc.confidence == "MEDIUM"
        else "#64748B"
        for rc in ordered
    ]

    fig.add_trace(
        go.Bar(
            x=pcts,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.0f}% ({rc.confidence})" for p, rc in zip(pcts, ordered)],
            textposition="auto",
        )
    )
    fig.update_xaxes(range=[0, 105], title_text="Evidence Confidence (%)")
    return _apply_dark_layout(fig, "Root Cause Hypothesis Confidence Ranking", height=290)
