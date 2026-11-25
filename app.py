"""TRACE AI — Streamlit Dark-Themed Incident Investigation & Root Cause Dashboard."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import streamlit as st

from src.analysis import analyze_incident
from src.config import SAMPLE_DATA_DIR, load_config
from src.ingestion import ingest_files
from src.rag import build_manual_index
from src.reporting import generate_investigation_report
from src.ui import (
    build_metrics_anomaly_chart,
    build_root_cause_confidence_chart,
    build_timeline_scatter_chart,
    export_report_to_csv,
    export_report_to_pdf,
    get_dark_dashboard_css,
)

st.set_page_config(
    page_title="TRACE AI — Incident Investigation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_dark_dashboard_css(), unsafe_allow_html=True)


def _render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="trace-kpi">
            <div class="trace-kpi-label">{label}</div>
            <div class="trace-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    base_cfg = load_config()

    with st.sidebar:
        st.markdown("## 🔍 TRACE AI Controls")
        data_source = st.radio(
            "Incident Data Source",
            options=["Built-in Sample Incident", "Upload Incident Files"],
            index=0,
        )

        uploaded_files = []
        if data_source == "Upload Incident Files":
            uploaded_files = st.file_uploader(
                "Upload CSV, LOG, TXT, and PDF files",
                type=["csv", "log", "txt", "pdf"],
                accept_multiple_files=True,
            )

        st.divider()
        st.markdown("### Detection & RAG Parameters")
        contamination = st.slider(
            "Anomaly Sensitivity (Contamination)",
            min_value=0.02,
            max_value=0.30,
            value=float(base_cfg.anomaly_contamination),
            step=0.02,
        )
        corr_window = st.slider(
            "Correlation Window (seconds)",
            min_value=30,
            max_value=600,
            value=int(base_cfg.correlation_window_seconds),
            step=30,
        )
        rag_top_k = st.slider(
            "RAG Manual Citations (Top-K)",
            min_value=1,
            max_value=6,
            value=int(base_cfg.rag_top_k),
            step=1,
        )

        st.divider()
        st.markdown("### LLM Reasoning Mode")
        llm_mode = st.selectbox(
            "Provider Mode",
            options=["mock", "gemini"],
            index=0 if base_cfg.llm_mode == "mock" else 1,
            help="Runs free & offline in 'mock' mode; switch to 'gemini' with an API key.",
        )
        gemini_key = base_cfg.gemini_api_key
        if llm_mode == "gemini":
            gemini_key = st.text_input(
                "Gemini API Key (optional)",
                value=base_cfg.gemini_api_key,
                type="password",
            )

    runtime_cfg = replace(
        base_cfg,
        llm_mode=llm_mode,
        gemini_api_key=gemini_key,
        anomaly_contamination=contamination,
        correlation_window_seconds=corr_window,
        rag_top_k=rag_top_k,
    )

    # Resolve input sources
    if data_source == "Built-in Sample Incident":
        sources = sorted(Path(SAMPLE_DATA_DIR).iterdir()) if SAMPLE_DATA_DIR.exists() else []
    else:
        sources = list(uploaded_files)

    if not sources:
        st.markdown(
            """
            <div class="trace-hero">
                <h1>TRACE AI — Incident Investigation & Root Cause Detection</h1>
                <p>Upload CSV telemetry, application logs, incident tickets, and PDF technical manuals in the sidebar to start.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Select **Built-in Sample Incident** or upload one or more `.csv`, `.log`, `.txt`, or `.pdf` files in the sidebar.")
        return

    ingestion = ingest_files(sources)
    analysis = analyze_incident(ingestion, config=runtime_cfg)
    rag_index = build_manual_index(ingestion.manual_pages, config=runtime_cfg)
    report = generate_investigation_report(
        ingestion=ingestion,
        analysis=analysis,
        rag_index=rag_index,
        config=runtime_cfg,
    )

    sev_class = "trace-badge-crit" if report.severity in {"CRITICAL", "HIGH"} else "trace-badge-ok"
    st.markdown(
        f"""
        <div class="trace-hero">
            <h1>TRACE AI — Incident Investigation & Root Cause Detection</h1>
            <p>
                <span class="trace-badge">{report.mode_used}</span>
                &nbsp;
                <span class="trace-badge {sev_class}">Peak Severity: {report.severity}</span>
                &nbsp;
                <span class="trace-badge">RAG Index: {rag_index.chunk_count} chunk(s) ({rag_index.vector_mode})</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ingestion.warnings:
        with st.expander(f"Ingestion Warnings ({len(ingestion.warnings)})", expanded=False):
            for warn in ingestion.warnings:
                st.warning(warn)

    # Top KPI row
    stats = analysis.summary_stats
    top_conf = (
        f"{int(report.ranked_root_causes[0].confidence_score * 100)}% ({report.ranked_root_causes[0].confidence})"
        if report.ranked_root_causes
        else "N/A"
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _render_kpi_card("Total Events", str(stats.get("total_events", 0)))
    with k2:
        _render_kpi_card("Anomalies Flagged", str(stats.get("total_anomalies", 0)))
    with k3:
        _render_kpi_card("Cross-Doc Clusters", str(stats.get("cross_document_clusters", 0)))
    with k4:
        _render_kpi_card("Manual Chunks (FAISS)", str(rag_index.chunk_count))
    with k5:
        _render_kpi_card("Lead Hypothesis", top_conf)

    st.write("")
    tab_overview, tab_anomalies, tab_timeline, tab_rag, tab_report = st.tabs(
        [
            "Overview & Root Cause",
            "Anomalies & Metrics",
            "Timeline & Correlation",
            "Technical Manual RAG",
            "Investigation Report & Export",
        ]
    )

    with tab_overview:
        st.subheader("Executive Summary")
        st.info(report.executive_summary)

        col_left, col_right = st.columns([1.15, 1.0])
        with col_left:
            st.subheader("Confirmed Observations (Directly Observed Facts)")
            st.caption("Timestamped facts directly recorded in metrics, logs, and tickets.")
            if report.confirmed_observations:
                obs_rows = [
                    {
                        "Timestamp": ev.timestamp,
                        "Source File": ev.source_file,
                        "Category": ev.category,
                        "Observation": ev.detail,
                    }
                    for ev in report.confirmed_observations
                ]
                st.dataframe(obs_rows, width="stretch", hide_index=True)
            else:
                st.write("No anomalous observations recorded.")

        with col_right:
            st.subheader("Ranked Possible Root Causes (Hypotheses)")
            st.plotly_chart(
                build_root_cause_confidence_chart(report.ranked_root_causes),
                width="stretch",
            )

        if report.ranked_root_causes:
            for rc in report.ranked_root_causes:
                with st.expander(
                    f"Rank #{rc.rank}: {rc.title} — {rc.confidence} ({int(rc.confidence_score * 100)}%)",
                    expanded=(rc.rank == 1),
                ):
                    st.markdown(f"**Hypothesis Rationale:** {rc.rationale}")
                    st.markdown("**Supporting Evidence:**")
                    for ev in rc.supporting_evidence:
                        st.markdown(f"- `{ev.timestamp}` | `{ev.source_file}` (*{ev.category}*) — {ev.detail}")
                    st.markdown("**Recommended Actions:**")
                    for rec in rc.recommendations:
                        st.markdown(f"- {rec}")

    with tab_anomalies:
        st.subheader("Telemetry Anomaly Detection")
        if not analysis.metrics_scored_df.empty:
            numeric_cols = [
                c
                for c in analysis.metrics_scored_df.select_dtypes(include="number").columns
                if c not in {"is_anomaly", "anomaly_score"}
            ]
            default_idx = numeric_cols.index("latency_ms") if "latency_ms" in numeric_cols else 0
            selected_metric = st.selectbox(
                "Select Metric to Inspect",
                options=numeric_cols,
                index=default_idx,
            )
            st.plotly_chart(
                build_metrics_anomaly_chart(analysis.metrics_scored_df, metric_column=selected_metric),
                width="stretch",
            )
        else:
            st.info("No CSV numeric telemetry uploaded.")

        st.subheader("All Detected Metric & Log Anomalies")
        if analysis.anomalies:
            anom_rows = [
                {
                    "Timestamp (UTC)": a.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Type": a.source_type.upper(),
                    "Severity": a.severity,
                    "Score": a.anomaly_score,
                    "Service": a.service,
                    "Source File": a.source_file,
                    "Summary": a.summary,
                }
                for a in analysis.anomalies
            ]
            st.dataframe(anom_rows, width="stretch", hide_index=True)
        else:
            st.success("No anomalies exceeded thresholds.")

    with tab_timeline:
        st.subheader("Cross-Document Correlation & Timeline")
        st.plotly_chart(
            build_timeline_scatter_chart(analysis.timeline_df),
            width="stretch",
        )

        if analysis.clusters:
            st.markdown("### Correlated Incident Clusters")
            for cl in analysis.clusters:
                st.success(
                    f"**{cl.summary}** — Services: `{', '.join(cl.services)}` | "
                    f"Files: `{', '.join(cl.source_files)}`"
                )

        if analysis.phases:
            st.markdown("### Reconstructed Incident Phases")
            for phase in analysis.phases:
                with st.expander(
                    f"{phase.phase_name} ({phase.start_time.strftime('%H:%M:%S')} → "
                    f"{phase.end_time.strftime('%H:%M:%S UTC')} | {phase.event_count} events)",
                    expanded=(phase.phase_name in {"Trigger & Escalation", "Peak Incident"}),
                ):
                    for key_ev in phase.key_events:
                        st.markdown(f"- {key_ev}")

        st.markdown("### Unified Event Log")
        if not analysis.timeline_df.empty:
            f1, f2 = st.columns(2)
            with f1:
                sev_options = sorted(analysis.timeline_df["severity"].astype(str).unique().tolist())
                sev_filter = st.multiselect("Filter by Severity", options=sev_options, default=sev_options)
            with f2:
                only_anom = st.checkbox("Show Flagged Anomaly Events Only", value=False)

            view_df = analysis.timeline_df[analysis.timeline_df["severity"].isin(sev_filter)]
            if only_anom:
                view_df = view_df[view_df["is_anomaly"] == True]  # noqa: E712
            display_cols = ["timestamp", "source_file", "source_type", "service", "severity", "is_anomaly", "message"]
            st.dataframe(view_df[display_cols], width="stretch", hide_index=True)

    with tab_rag:
        st.subheader("Technical Manual Search (FAISS + Local TF-IDF/LSA)")
        if not rag_index.is_ready:
            st.info("Upload one or more `.pdf` technical manuals to enable RAG search.")
        else:
            query_text = st.text_input(
                "Search indexed manuals",
                placeholder="e.g., ConnectionPoolExhausted 504 Gateway Timeout batch job",
            )
            if query_text.strip():
                manual_hits = rag_index.search(query_text.strip(), top_k=rag_top_k)
                st.markdown("#### Interactive Query Results")
            else:
                manual_hits = report.manual_evidence
                st.markdown("#### Auto-Retrieved Evidence for This Incident")

            if manual_hits:
                for hit in manual_hits:
                    with st.expander(
                        f"Rank #{hit.rank} — {hit.citation} (Similarity: {hit.score:.4f})",
                        expanded=True,
                    ):
                        st.text(hit.chunk.text)
            else:
                st.warning("No matching manual passages found for that query.")

    with tab_report:
        st.subheader("Evidence-Based Investigation Report")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                label="Download PDF Report (.pdf)",
                data=export_report_to_pdf(report),
                file_name="trace_ai_investigation_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
        with d2:
            st.download_button(
                label="Download CSV Report (.csv)",
                data=export_report_to_csv(report),
                file_name="trace_ai_investigation_report.csv",
                mime="text/csv",
                width="stretch",
            )
        with d3:
            st.download_button(
                label="Download Markdown (.md)",
                data=report.markdown_report,
                file_name="trace_ai_investigation_report.md",
                mime="text/markdown",
                width="stretch",
            )
        with d4:
            st.download_button(
                label="Download JSON (.json)",
                data=json.dumps(report.to_dict(), indent=2),
                file_name="trace_ai_investigation_report.json",
                mime="application/json",
                width="stretch",
            )

        st.divider()
        st.markdown(report.markdown_report)


if __name__ == "__main__":
    main()
