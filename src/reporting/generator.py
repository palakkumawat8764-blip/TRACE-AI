"""Milestone 5 end-to-end evidence-based investigation report generator."""

from __future__ import annotations

from typing import Any

from src.analysis.models import AnalysisResult
from src.config import AppConfig, load_config
from src.ingestion.models import IngestionResult
from src.rag.models import RetrievedChunk
from src.rag.retriever import ManualRAGIndex, build_manual_index
from src.reporting.analyzer import extract_confirmed_observations, rank_root_causes
from src.reporting.llm_client import build_grounded_prompt, maybe_refine_summary_with_llm
from src.reporting.models import EvidenceItem, InvestigationReport, RootCauseCandidate


def _build_timeline_highlights(analysis: AnalysisResult) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for phase in analysis.phases:
        highlights.append(
            {
                "phase": phase.phase_name,
                "start": phase.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "end": phase.end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "event_count": phase.event_count,
                "key_events": list(phase.key_events),
            }
        )
    return highlights


def _collect_recommendations(
    ranked_causes: list[RootCauseCandidate],
    manual_hits: list[RetrievedChunk],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for rc in ranked_causes:
        for rec in rc.recommendations:
            if rec not in seen:
                seen.add(rec)
                ordered.append(rec)

    if manual_hits and not ordered:
        ordered.append(f"Review troubleshooting procedures in {manual_hits[0].citation}.")
    return ordered


def _render_markdown(
    incident_title: str,
    mode_used: str,
    severity: str,
    executive_summary: str,
    confirmed_observations: list[EvidenceItem],
    timeline_highlights: list[dict[str, Any]],
    ranked_root_causes: list[RootCauseCandidate],
    manual_hits: list[RetrievedChunk],
    recommendations: list[str],
) -> str:
    lines: list[str] = [
        f"# {incident_title}",
        f"**Analysis Mode:** `{mode_used}` | **Peak Severity:** `{severity}`",
        "",
        "## 1. Executive Summary",
        executive_summary,
        "",
        "## 2. Confirmed Observations (Directly Observed Facts)",
        "> These statements reflect direct measurements, log lines, and ticket records with exact timestamps and source files.",
        "",
    ]

    if confirmed_observations:
        for ev in confirmed_observations:
            lines.append(f"- **`{ev.timestamp}`** | `{ev.source_file}` | *{ev.category}*: {ev.detail}")
    else:
        lines.append("- No anomalous observations were detected in the supplied files.")

    lines.extend(["", "## 3. Reconstructed Incident Timeline"])
    if timeline_highlights:
        for phase in timeline_highlights:
            lines.append(
                f"### {phase['phase']} (`{phase['start']}` -> `{phase['end']}`, {phase['event_count']} events)"
            )
            for key_ev in phase["key_events"]:
                lines.append(f"- {key_ev}")
            lines.append("")
    else:
        lines.append("- No timestamped timeline events available.")

    lines.extend([
        "",
        "## 4. Ranked Possible Root Causes (Hypotheses — Not Confirmed Fact)",
        "> These candidates are inferred from correlated signals and runbook patterns; verify before applying permanent changes.",
        "",
    ])

    if ranked_root_causes:
        for rc in ranked_root_causes:
            lines.append(
                f"### Rank #{rc.rank}: {rc.title} — Confidence: **{rc.confidence}** ({int(rc.confidence_score * 100)}%)"
            )
            lines.append(f"- **Assessment:** {rc.rationale}")
            if rc.supporting_evidence:
                lines.append("- **Supporting Evidence:**")
                for ev in rc.supporting_evidence:
                    lines.append(f"  - `{ev.timestamp}` | `{ev.source_file}` — {ev.detail}")
            if rc.recommendations:
                lines.append("- **Targeted Actions:**")
                for rec in rc.recommendations:
                    lines.append(f"  - {rec}")
            lines.append("")
    else:
        lines.append("- No root cause hypotheses identified.")

    lines.extend(["## 5. Technical Manual Evidence (FAISS RAG Citations)"])
    if manual_hits:
        for hit in manual_hits:
            excerpt = hit.chunk.text.replace("\n", " ")[:280]
            lines.append(f"- **[{hit.rank}] `{hit.citation}`** (similarity `{hit.score:.4f}`): {excerpt}...")
    else:
        lines.append("- No technical manual PDF pages were available or matched.")

    lines.extend(["", "## 6. Prioritized Troubleshooting Recommendations"])
    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            lines.append(f"{idx}. {rec}")
    else:
        lines.append("1. Continue monitoring baseline metrics and logs.")

    return "\n".join(lines).strip() + "\n"


def generate_investigation_report(
    ingestion: IngestionResult,
    analysis: AnalysisResult,
    rag_index: ManualRAGIndex | None = None,
    config: AppConfig | None = None,
) -> InvestigationReport:
    """Generate a complete evidence-grounded root cause investigation report."""
    cfg = config or load_config()
    active_rag = rag_index or build_manual_index(ingestion.manual_pages, config=cfg)

    manual_hits = (
        active_rag.search_for_incident(analysis, top_k=cfg.rag_top_k)
        if active_rag.is_ready
        else []
    )
    confirmed_observations = extract_confirmed_observations(analysis)
    ranked_root_causes = rank_root_causes(analysis, manual_hits)
    timeline_highlights = _build_timeline_highlights(analysis)
    recommendations = _collect_recommendations(ranked_root_causes, manual_hits)

    stats = analysis.summary_stats
    severity = str(stats.get("peak_severity", "NONE"))
    peak_ts = stats.get("peak_timestamp") or "N/A"
    cluster_count = stats.get("correlated_clusters", 0)
    cross_doc_count = stats.get("cross_document_clusters", 0)

    if not analysis.anomalies:
        incident_title = "TRACE AI Investigation Report — No Anomalies Detected"
        base_summary = (
            f"Analyzed {stats.get('total_events', 0)} ingested event(s). "
            "No metric or log-burst anomalies exceeded detection thresholds."
        )
    else:
        lead_cause = ranked_root_causes[0].title if ranked_root_causes else "Undetermined"
        lead_conf = ranked_root_causes[0].confidence if ranked_root_causes else "LOW"
        services = sorted({svc for c in analysis.clusters for svc in c.services})
        svc_label = ", ".join(services[:4]) if services else "monitored services"
        incident_title = f"TRACE AI Incident Investigation Report — {svc_label}"
        base_summary = (
            f"Detected {stats.get('total_anomalies', 0)} anomaly signal(s) "
            f"({stats.get('metric_anomalies', 0)} metric, {stats.get('log_anomalies', 0)} log) "
            f"across {cluster_count} correlated cluster(s) ({cross_doc_count} cross-document), "
            f"peaking at {peak_ts} ({severity}). "
            f"Top ranked hypothesis: **{lead_cause}** (Confidence: **{lead_conf}**)."
        )

    prompt = build_grounded_prompt(
        executive_summary=base_summary,
        confirmed_observations=confirmed_observations,
        ranked_root_causes=ranked_root_causes,
        manual_citations=[f"{h.citation} (score={h.score:.4f})" for h in manual_hits],
    )
    executive_summary, mode_used = maybe_refine_summary_with_llm(base_summary, prompt, cfg)

    markdown_report = _render_markdown(
        incident_title=incident_title,
        mode_used=mode_used,
        severity=severity,
        executive_summary=executive_summary,
        confirmed_observations=confirmed_observations,
        timeline_highlights=timeline_highlights,
        ranked_root_causes=ranked_root_causes,
        manual_hits=manual_hits,
        recommendations=recommendations,
    )

    return InvestigationReport(
        incident_title=incident_title,
        mode_used=mode_used,
        severity=severity,
        executive_summary=executive_summary,
        confirmed_observations=confirmed_observations,
        timeline_highlights=timeline_highlights,
        ranked_root_causes=ranked_root_causes,
        manual_evidence=manual_hits,
        recommendations=recommendations,
        markdown_report=markdown_report,
    )
