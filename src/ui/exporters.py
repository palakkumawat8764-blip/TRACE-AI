"""PDF and CSV report exporters for TRACE AI investigation reports."""

from __future__ import annotations

from io import StringIO
import textwrap

import pandas as pd

from src.reporting.models import InvestigationReport


def _ascii_safe(text: str) -> str:
    """Normalize common Unicode punctuation to clean Latin-1/ASCII for Helvetica PDF output."""
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2192": "->",
        "\u2022": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    cleaned = text
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned.encode("latin-1", errors="replace").decode("latin-1")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf_bytes(lines: list[str], lines_per_page: int = 46) -> bytes:
    """Render wrapped text lines into a valid multi-page PDF 1.4 byte buffer."""
    pages = [
        lines[i : i + lines_per_page]
        for i in range(0, max(1, len(lines)), lines_per_page)
    ]

    objects: list[bytes] = [
        b"",  # 1: Catalog placeholder
        b"",  # 2: Pages placeholder
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids: list[int] = []
    next_id = 4

    for page_lines in pages:
        stream_cmds = ["BT", "/F1 10 Tf", "48 750 Td", "14 TL"]
        for idx, raw_line in enumerate(page_lines):
            escaped = _escape_pdf_text(_ascii_safe(raw_line))
            if idx == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
        stream_cmds.append("ET")
        stream_data = "\n".join(stream_cmds).encode("latin-1")

        content_id = next_id
        page_id = next_id + 1
        next_id += 2

        objects.append(
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream"
        )
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("latin-1")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")

    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        buf.extend(obj)
        buf.extend(b"\nendobj\n")

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(buf)


def export_report_to_pdf(report: InvestigationReport) -> bytes:
    """Convert an InvestigationReport into a clean multi-page PDF document."""
    wrapped_lines: list[str] = []

    def add_wrapped(text: str, indent: str = "", width: int = 88) -> None:
        cleaned = _ascii_safe(text).replace("**", "").replace("`", "")
        if not cleaned.strip():
            wrapped_lines.append("")
            return
        for raw_part in cleaned.splitlines():
            parts = textwrap.wrap(raw_part, width=width - len(indent)) or [""]
            for part in parts:
                wrapped_lines.append(f"{indent}{part}")

    add_wrapped(report.incident_title)
    add_wrapped(f"Mode: {report.mode_used} | Peak Severity: {report.severity}")
    add_wrapped("=" * 78)
    add_wrapped("")

    add_wrapped("1. EXECUTIVE SUMMARY")
    add_wrapped(report.executive_summary)
    add_wrapped("")

    add_wrapped("2. CONFIRMED OBSERVATIONS (DIRECTLY OBSERVED FACTS)")
    if report.confirmed_observations:
        for ev in report.confirmed_observations:
            add_wrapped(f"- [{ev.timestamp}] ({ev.source_file} | {ev.category}) {ev.detail}", indent="  ")
    else:
        add_wrapped("- No anomalous observations recorded.", indent="  ")
    add_wrapped("")

    add_wrapped("3. RECONSTRUCTED INCIDENT TIMELINE")
    for phase in report.timeline_highlights:
        add_wrapped(f"* {phase['phase']} ({phase['start']} -> {phase['end']}, {phase['event_count']} events)")
        for key_ev in phase["key_events"]:
            add_wrapped(f"- {key_ev}", indent="  ")
    add_wrapped("")

    add_wrapped("4. RANKED POSSIBLE ROOT CAUSES (HYPOTHESES)")
    if report.ranked_root_causes:
        for rc in report.ranked_root_causes:
            add_wrapped(
                f"Rank #{rc.rank}: {rc.title} | Confidence: {rc.confidence} ({int(rc.confidence_score * 100)}%)"
            )
            add_wrapped(f"Rationale: {rc.rationale}", indent="  ")
            for ev in rc.supporting_evidence:
                add_wrapped(f"- Evidence [{ev.timestamp} | {ev.source_file}]: {ev.detail}", indent="  ")
            for rec in rc.recommendations:
                add_wrapped(f"- Action: {rec}", indent="  ")
            add_wrapped("")
    else:
        add_wrapped("- No root cause hypotheses identified.", indent="  ")
        add_wrapped("")

    add_wrapped("5. TECHNICAL MANUAL EVIDENCE (FAISS RAG CITATIONS)")
    if report.manual_evidence:
        for hit in report.manual_evidence:
            excerpt = hit.chunk.text.replace("\n", " ")[:240]
            add_wrapped(f"[{hit.rank}] {hit.citation} (score={hit.score:.4f}): {excerpt}...", indent="  ")
    else:
        add_wrapped("- No technical manual citations available.", indent="  ")
    add_wrapped("")

    add_wrapped("6. PRIORITIZED TROUBLESHOOTING RECOMMENDATIONS")
    for idx, rec in enumerate(report.recommendations, start=1):
        add_wrapped(f"{idx}. {rec}", indent="  ")

    return _build_pdf_bytes(wrapped_lines)


def export_report_to_csv(report: InvestigationReport) -> bytes:
    """Convert an InvestigationReport into a structured UTF-8 CSV byte buffer."""
    rows: list[dict[str, str]] = [
        {
            "section": "Executive Summary",
            "rank": "",
            "timestamp": "",
            "source_file": "",
            "confidence": report.severity,
            "score": "",
            "title": report.incident_title,
            "detail": report.executive_summary,
        }
    ]

    for ev in report.confirmed_observations:
        rows.append(
            {
                "section": "Confirmed Observation",
                "rank": "",
                "timestamp": ev.timestamp,
                "source_file": ev.source_file,
                "confidence": "OBSERVED",
                "score": "",
                "title": ev.category,
                "detail": ev.detail,
            }
        )

    for phase in report.timeline_highlights:
        rows.append(
            {
                "section": "Timeline Phase",
                "rank": "",
                "timestamp": f"{phase['start']} -> {phase['end']}",
                "source_file": "",
                "confidence": "",
                "score": str(phase["event_count"]),
                "title": str(phase["phase"]),
                "detail": " | ".join(phase["key_events"]),
            }
        )

    for rc in report.ranked_root_causes:
        ev_summary = " || ".join(f"[{e.timestamp} | {e.source_file}] {e.detail}" for e in rc.supporting_evidence)
        rows.append(
            {
                "section": "Root Cause Hypothesis",
                "rank": str(rc.rank),
                "timestamp": "",
                "source_file": "",
                "confidence": rc.confidence,
                "score": f"{rc.confidence_score:.2f}",
                "title": rc.title,
                "detail": f"{rc.rationale} | Evidence: {ev_summary}",
            }
        )

    for hit in report.manual_evidence:
        rows.append(
            {
                "section": "RAG Manual Evidence",
                "rank": str(hit.rank),
                "timestamp": "",
                "source_file": hit.citation,
                "confidence": "",
                "score": f"{hit.score:.4f}",
                "title": hit.chunk.chunk_id,
                "detail": hit.chunk.text.replace("\n", " "),
            }
        )

    for idx, rec in enumerate(report.recommendations, start=1):
        rows.append(
            {
                "section": "Recommendation",
                "rank": str(idx),
                "timestamp": "",
                "source_file": "",
                "confidence": "",
                "score": "",
                "title": f"Action #{idx}",
                "detail": rec,
            }
        )

    buf = StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
