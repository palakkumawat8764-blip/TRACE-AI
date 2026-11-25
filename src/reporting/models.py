"""Data models for Milestone 5 evidence-based root cause investigation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.rag.models import RetrievedChunk


@dataclass(frozen=True)
class EvidenceItem:
    """A single grounded piece of evidence tied to a source file and timestamp."""

    timestamp: str
    source_file: str
    category: str
    detail: str


@dataclass(frozen=True)
class RootCauseCandidate:
    """One ranked possible root cause (explicitly separated from confirmed observations)."""

    rank: int
    title: str
    confidence: str
    confidence_score: float
    rationale: str
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class InvestigationReport:
    """Complete structured and Markdown investigation report."""

    incident_title: str
    mode_used: str
    severity: str
    executive_summary: str
    confirmed_observations: list[EvidenceItem]
    timeline_highlights: list[dict[str, Any]]
    ranked_root_causes: list[RootCauseCandidate]
    manual_evidence: list[RetrievedChunk]
    recommendations: list[str]
    markdown_report: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly dictionary for export."""
        return {
            "incident_title": self.incident_title,
            "mode_used": self.mode_used,
            "severity": self.severity,
            "executive_summary": self.executive_summary,
            "confirmed_observations": [asdict(item) for item in self.confirmed_observations],
            "timeline_highlights": self.timeline_highlights,
            "ranked_root_causes": [
                {
                    "rank": rc.rank,
                    "title": rc.title,
                    "confidence": rc.confidence,
                    "confidence_score": rc.confidence_score,
                    "rationale": rc.rationale,
                    "supporting_evidence": [asdict(ev) for ev in rc.supporting_evidence],
                    "recommendations": rc.recommendations,
                }
                for rc in self.ranked_root_causes
            ],
            "manual_evidence": [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "citation": hit.citation,
                    "text": hit.chunk.text,
                }
                for hit in self.manual_evidence
            ],
            "recommendations": self.recommendations,
            "markdown_report": self.markdown_report,
        }
