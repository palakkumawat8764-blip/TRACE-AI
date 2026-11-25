"""Configurable LLM integration (free offline mock mode or optional Gemini API)."""

from __future__ import annotations

from src.config import AppConfig
from src.reporting.models import EvidenceItem, RootCauseCandidate


def build_grounded_prompt(
    executive_summary: str,
    confirmed_observations: list[EvidenceItem],
    ranked_root_causes: list[RootCauseCandidate],
    manual_citations: list[str],
) -> str:
    """Build a citation-strict prompt for future Gemini API refinement."""
    obs_lines = "\n".join(
        f"- [{ev.timestamp}] ({ev.source_file} | {ev.category}) {ev.detail}"
        for ev in confirmed_observations
    )
    cause_lines = "\n".join(
        f"- Rank {rc.rank}: {rc.title} (confidence={rc.confidence}, score={rc.confidence_score})"
        for rc in ranked_root_causes
    )
    cite_lines = "\n".join(f"- {c}" for c in manual_citations) or "- None"

    return (
        "You are an SRE incident investigator. Summarize the incident using ONLY the verified "
        "facts below. Never invent timestamps, filenames, or metrics. Clearly distinguish "
        "CONFIRMED OBSERVATIONS from POSSIBLE ROOT CAUSES.\n\n"
        f"Summary:\n{executive_summary}\n\n"
        f"Confirmed Observations:\n{obs_lines}\n\n"
        f"Ranked Possible Root Causes:\n{cause_lines}\n\n"
        f"Manual Citations:\n{cite_lines}\n"
    )


def maybe_refine_summary_with_llm(
    default_summary: str,
    prompt: str,
    config: AppConfig,
) -> tuple[str, str]:
    """Return (summary_text, mode_used). Uses deterministic mock mode unless Gemini is enabled."""
    if not config.is_gemini_ready:
        return default_summary, "mock (rule-based local)"

    try:
        from google import genai

        client = genai.Client(api_key=config.gemini_api_key)
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=prompt,
        )
        text = (getattr(response, "text", "") or "").strip()
        if text:
            return text, f"gemini ({config.gemini_model})"
    except Exception as exc:
        return (
            f"{default_summary}\n\n*(Note: Gemini API unavailable; fell back to local rule-based mode: {exc})*",
            "mock (fallback)",
        )

    return default_summary, "mock (rule-based local)"
