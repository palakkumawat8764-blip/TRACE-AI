"""Central configuration for TRACE AI.

Loads settings from environment variables (and `.env` if present) so that
switching between free `mock` mode and live `gemini` mode requires no code changes.
"""

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows config.py to run before optional dependencies are installed
    load_dotenv = None

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

# Dark theme palette shared across Streamlit CSS and Plotly figures
DARK_THEME = {
    "bg_primary": "#0B0F19",
    "bg_secondary": "#131A2A",
    "bg_card": "#1A2236",
    "text_primary": "#F1F5F9",
    "text_muted": "#94A3B8",
    "accent_cyan": "#38BDF8",
    "accent_amber": "#F59E0B",
    "accent_red": "#EF4444",
    "accent_green": "#10B981",
    "border": "#26314A",
}


@dataclass(frozen=True)
class AppConfig:
    """Typed application settings for TRACE AI."""

    llm_mode: str
    gemini_api_key: str
    gemini_model: str
    anomaly_contamination: float
    correlation_window_seconds: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_top_k: int

    @property
    def is_gemini_ready(self) -> bool:
        """Return True only when Gemini mode is selected and an API key is provided."""
        return self.llm_mode == "gemini" and bool(self.gemini_api_key.strip())


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config(env_path: Path | None = None) -> AppConfig:
    """Load configuration from `.env` and environment variables."""
    target_env = env_path or (PROJECT_ROOT / ".env")
    if load_dotenv is not None and target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    llm_mode = os.getenv("LLM_MODE", "mock").strip().lower()
    if llm_mode not in {"mock", "gemini"}:
        llm_mode = "mock"

    return AppConfig(
        llm_mode=llm_mode,
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        anomaly_contamination=_get_float("ANOMALY_CONTAMINATION", 0.08),
        correlation_window_seconds=_get_int("CORRELATION_WINDOW_SECONDS", 180),
        rag_chunk_size=_get_int("RAG_CHUNK_SIZE", 500),
        rag_chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 100),
        rag_top_k=_get_int("RAG_TOP_K", 3),
    )
