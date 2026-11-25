"""Custom dark-theme CSS styling for the TRACE AI Streamlit dashboard."""

from src.config import DARK_THEME


def get_dark_dashboard_css() -> str:
    """Return CSS rules aligned with DARK_THEME for KPI cards, badges, and layout."""
    t = DARK_THEME
    return f"""
    <style>
        .stApp {{
            background-color: {t['bg_primary']};
            color: {t['text_primary']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {t['bg_secondary']};
            border-right: 1px solid {t['border']};
        }}
        .trace-hero {{
            background: linear-gradient(135deg, {t['bg_secondary']} 0%, {t['bg_card']} 100%);
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1rem;
        }}
        .trace-hero h1 {{
            margin: 0;
            font-size: 1.75rem;
            color: {t['text_primary']};
            letter-spacing: 0.4px;
        }}
        .trace-hero p {{
            margin: 0.35rem 0 0 0;
            color: {t['text_muted']};
            font-size: 0.95rem;
        }}
        .trace-kpi {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }}
        .trace-kpi-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: {t['text_muted']};
        }}
        .trace-kpi-value {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {t['text_primary']};
            margin-top: 0.2rem;
        }}
        .trace-badge {{
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid {t['border']};
            background-color: {t['bg_secondary']};
            color: {t['accent_cyan']};
        }}
        .trace-badge-crit {{
            color: {t['accent_red']};
            border-color: {t['accent_red']};
        }}
        .trace-badge-ok {{
            color: {t['accent_green']};
            border-color: {t['accent_green']};
        }}
    </style>
    """
