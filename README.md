# TRACE AI — AI-Powered Incident Investigation & Root Cause Detection System

TRACE AI is a modular Python and Streamlit system for investigating production incidents across **CSV telemetry**, **`.log` / `.txt` event logs & tickets**, and **`.pdf` technical manuals**. It combines classical machine learning anomaly detection, cross-document timestamp correlation, local **FAISS** RAG manual retrieval, and evidence-grounded root cause reporting in a dark-themed dashboard.

---

## Key Capabilities

- **Multi-File Ingestion (`src/ingestion/`):** Parses CSV metrics, `.log`/`.txt` event logs, and `.pdf` documents into normalized UTC timelines (`events_df`, `metrics_df`, `manual_pages`).
- **Hybrid Anomaly Detection (`src/analysis/`):** Combines `scikit-learn` **Isolation Forest** + **Robust Z-Score (MAD)** on scaled numeric telemetry and 1-minute severity-weighted burst detection on logs.
- **Cross-Document Correlation & Timeline Reconstruction (`src/analysis/`):** Links correlated events across file types within a configurable time window and segments incidents into readable phases (`Pre-Incident Baseline`, `Trigger & Escalation`, `Peak Incident`, `Mitigation & Recovery`, `Post-Recovery Stability`).
- **Local FAISS RAG Search (`src/rag/`):** Chunks PDF manuals with sliding-window overlap and indexes vectors in **`faiss-cpu`** using local `scikit-learn` TF-IDF / `TruncatedSVD` (LSA) — zero PyTorch or external embedding API required.
- **Evidence-Grounded Root Cause Reports (`src/reporting/`):** Ranks possible root causes with confidence scores (`HIGH` / `MEDIUM` / `LOW`), separates **Confirmed Observations** from **Possible Root Causes**, cites exact timestamps, filenames, and PDF pages, and works offline in `mock` mode (with optional **Google Gemini API** support).
- **Modern Dark Dashboard & Exports (`app.py`, `src/ui/`):** Interactive Plotly charts, filterable unified logs, live RAG manual search, and downloadable **PDF**, **CSV**, **Markdown**, and **JSON** reports.

---

## Project Structure

```text
TRACE-AI/
├── app.py                         # Dark-themed Streamlit dashboard entry point
├── requirements.txt               # Lightweight dependencies (Streamlit Cloud ready)
├── .env.example                   # Local environment configuration template
├── .gitignore                     # Excludes .env, secrets.toml, .venv, and ZIPs
├── .streamlit/
│   ├── config.toml                # Dark theme styling
│   └── secrets.toml.example       # Template for Streamlit Cloud Secrets
├── sample_data/                   # Built-in synthetic SEV-1 incident dataset
│   ├── server_metrics.csv
│   ├── app_events.log
│   ├── incident_summary_ticket.txt
│   └── technical_manual.pdf
├── src/
│   ├── config.py                  # Typed AppConfig & DARK_THEME palette
│   ├── ingestion/                 # CSV, TXT/LOG, PDF parsers & unified loader
│   ├── analysis/                  # Isolation Forest, log burst, correlation, timeline
│   ├── rag/                       # Page chunker, TF-IDF/LSA + FAISS index
│   ├── reporting/                 # Rule-based RCA, Gemini client, report generator
│   └── ui/                        # Plotly charts, dark CSS, PDF & CSV exporters
└── tests/                         # Unit, UI, and end-to-end integration tests
```

---

## Local Installation & Setup

### 1. Create a Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 3. Launch the Dashboard Locally
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The app starts in **free local `mock` mode** by default—no API key is required. You can immediately test using the **Built-in Sample Incident** or upload your own `.csv`, `.log`, `.txt`, and `.pdf` files in the sidebar.

---

## Optional: Enabling Google Gemini API

1. Copy `.env.example` to `.env` (never commit `.env` to GitHub).
2. Set:
   ```env
   LLM_MODE=gemini
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
3. Restart `streamlit run app.py` (or select `gemini` and paste your key temporarily in the sidebar). If the API call fails or no key is provided, TRACE AI automatically falls back to local rule-based mode.

---

## Deploying to Streamlit Community Cloud

1. **Verify Secrets Are Excluded:** Confirm `.gitignore` excludes `.env`, `.streamlit/secrets.toml`, `.venv/`, and `dist/` before pushing to GitHub.
2. **Push to GitHub:** Push your repository (`app.py`, `requirements.txt`, `.streamlit/config.toml`, `sample_data/`, `src/`, `tests/`, `README.md`) to a GitHub repository.
3. **Create App on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
   - Click **New app**, select your repository and branch, and set **Main file path** to `app.py`.
4. **Configure Secrets (Optional):**
   - In the Streamlit Cloud dashboard, open **App Settings -> Secrets** and paste the template from `.streamlit/secrets.toml.example` (keep `LLM_MODE = "mock"` for free offline mode, or set `LLM_MODE = "gemini"` with your `GEMINI_API_KEY`).
5. **Deploy:** Click **Deploy!** — Streamlit Cloud will install `requirements.txt` and serve your live dark-themed dashboard.

---

## ZIP Distribution

A clean distribution archive (excluding `.venv/`, `.env`, `.streamlit/secrets.toml`, `__pycache__/`, and `dist/`) can be packaged under `dist/TRACE-AI-release.zip` for offline sharing or submission. Recipients simply unzip, create a virtual environment, run `pip install -r requirements.txt`, and run `streamlit run app.py`.
