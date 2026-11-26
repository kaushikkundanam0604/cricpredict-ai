# CricPredict AI

AI-powered cricket analytics, match prediction, and live win probability platform.

## Tech Stack (Phase 1)
- Python 3.14 (Apple Silicon ARM64)
- FastAPI + Uvicorn

## Local Setup
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload