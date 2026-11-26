from datetime import datetime, timezone
from fastapi import FastAPI

app = FastAPI(
    title="CricPredict AI",
    description="AI-powered cricket analytics, match prediction, and live win probability API.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "CricPredict AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }