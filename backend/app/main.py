from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import MatchPredictionRequest, MatchPredictionResponse
from backend.app.services.predictor import PredictorService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = PredictorService()
    yield


app = FastAPI(
    title="CricPredict AI",
    description="AI-powered cricket analytics, match prediction, and live win probability API.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "CricPredict AI", "status": "running", "docs": "/docs"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/api/teams")
def get_teams(request: Request, competition: str | None = Query(default="ilt20")) -> dict:
    return {"teams": request.app.state.predictor.list_teams(competition)}


@app.get("/api/venues")
def get_venues(request: Request, competition: str | None = Query(default="ilt20")) -> dict:
    return {"venues": request.app.state.predictor.list_venues(competition)}


@app.get("/api/matches/recent")
def get_recent_matches(
    request: Request,
    competition: str = Query(default="ilt20"),
    limit: int = Query(default=15, ge=1, le=50),
) -> dict:
    return {"matches": request.app.state.predictor.get_recent_matches(competition, limit)}


@app.get("/api/players/search")
def search_players(
    request: Request,
    q: str = Query(..., min_length=2),
    limit: int = Query(default=10, ge=1, le=25),
) -> dict:
    return {"players": request.app.state.predictor.search_players(q, limit)}


@app.get("/api/metrics")
def get_metrics(request: Request) -> dict:
    return request.app.state.predictor.get_model_metrics()


@app.post("/api/predict/match", response_model=MatchPredictionResponse)
def predict_match(request: Request, payload: MatchPredictionRequest) -> dict:
    return request.app.state.predictor.predict_match(
        team_1=payload.team_1,
        team_2=payload.team_2,
        venue=payload.venue,
        competition=payload.competition,
        toss_winner=payload.toss_winner,
        toss_decision=payload.toss_decision,
    )