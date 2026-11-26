from pydantic import BaseModel, Field


class MatchPredictionRequest(BaseModel):
    team_1: str = Field(..., examples=["Dubai Capitals"])
    team_2: str = Field(..., examples=["Desert Vipers"])
    venue: str = Field(..., examples=["Dubai International Cricket Stadium"])
    competition: str = Field(default="ilt20", examples=["ilt20"])
    toss_winner: str | None = Field(default=None, examples=["Desert Vipers"])
    toss_decision: str | None = Field(default=None, examples=["field"])


class ShapFactor(BaseModel):
    feature: str
    impact: float
    description: str


class MatchPredictionResponse(BaseModel):
    team_1: str
    team_2: str
    venue: str
    competition: str
    predicted_winner: str
    team_1_win_prob: float
    team_2_win_prob: float
    confidence_level: str
    expected_first_innings_score: float
    pp_collapse_prob: float
    positive_factors_t1: list[ShapFactor]
    negative_factors_t1: list[ShapFactor]
    uncertainty_notes: list[str]


class LivePredictionRequest(BaseModel):
    target: int = Field(..., ge=1, le=400, examples=[175])
    current_score: int = Field(..., ge=0, le=400, examples=[112])
    balls_bowled: int = Field(..., ge=0, le=120, examples=[84])
    wickets_lost: int = Field(..., ge=0, le=10, examples=[3])
    last_18_runs: int = Field(default=24, ge=0, le=108)
    last_18_wkts: int = Field(default=1, ge=0, le=10)
    competition: str = Field(default="ilt20")


class LivePredictionResponse(BaseModel):
    ml_chase_win_prob: float
    monte_carlo_win_prob: float
    current_run_rate: float
    required_run_rate: float
    projected_score_p10: int
    projected_score_p50: int
    projected_score_p90: int