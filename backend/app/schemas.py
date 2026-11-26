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