from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"


def test_match_prediction_and_shap():
    with TestClient(app) as client:
        res = client.post(
            "/api/predict/match",
            json={
                "team_1": "Dubai Capitals",
                "team_2": "Desert Vipers",
                "venue": "Dubai International Cricket Stadium",
                "competition": "ilt20",
                "toss_winner": "Desert Vipers",
                "toss_decision": "field",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["team_1_win_prob"] <= 1.0
        assert abs(data["team_1_win_prob"] + data["team_2_win_prob"] - 1.0) < 1e-3
        assert len(data["positive_factors_t1"]) + len(data["negative_factors_t1"]) > 0


def test_live_chase_endpoint():
    with TestClient(app) as client:
        res = client.post(
            "/api/predict/live",
            json={
                "target": 175,
                "current_score": 130,
                "balls_bowled": 90,
                "wickets_lost": 3,
                "last_18_runs": 28,
                "last_18_wkts": 0,
                "competition": "ilt20",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["ml_chase_win_prob"] <= 1.0
        assert data["projected_score_p10"] <= data["projected_score_p50"] <= data["projected_score_p90"]