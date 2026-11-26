import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURE_LABELS = {
    "t1_win_rate_5": "Team 1 recent 5-match win rate",
    "t1_win_rate_10": "Team 1 10-match win rate",
    "t2_win_rate_5": "Team 2 recent 5-match win rate",
    "t2_win_rate_10": "Team 2 10-match win rate",
    "diff_win_rate_5": "5-match form difference",
    "diff_win_rate_10": "10-match win-rate difference",
    "diff_avg_runs_for_10": "Average runs scored difference",
    "diff_xi_bat_sr_10": "Playing XI batting strike-rate edge",
    "diff_xi_bowl_econ_10": "Playing XI bowling economy difference",
    "diff_xi_total_exp": "Playing XI combined experience edge",
    "t1_xi_bat_avg_10": "Team 1 XI batting average",
    "t2_xi_bat_avg_10": "Team 2 XI batting average",
    "venue_avg_1st_inn_runs": "Venue average 1st-innings score",
    "venue_chase_win_rate": "Venue historical chasing win rate",
    "t1_is_chasing": "Team 1 chasing after toss",
    "t1_venue_chase_edge": "Toss + venue chasing advantage",
    "is_ilt20": "ILT20 competition baseline",
    "is_bbl": "BBL competition baseline",
}


class PredictorService:
    def __init__(self) -> None:
        self.matches = pd.read_parquet(PROCESSED_DIR / "match_features.parquet").sort_values(
            ["match_date", "match_id"]
        )
        self.player_stats = pd.read_parquet(PROCESSED_DIR / "player_match_stats.parquet").sort_values(
            ["match_date", "match_id"]
        )

        self.winner_model = joblib.load(MODELS_DIR / "lgbm_match_winner.joblib")
        self.score_model = joblib.load(MODELS_DIR / "lgbm_first_innings_score.joblib")
        self.collapse_model = joblib.load(MODELS_DIR / "lgbm_pp_collapse.joblib")

        self.bat_runs_model = joblib.load(MODELS_DIR / "players/bat_runs.joblib")
        self.bat_20_model = joblib.load(MODELS_DIR / "players/bat_20plus.joblib")
        self.bat_30_model = joblib.load(MODELS_DIR / "players/bat_30plus.joblib")
        self.bowl_wkts_model = joblib.load(MODELS_DIR / "players/bowl_wickets.joblib")
        self.bowl_econ_model = joblib.load(MODELS_DIR / "players/bowl_economy.joblib")

        self.explainer = shap.TreeExplainer(self.winner_model)
        self.winner_features = list(self.winner_model.feature_name_)
        self.innings_features = list(self.score_model.feature_name_)

    def list_teams(self, competition: str | None = None) -> list[str]:
        df = self.matches if not competition else self.matches[self.matches["competition"] == competition]
        teams = sorted(set(df["team_1"]).union(set(df["team_2"])))
        return teams

    def list_venues(self, competition: str | None = None) -> list[str]:
        df = self.matches if not competition else self.matches[self.matches["competition"] == competition]
        return sorted(df["venue_clean"].dropna().unique().tolist())

    def get_recent_matches(self, competition: str = "ilt20", limit: int = 15) -> list[dict]:
        df = self.matches[self.matches["competition"] == competition].tail(limit).iloc[::-1]
        cols = [
            "match_id", "match_date", "competition", "venue_clean",
            "team_1", "team_2", "toss_winner", "toss_decision",
            "winner", "innings_1_runs", "innings_2_runs",
        ]
        records = df[cols].copy()
        records["match_date"] = records["match_date"].dt.strftime("%Y-%m-%d")
        return records.to_dict(orient="records")

    def _latest_team_snapshot(self, team: str) -> dict[str, float]:
        t1_rows = self.matches[self.matches["team_1"] == team]
        t2_rows = self.matches[self.matches["team_2"] == team]

        latest_t1 = t1_rows.iloc[-1] if not t1_rows.empty else None
        latest_t2 = t2_rows.iloc[-1] if not t2_rows.empty else None

        if latest_t1 is None and latest_t2 is None:
            return {
                "matches_prior": 0.0, "win_rate_5": 0.5, "win_rate_10": 0.5,
                "avg_runs_for_10": 155.0, "avg_runs_against_10": 155.0,
                "xi_bat_sr_10": 128.0, "xi_bat_avg_10": 24.0,
                "xi_bowl_econ_10": 8.0, "xi_bowl_wkts_10": 1.1, "xi_total_exp": 300.0,
            }

        use_t1 = latest_t2 is None or (latest_t1 is not None and latest_t1["match_date"] >= latest_t2["match_date"])
        row = latest_t1 if use_t1 else latest_t2
        p = "t1_" if use_t1 else "t2_"

        return {
            "matches_prior": float(row[f"{p}matches_prior"] + 1),
            "win_rate_5": float(row[f"{p}win_rate_5"]) if pd.notna(row[f"{p}win_rate_5"]) else 0.5,
            "win_rate_10": float(row[f"{p}win_rate_10"]) if pd.notna(row[f"{p}win_rate_10"]) else 0.5,
            "avg_runs_for_10": float(row[f"{p}avg_runs_for_10"]) if pd.notna(row[f"{p}avg_runs_for_10"]) else 155.0,
            "avg_runs_against_10": float(row[f"{p}avg_runs_against_10"]) if pd.notna(row[f"{p}avg_runs_against_10"]) else 155.0,
            "xi_bat_sr_10": float(row[f"{p}xi_bat_sr_10"]) if pd.notna(row[f"{p}xi_bat_sr_10"]) else 128.0,
            "xi_bat_avg_10": float(row[f"{p}xi_bat_avg_10"]) if pd.notna(row[f"{p}xi_bat_avg_10"]) else 24.0,
            "xi_bowl_econ_10": float(row[f"{p}xi_bowl_econ_10"]) if pd.notna(row[f"{p}xi_bowl_econ_10"]) else 8.0,
            "xi_bowl_wkts_10": float(row[f"{p}xi_bowl_wkts_10"]) if pd.notna(row[f"{p}xi_bowl_wkts_10"]) else 1.1,
            "xi_total_exp": float(row[f"{p}xi_total_exp"]) if pd.notna(row[f"{p}xi_total_exp"]) else 300.0,
        }

    def _latest_venue_snapshot(self, venue: str) -> tuple[float, float, int]:
        v_rows = self.matches[self.matches["venue_clean"] == venue]
        if v_rows.empty:
            return 160.0, 0.55, 0
        avg_1st = float(v_rows["innings_1_runs"].mean())
        chase_rate = float(v_rows["chase_won"].mean())
        return avg_1st, chase_rate, len(v_rows)

    def predict_match(
        self,
        team_1: str,
        team_2: str,
        venue: str,
        competition: str = "ilt20",
        toss_winner: str | None = None,
        toss_decision: str | None = None,
    ) -> dict:
        s1 = self._latest_team_snapshot(team_1)
        s2 = self._latest_team_snapshot(team_2)
        v_avg_runs, v_chase_rate, v_count = self._latest_venue_snapshot(venue)

        # Determine chasing role (default to neutral 0.5 edge if pre-toss)
        if toss_winner and toss_decision:
            t1_won_toss = toss_winner == team_1
            field_first = toss_decision.lower() == "field"
            t1_chasing = 1 if ((t1_won_toss and field_first) or (not t1_won_toss and not field_first)) else 0
            t1_chase_edge = (t1_chasing * 2 - 1) * (v_chase_rate - 0.5)
        else:
            t1_chasing = 0
            t1_chase_edge = 0.0

        row_dict = {
            "t1_win_rate_5": s1["win_rate_5"],
            "t1_win_rate_10": s1["win_rate_10"],
            "t2_win_rate_5": s2["win_rate_5"],
            "t2_win_rate_10": s2["win_rate_10"],
            "diff_win_rate_5": s1["win_rate_5"] - s2["win_rate_5"],
            "diff_win_rate_10": s1["win_rate_10"] - s2["win_rate_10"],
            "diff_avg_runs_for_10": s1["avg_runs_for_10"] - s2["avg_runs_for_10"],
            "diff_xi_bat_sr_10": s1["xi_bat_sr_10"] - s2["xi_bat_sr_10"],
            "diff_xi_bowl_econ_10": s1["xi_bowl_econ_10"] - s2["xi_bowl_econ_10"],
            "diff_xi_total_exp": s1["xi_total_exp"] - s2["xi_total_exp"],
            "t1_xi_bat_avg_10": s1["xi_bat_avg_10"],
            "t2_xi_bat_avg_10": s2["xi_bat_avg_10"],
            "venue_avg_1st_inn_runs": v_avg_runs,
            "venue_chase_win_rate": v_chase_rate,
            "t1_is_chasing": t1_chasing,
            "t1_venue_chase_edge": t1_chase_edge,
            "is_ilt20": 1 if competition == "ilt20" else 0,
            "is_bbl": 1 if competition == "bbl" else 0,
        }

        X_win = pd.DataFrame([row_dict])[self.winner_features]
        prob_t1 = float(self.winner_model.predict_proba(X_win)[0, 1])
        prob_t2 = 1.0 - prob_t1

        shap_vals = self.explainer.shap_values(X_win)
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        contrib = pd.Series(sv, index=self.winner_features).sort_values()

        pos_factors = [
            {"feature": feat, "impact": round(float(val), 4), "description": FEATURE_LABELS.get(feat, feat)}
            for feat, val in contrib[contrib > 0].iloc[::-1].head(4).items()
        ]
        neg_factors = [
            {"feature": feat, "impact": round(float(val), 4), "description": FEATURE_LABELS.get(feat, feat)}
            for feat, val in contrib[contrib < 0].head(4).items()
        ]

        # Expected 1st innings score & collapse risk (assuming t1 bats first if chasing==0)
        bat_s, bowl_s = (s2, s1) if t1_chasing == 1 else (s1, s2)
        inn_dict = {
            "bat_win_rate_10": bat_s["win_rate_10"],
            "bat_avg_runs_for_10": bat_s["avg_runs_for_10"],
            "bat_xi_bat_sr_10": bat_s["xi_bat_sr_10"],
            "bat_xi_bat_avg_10": bat_s["xi_bat_avg_10"],
            "bat_xi_total_exp": bat_s["xi_total_exp"],
            "bowl_win_rate_10": bowl_s["win_rate_10"],
            "bowl_avg_runs_against_10": bowl_s["avg_runs_against_10"],
            "bowl_xi_bowl_econ_10": bowl_s["xi_bowl_econ_10"],
            "bowl_xi_bowl_wkts_10": bowl_s["xi_bowl_wkts_10"],
            "bowl_xi_total_exp": bowl_s["xi_total_exp"],
            "venue_avg_1st_inn_runs": v_avg_runs,
            "venue_chase_win_rate": v_chase_rate,
            "is_ilt20": 1 if competition == "ilt20" else 0,
            "is_bbl": 1 if competition == "bbl" else 0,
        }
        X_inn = pd.DataFrame([inn_dict])[self.innings_features]
        exp_score = float(self.score_model.predict(X_inn)[0])
        collapse_prob = float(self.collapse_model.predict_proba(X_inn)[0, 1])

        margin = abs(prob_t1 - 0.5)
        confidence = "High" if margin >= 0.15 else ("Moderate" if margin >= 0.07 else "Low (Toss-up)")

        notes: list[str] = []
        if not toss_winner:
            notes.append("Pre-toss estimate: probability will update once toss decision is known.")
        if s1["matches_prior"] < 5 or s2["matches_prior"] < 5:
            notes.append("Limited recent team history (< 5 matches) for at least one side.")
        if v_count < 10:
            notes.append(f"Small venue sample size ({v_count} prior matches) at {venue}.")

        return {
            "team_1": team_1,
            "team_2": team_2,
            "venue": venue,
            "competition": competition,
            "predicted_winner": team_1 if prob_t1 >= 0.5 else team_2,
            "team_1_win_prob": round(prob_t1, 4),
            "team_2_win_prob": round(prob_t2, 4),
            "confidence_level": confidence,
            "expected_first_innings_score": round(exp_score, 1),
            "pp_collapse_prob": round(collapse_prob, 4),
            "positive_factors_t1": pos_factors,
            "negative_factors_t1": neg_factors,
            "uncertainty_notes": notes,
        }

    def search_players(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        latest = self.player_stats.groupby("player_id").tail(1)
        matched = latest[latest["player_name"].str.lower().str.contains(q, na=False)].head(limit)

        results: list[dict] = []
        for _, r in matched.iterrows():
            bat_X = pd.DataFrame(
                [
                    {
                        "roster_order": r["roster_order"],
                        "pre_bat_innings": r["pre_bat_innings"],
                        "pre_bat_avg_10": r["pre_bat_avg_10"],
                        "pre_bat_sr_10": r["pre_bat_sr_10"],
                        "venue_avg_1st_inn_runs": 161.0,
                        "opp_bowl_econ_10": 8.0,
                        "is_ilt20": 1,
                        "is_bbl": 0,
                    }
                ]
            )
            bowl_X = pd.DataFrame(
                [
                    {
                        "pre_bowl_innings": r["pre_bowl_innings"],
                        "pre_bowl_econ_10": r["pre_bowl_econ_10"],
                        "pre_bowl_wkts_avg_10": r["pre_bowl_wkts_avg_10"],
                        "venue_avg_1st_inn_runs": 161.0,
                        "opp_bat_sr_10": 132.0,
                        "is_ilt20": 1,
                        "is_bbl": 0,
                    }
                ]
            )
            results.append(
                {
                    "player_id": r["player_id"],
                    "player_name": r["player_name"],
                    "latest_team": r["team"],
                    "bat_innings": int(r["pre_bat_innings"]),
                    "bat_sr_10": round(float(r["pre_bat_sr_10"]), 1) if pd.notna(r["pre_bat_sr_10"]) else None,
                    "proj_runs": round(float(self.bat_runs_model.predict(bat_X)[0]), 1),
                    "prob_20plus": round(float(self.bat_20_model.predict_proba(bat_X)[0, 1]), 3),
                    "prob_30plus": round(float(self.bat_30_model.predict_proba(bat_X)[0, 1]), 3),
                    "bowl_innings": int(r["pre_bowl_innings"]),
                    "bowl_econ_10": round(float(r["pre_bowl_econ_10"]), 2) if pd.notna(r["pre_bowl_econ_10"]) else None,
                    "proj_wickets": round(float(self.bowl_wkts_model.predict(bowl_X)[0]), 2),
                    "proj_economy": round(float(self.bowl_econ_model.predict(bowl_X)[0]), 2),
                }
            )
        return results

    def get_model_metrics(self) -> dict:
        out = {}
        for name in ["baseline_metrics.json", "match_winner_metrics.json", "team_collapse_metrics.json", "player_models_metrics.json"]:
            p = MODELS_DIR / name
            if p.exists():
                out[name.replace(".json", "")] = json.loads(p.read_text())
        return out