from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
DELIVERIES_PATH = Path("data/processed/deliveries.parquet")


class LiveChaseSimulator:
    def __init__(self) -> None:
        self.model = joblib.load(MODELS_DIR / "lgbm_live_chase.joblib")
        self.features = list(self.model.feature_name_)

        # Compute empirical 2nd-innings ball outcome probabilities by phase
        deliveries = pd.read_parquet(
            DELIVERIES_PATH,
            columns=["innings", "is_super_over", "is_legal_ball", "over", "runs_total", "is_wicket"],
        )
        chase = deliveries[(~deliveries["is_super_over"]) & (deliveries["innings"] == 2) & deliveries["is_legal_ball"]].copy()

        def phase_probs(sub: pd.DataFrame) -> np.ndarray:
            n = len(sub)
            p_w = float(sub["is_wicket"].sum()) / n
            non_w = sub[~sub["is_wicket"]]
            p0 = float((non_w["runs_total"] == 0).sum()) / n
            p1 = float((non_w["runs_total"] == 1).sum()) / n
            p2 = float((non_w["runs_total"] == 2).sum()) / n
            p3 = float((non_w["runs_total"] == 3).sum()) / n
            p4 = float((non_w["runs_total"] == 4).sum()) / n
            p6 = max(0.0, 1.0 - (p_w + p0 + p1 + p2 + p3 + p4))
            probs = np.array([p0, p1, p2, p3, p4, p6, p_w], dtype=float)
            return probs / probs.sum()

        self.phase_probs = {
            "pp": phase_probs(chase[chase["over"] < 6]),
            "mid": phase_probs(chase[(chase["over"] >= 6) & (chase["over"] < 15)]),
            "death": phase_probs(chase[chase["over"] >= 15]),
        }
        self.outcomes_runs = np.array([0, 1, 2, 3, 4, 6, 0], dtype=int)
        self.outcomes_wkts = np.array([0, 0, 0, 0, 0, 0, 1], dtype=int)

    def simulate_and_predict(
        self,
        target: int,
        current_score: int,
        balls_bowled: int,
        wickets_lost: int,
        last_18_runs: int,
        last_18_wkts: int,
        competition: str = "ilt20",
        n_sims: int = 2000,
    ) -> dict:
        runs_remaining = max(0, target - current_score)
        balls_remaining = max(0, 120 - balls_bowled)
        wickets_remaining = max(0, 10 - wickets_lost)

        crr = current_score / (balls_bowled / 6.0) if balls_bowled > 0 else 0.0
        rrr = (runs_remaining / (balls_remaining / 6.0)) if balls_remaining > 0 else (36.0 if runs_remaining > 0 else 0.0)

        # Handle already-decided states cleanly
        if runs_remaining == 0:
            return {
                "ml_chase_win_prob": 1.0,
                "monte_carlo_win_prob": 1.0,
                "current_run_rate": round(crr, 2),
                "required_run_rate": 0.0,
                "projected_score_p10": current_score,
                "projected_score_p50": current_score,
                "projected_score_p90": current_score,
            }
        if balls_remaining == 0 or wickets_remaining == 0:
            return {
                "ml_chase_win_prob": 0.0,
                "monte_carlo_win_prob": 0.0,
                "current_run_rate": round(crr, 2),
                "required_run_rate": round(rrr, 2),
                "projected_score_p10": current_score,
                "projected_score_p50": current_score,
                "projected_score_p90": current_score,
            }

        row = pd.DataFrame(
            [
                {
                    "runs_remaining": runs_remaining,
                    "balls_remaining": balls_remaining,
                    "wickets_remaining": wickets_remaining,
                    "current_run_rate": crr,
                    "required_run_rate": rrr,
                    "rrr_minus_crr": rrr - crr,
                    "last_18_runs": last_18_runs,
                    "last_18_wkts": last_18_wkts,
                    "is_ilt20": 1 if competition == "ilt20" else 0,
                    "is_bbl": 1 if competition == "bbl" else 0,
                }
            ]
        )[self.features]
        ml_prob = float(self.model.predict_proba(row)[0, 1])

        # Vectorized Monte Carlo simulation over remaining balls
        rng = np.random.default_rng()
        sim_scores = np.full(n_sims, current_score, dtype=int)
        sim_wkts = np.full(n_sims, wickets_lost, dtype=int)
        sim_won = np.zeros(n_sims, dtype=bool)

        for b in range(balls_bowled, 120):
            over = b // 6
            probs = self.phase_probs["pp"] if over < 6 else (self.phase_probs["mid"] if over < 15 else self.phase_probs["death"])
            active = (~sim_won) & (sim_wkts < 10)
            if not np.any(active):
                break
            draws = rng.choice(7, size=int(active.sum()), p=probs)
            sim_scores[active] += self.outcomes_runs[draws]
            sim_wkts[active] += self.outcomes_wkts[draws]
            sim_won = sim_scores >= target

        mc_prob = float(sim_won.mean())
        p10, p50, p90 = np.percentile(sim_scores, [10, 50, 90])

        return {
            "ml_chase_win_prob": round(ml_prob, 4),
            "monte_carlo_win_prob": round(mc_prob, 4),
            "current_run_rate": round(crr, 2),
            "required_run_rate": round(rrr, 2),
            "projected_score_p10": round(p10),
            "projected_score_p50": round(p50),
            "projected_score_p90": round(p90),
        }