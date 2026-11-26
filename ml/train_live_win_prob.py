import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DELIVERIES_PATH = Path("data/processed/deliveries.parquet")
MATCHES_PATH = Path("data/processed/match_features.parquet")
MODELS_DIR = Path("models")

LIVE_FEATURES = [
    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
    "rrr_minus_crr",
    "last_18_runs",
    "last_18_wkts",
    "is_ilt20",
    "is_bbl",
]


def build_live_chase_dataset() -> pd.DataFrame:
    matches = pd.read_parquet(MATCHES_PATH)[
        ["match_id", "match_date", "competition", "innings_1_runs", "innings_2_team", "winner", "win_method"]
    ]
    # Keep standard decided 20-over matches
    matches = matches[(matches["win_method"] == "standard") & matches["innings_1_runs"].notna()].copy()
    matches["target"] = matches["innings_1_runs"] + 1
    matches["chase_won"] = (matches["winner"] == matches["innings_2_team"]).astype(int)

    deliveries = pd.read_parquet(DELIVERIES_PATH)
    chase = deliveries[(~deliveries["is_super_over"]) & (deliveries["innings"] == 2)].copy()
    chase = chase.merge(
        matches[["match_id", "target", "chase_won"]],
        on="match_id",
        how="inner",
    )

    chase = chase.sort_values(["match_date", "match_id", "over", "ball_in_over_seq"]).reset_index(drop=True)

    g = chase.groupby("match_id")
    chase["current_score"] = g["runs_total"].cumsum()
    chase["wickets_lost"] = g["is_wicket"].cumsum()
    chase["legal_balls"] = g["is_legal_ball"].cumsum()

    chase["last_18_runs"] = g["runs_total"].transform(lambda s: s.rolling(18, min_periods=1).sum())
    chase["last_18_wkts"] = g["is_wicket"].transform(lambda s: s.rolling(18, min_periods=1).sum())

    chase["runs_remaining"] = (chase["target"] - chase["current_score"]).clip(lower=0)
    chase["balls_remaining"] = (120 - chase["legal_balls"]).clip(lower=0)
    chase["wickets_remaining"] = (10 - chase["wickets_lost"]).clip(lower=0)

    chase["current_run_rate"] = np.where(
        chase["legal_balls"] > 0,
        chase["current_score"] / (chase["legal_balls"] / 6.0),
        0.0,
    )
    chase["required_run_rate"] = np.where(
        chase["balls_remaining"] > 0,
        chase["runs_remaining"] / (chase["balls_remaining"] / 6.0),
        np.where(chase["runs_remaining"] > 0, 36.0, 0.0),
    )
    chase["rrr_minus_crr"] = chase["required_run_rate"] - chase["current_run_rate"]
    chase["is_ilt20"] = (chase["competition"] == "ilt20").astype(int)
    chase["is_bbl"] = (chase["competition"] == "bbl").astype(int)

    # Filter out post-win/all-out terminal balls where match is already decided
    chase = chase[(chase["balls_remaining"] > 0) & (chase["runs_remaining"] > 0) & (chase["wickets_remaining"] > 0)].copy()
    return chase


def train_live_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_live_chase_dataset()

    # Split strictly by chronological match_id order
    match_order = df[["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"])
    n_m = len(match_order)
    train_ids = set(match_order.iloc[: int(n_m * 0.70)]["match_id"])
    val_ids = set(match_order.iloc[int(n_m * 0.70) : int(n_m * 0.85)]["match_id"])
    test_ids = set(match_order.iloc[int(n_m * 0.85) :]["match_id"])

    train_df = df[df["match_id"].isin(train_ids)]
    val_df = df[df["match_id"].isin(val_ids)]
    test_df = df[df["match_id"].isin(test_ids)]

    print(f"Live Chase Dataset | Train balls: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_tr, y_tr = train_df[LIVE_FEATURES], train_df["chase_won"].values
    X_val, y_val = val_df[LIVE_FEATURES], val_df["chase_won"].values
    X_te, y_te = test_df[LIVE_FEATURES], test_df["chase_won"].values

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    probs = model.predict_proba(X_te)[:, 1]
    acc = float(accuracy_score(y_te, (probs >= 0.5).astype(int)))
    ll = float(log_loss(y_te, probs))
    bs = float(brier_score_loss(y_te, probs))

    print("\n=== LIVE CHASE WIN PROBABILITY (TEST BALLS) ===")
    print(f"LightGBM Live Model | Acc: {acc:.3f} | LogLoss: {ll:.4f} | Brier: {bs:.4f}")

    # Sanity check scenarios
    scenarios = pd.DataFrame(
        [
            {"runs_remaining": 45, "balls_remaining": 30, "wickets_remaining": 7, "current_run_rate": 8.0, "required_run_rate": 9.0, "rrr_minus_crr": 1.0, "last_18_runs": 28, "last_18_wkts": 0, "is_ilt20": 1, "is_bbl": 0},
            {"runs_remaining": 45, "balls_remaining": 30, "wickets_remaining": 2, "current_run_rate": 8.0, "required_run_rate": 9.0, "rrr_minus_crr": 1.0, "last_18_runs": 14, "last_18_wkts": 2, "is_ilt20": 1, "is_bbl": 0},
            {"runs_remaining": 18, "balls_remaining": 12, "wickets_remaining": 5, "current_run_rate": 8.5, "required_run_rate": 9.0, "rrr_minus_crr": 0.5, "last_18_runs": 26, "last_18_wkts": 1, "is_ilt20": 1, "is_bbl": 0},
        ]
    )
    sc_probs = model.predict_proba(scenarios)[:, 1]
    print("\nSanity-Check Chase Scenarios (ILT20):")
    print(f"  Need 45 off 30, 7 wkts left -> Chase Win Prob: {sc_probs[0]:.1%}")
    print(f"  Need 45 off 30, 2 wkts left -> Chase Win Prob: {sc_probs[1]:.1%}")
    print(f"  Need 18 off 12, 5 wkts left -> Chase Win Prob: {sc_probs[2]:.1%}")

    joblib.dump(model, MODELS_DIR / "lgbm_live_chase.joblib")
    with (MODELS_DIR / "live_chase_metrics.json").open("w") as f:
        json.dump({"features": LIVE_FEATURES, "test_acc": acc, "test_log_loss": ll, "test_brier": bs}, f, indent=2)


if __name__ == "__main__":
    train_live_model()