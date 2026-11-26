import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DATA_PATH = Path("data/processed/match_features.parquet")
MODELS_DIR = Path("models")

FEATURE_COLS = [
    "t1_win_rate_5",
    "t1_win_rate_10",
    "t2_win_rate_5",
    "t2_win_rate_10",
    "diff_win_rate_5",
    "diff_win_rate_10",
    "diff_avg_runs_for_10",
    "diff_xi_bat_sr_10",
    "diff_xi_bowl_econ_10",
    "diff_xi_total_exp",
    "t1_xi_bat_avg_10",
    "t2_xi_bat_avg_10",
    "venue_avg_1st_inn_runs",
    "venue_chase_win_rate",
    "t1_is_chasing",
    "t1_venue_chase_edge",
    "is_ilt20",
    "is_bbl",
]


def engineer_post_toss_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_ilt20"] = (out["competition"] == "ilt20").astype(int)
    out["is_bbl"] = (out["competition"] == "bbl").astype(int)

    # Team 1 chases if: (T1 won toss & fielded) OR (T2 won toss & batted)
    out["t1_is_chasing"] = (
        ((out["toss_winner_is_t1"] == 1) & (out["toss_decision_field"] == 1))
        | ((out["toss_winner_is_t1"] == 0) & (out["toss_decision_field"] == 0))
    ).astype(int)

    # Positive if T1 is in the favored role at this venue, negative if disadvantaged
    venue_chase = out["venue_chase_win_rate"].fillna(0.5)
    out["t1_venue_chase_edge"] = (out["t1_is_chasing"] * 2 - 1) * (venue_chase - 0.5)

    out["diff_win_rate_5"] = out["t1_win_rate_5"] - out["t2_win_rate_5"]
    out["diff_xi_total_exp"] = out["t1_xi_total_exp"] - out["t2_xi_total_exp"]
    return out


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    ll = float(log_loss(y_true, y_prob))
    bs = float(brier_score_loss(y_true, y_prob))
    print(f"{name:<26} | Acc: {acc:.3f} | LogLoss: {ll:.4f} | Brier: {bs:.4f}")
    return {"accuracy": acc, "log_loss": ll, "brier_score": bs}


def train_match_winner() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(DATA_PATH).sort_values(["match_date", "match_id"]).reset_index(drop=True)
    df = engineer_post_toss_features(raw_df)

    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:].reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLS], train_df["team_1_win"].values
    X_val, y_val = val_df[FEATURE_COLS], val_df["team_1_win"].values
    X_test, y_test = test_df[FEATURE_COLS], test_df["team_1_win"].values

    # Slightly upweight franchise T20s (ILT20 & BBL) so T20I mismatches don't dominate
    sample_weight = np.where(train_df["competition"].isin(["ilt20", "bbl"]), 2.0, 1.0)

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    print(f"Best iteration: {model.best_iteration_}")
    print("\n=== LIGHTGBM PERFORMANCE ===")
    val_metrics = evaluate("Validation (All T20s)", y_val, model.predict_proba(X_val)[:, 1])
    test_metrics = evaluate("Test (All T20s)", y_test, model.predict_proba(X_test)[:, 1])

    ilt_mask = (test_df["competition"] == "ilt20").values
    ilt_probs = model.predict_proba(X_test)[:, 1][ilt_mask]
    ilt_metrics = evaluate(f"Test (ILT20 only, n={ilt_mask.sum()})", y_test[ilt_mask], ilt_probs)

    # SHAP Explanation for the most recent ILT20 match in Test set
    explainer = shap.TreeExplainer(model)
    ilt_indices = test_df.index[test_df["competition"] == "ilt20"].tolist()
    if ilt_indices:
        idx = ilt_indices[-1]
        row_meta = test_df.loc[idx]
        row_X = X_test.iloc[[idx]]
        prob_t1 = float(model.predict_proba(row_X)[0, 1])

        shap_vals = explainer.shap_values(row_X)
        # Handle both 2D array and list-of-arrays return formats across SHAP versions
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        contrib = pd.Series(sv, index=FEATURE_COLS).sort_values()

        print(f"\n=== SAMPLE EXPLAINABLE PREDICTION (ILT20: {row_meta['match_date'].date()}) ===")
        print(f"Match   : {row_meta['team_1']} vs {row_meta['team_2']} at {row_meta['venue_clean']}")
        print(f"Actual  : {row_meta['winner']} won")
        print(f"Pred    : {row_meta['team_1']} {prob_t1:.1%} | {row_meta['team_2']} {1 - prob_t1:.1%}")
        print("\nTop factors HELPING Team 1:")
        print(contrib.tail(4)[::-1].round(4).to_string())
        print("\nTop factors HURTING Team 1:")
        print(contrib.head(4).round(4).to_string())

    joblib.dump(model, MODELS_DIR / "lgbm_match_winner.joblib")
    with (MODELS_DIR / "match_winner_metrics.json").open("w") as f:
        json.dump(
            {"features": FEATURE_COLS, "val": val_metrics, "test": test_metrics, "test_ilt20": ilt_metrics},
            f,
            indent=2,
        )
    print("\nSaved LightGBM model to models/lgbm_match_winner.joblib")


if __name__ == "__main__":
    train_match_winner()