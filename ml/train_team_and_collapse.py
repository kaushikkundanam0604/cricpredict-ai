import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
)

MATCH_FEATURES_PATH = Path("data/processed/match_features.parquet")
DELIVERIES_PATH = Path("data/processed/deliveries.parquet")
MODELS_DIR = Path("models")

INNINGS_FEATURES = [
    "bat_win_rate_10",
    "bat_avg_runs_for_10",
    "bat_xi_bat_sr_10",
    "bat_xi_bat_avg_10",
    "bat_xi_total_exp",
    "bowl_win_rate_10",
    "bowl_avg_runs_against_10",
    "bowl_xi_bowl_econ_10",
    "bowl_xi_bowl_wkts_10",
    "bowl_xi_total_exp",
    "venue_avg_1st_inn_runs",
    "venue_chase_win_rate",
    "is_ilt20",
    "is_bbl",
]


def prepare_first_innings_dataset() -> pd.DataFrame:
    matches = pd.read_parquet(MATCH_FEATURES_PATH)
    deliveries = pd.read_parquet(DELIVERIES_PATH)

    # Compute 1st-innings powerplay wickets (overs 0-5)
    pp = deliveries[
        (~deliveries["is_super_over"])
        & (deliveries["innings"] == 1)
        & (deliveries["over"] < 6)
    ]
    pp_wkts = (
        pp.groupby("match_id", as_index=False)["is_wicket"]
        .sum()
        .rename(columns={"is_wicket": "pp_wickets_inn1"})
    )

    df = matches.merge(pp_wkts, on="match_id", how="inner")
    # Keep standard 20-over 1st innings (exclude rain-shortened < 15 overs)
    df = df[df["innings_1_legal_balls"] >= 90].copy()
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    t1_batted_first = df["innings_1_team"] == df["team_1"]

    df["bat_win_rate_10"] = np.where(t1_batted_first, df["t1_win_rate_10"], df["t2_win_rate_10"])
    df["bat_avg_runs_for_10"] = np.where(t1_batted_first, df["t1_avg_runs_for_10"], df["t2_avg_runs_for_10"])
    df["bat_xi_bat_sr_10"] = np.where(t1_batted_first, df["t1_xi_bat_sr_10"], df["t2_xi_bat_sr_10"])
    df["bat_xi_bat_avg_10"] = np.where(t1_batted_first, df["t1_xi_bat_avg_10"], df["t2_xi_bat_avg_10"])
    df["bat_xi_total_exp"] = np.where(t1_batted_first, df["t1_xi_total_exp"], df["t2_xi_total_exp"])

    df["bowl_win_rate_10"] = np.where(t1_batted_first, df["t2_win_rate_10"], df["t1_win_rate_10"])
    df["bowl_avg_runs_against_10"] = np.where(t1_batted_first, df["t2_avg_runs_against_10"], df["t1_avg_runs_against_10"])
    df["bowl_xi_bowl_econ_10"] = np.where(t1_batted_first, df["t2_xi_bowl_econ_10"], df["t1_xi_bowl_econ_10"])
    df["bowl_xi_bowl_wkts_10"] = np.where(t1_batted_first, df["t2_xi_bowl_wkts_10"], df["t1_xi_bowl_wkts_10"])
    df["bowl_xi_total_exp"] = np.where(t1_batted_first, df["t2_xi_total_exp"], df["t1_xi_total_exp"])

    df["is_ilt20"] = (df["competition"] == "ilt20").astype(int)
    df["is_bbl"] = (df["competition"] == "bbl").astype(int)
    df["pp_collapse_3plus"] = (df["pp_wickets_inn1"] >= 3).astype(int)
    return df


def train_team_and_collapse_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_first_innings_dataset()

    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:].reset_index(drop=True)

    X_train = train_df[INNINGS_FEATURES]
    X_val = val_df[INNINGS_FEATURES]
    X_test = test_df[INNINGS_FEATURES]

    # --- 1. 1st Innings Score Regressor ---
    y_score_train = train_df["innings_1_runs"].values
    y_score_val = val_df["innings_1_runs"].values
    y_score_test = test_df["innings_1_runs"].values

    score_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    score_model.fit(
        X_train,
        y_score_train,
        eval_set=[(X_val, y_score_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    pred_score_test = score_model.predict(X_test)
    naive_score_pred = np.full(len(y_score_test), y_score_train.mean())

    print("=== 1ST INNINGS SCORE PREDICTION (TEST SET) ===")
    print(f"Naive Mean Baseline | MAE: {mean_absolute_error(y_score_test, naive_score_pred):.2f} runs | "
          f"RMSE: {root_mean_squared_error(y_score_test, naive_score_pred):.2f}")
    score_mae = float(mean_absolute_error(y_score_test, pred_score_test))
    score_rmse = float(root_mean_squared_error(y_score_test, pred_score_test))
    print(f"LightGBM Regressor  | MAE: {score_mae:.2f} runs | RMSE: {score_rmse:.2f}")

    # --- 2. Powerplay Collapse Classifier (3+ Wickets in Overs 0-5) ---
    y_col_train = train_df["pp_collapse_3plus"].values
    y_col_val = val_df["pp_collapse_3plus"].values
    y_col_test = test_df["pp_collapse_3plus"].values

    collapse_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=35,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    collapse_model.fit(
        X_train,
        y_col_train,
        eval_set=[(X_val, y_col_val)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    pred_col_prob = collapse_model.predict_proba(X_test)[:, 1]
    naive_col_prob = np.full(len(y_col_test), y_col_train.mean())

    print("\n=== POWERPLAY COLLAPSE PREDICTION (TEST SET) ===")
    print(f"Base-Rate Baseline  | LogLoss: {log_loss(y_col_test, naive_col_prob):.4f} | "
          f"Brier: {brier_score_loss(y_col_test, naive_col_prob):.4f}")
    col_ll = float(log_loss(y_col_test, pred_col_prob))
    col_bs = float(brier_score_loss(y_col_test, pred_col_prob))
    col_auc = float(roc_auc_score(y_col_test, pred_col_prob))
    print(f"LightGBM Classifier | LogLoss: {col_ll:.4f} | Brier: {col_bs:.4f} | ROC-AUC: {col_auc:.3f}")

    # Sample ILT20 output
    ilt_idx = test_df.index[test_df["competition"] == "ilt20"].tolist()
    if ilt_idx:
        i = ilt_idx[-1]
        row = test_df.loc[i]
        print(f"\n=== SAMPLE ILT20 INNINGS FORECAST ({row['match_date'].date()}) ===")
        print(f"Batting 1st: {row['innings_1_team']} vs {row['innings_2_team']} ({row['venue_clean']})")
        print(f"Expected 1st Inn Score: {pred_score_test[i]:.1f} runs (Actual: {row['innings_1_runs']})")
        print(f"Powerplay Collapse Prob (3+ wkts): {pred_col_prob[i]:.1%} (Actual PP wkts: {row['pp_wickets_inn1']})")

    joblib.dump(score_model, MODELS_DIR / "lgbm_first_innings_score.joblib")
    joblib.dump(collapse_model, MODELS_DIR / "lgbm_pp_collapse.joblib")
    with (MODELS_DIR / "team_collapse_metrics.json").open("w") as f:
        json.dump(
            {
                "features": INNINGS_FEATURES,
                "score_mae": score_mae,
                "score_rmse": score_rmse,
                "collapse_log_loss": col_ll,
                "collapse_brier": col_bs,
                "collapse_roc_auc": col_auc,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    train_team_and_collapse_models()