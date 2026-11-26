import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

PLAYER_STATS_PATH = Path("data/processed/player_match_stats.parquet")
MATCH_FEATURES_PATH = Path("data/processed/match_features.parquet")
PLAYERS_MODELS_DIR = Path("models/players")

BAT_FEATURES = [
    "roster_order",
    "pre_bat_innings",
    "pre_bat_avg_10",
    "pre_bat_sr_10",
    "venue_avg_1st_inn_runs",
    "opp_bowl_econ_10",
    "is_ilt20",
    "is_bbl",
]

BOWL_FEATURES = [
    "pre_bowl_innings",
    "pre_bowl_econ_10",
    "pre_bowl_wkts_avg_10",
    "venue_avg_1st_inn_runs",
    "opp_bat_sr_10",
    "is_ilt20",
    "is_bbl",
]


def prepare_player_dataset() -> pd.DataFrame:
    ps = pd.read_parquet(PLAYER_STATS_PATH)
    mf = pd.read_parquet(MATCH_FEATURES_PATH)[
        [
            "match_id",
            "team_1",
            "team_2",
            "venue_clean",
            "venue_avg_1st_inn_runs",
            "t1_xi_bat_sr_10",
            "t2_xi_bat_sr_10",
            "t1_xi_bowl_econ_10",
            "t2_xi_bowl_econ_10",
        ]
    ]

    df = ps.merge(mf, on="match_id", how="inner")
    is_t1 = df["team"] == df["team_1"]
    df["opp_bowl_econ_10"] = np.where(is_t1, df["t2_xi_bowl_econ_10"], df["t1_xi_bowl_econ_10"])
    df["opp_bat_sr_10"] = np.where(is_t1, df["t2_xi_bat_sr_10"], df["t1_xi_bat_sr_10"])
    df["is_ilt20"] = (df["competition"] == "ilt20").astype(int)
    df["is_bbl"] = (df["competition"] == "bbl").astype(int)

    df = df.sort_values(["match_date", "match_id", "player_id"]).reset_index(drop=True)
    return df


def split_chrono(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    return df.iloc[: int(n * 0.70)], df.iloc[int(n * 0.70) : int(n * 0.85)], df.iloc[int(n * 0.85) :].reset_index(drop=True)


def fit_regressor(X_tr, y_tr, X_val, y_val) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def fit_classifier(X_tr, y_tr, X_val, y_val) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def train_player_models() -> None:
    PLAYERS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_player_dataset()
    metrics: dict[str, dict[str, float]] = {}

    # === 1. BATTING MODELS ===
    bat_df = df[df["did_bat"]].copy()
    bat_df["scored_20plus"] = (bat_df["runs_scored"] >= 20).astype(int)
    bat_df["scored_30plus"] = (bat_df["runs_scored"] >= 30).astype(int)

    b_tr, b_val, b_te = split_chrono(bat_df)
    Xb_tr, Xb_val, Xb_te = b_tr[BAT_FEATURES], b_val[BAT_FEATURES], b_te[BAT_FEATURES]

    bat_runs_m = fit_regressor(Xb_tr, b_tr["runs_scored"], Xb_val, b_val["runs_scored"])
    bat_20_m = fit_classifier(Xb_tr, b_tr["scored_20plus"], Xb_val, b_val["scored_20plus"])
    bat_30_m = fit_classifier(Xb_tr, b_tr["scored_30plus"], Xb_val, b_val["scored_30plus"])

    pred_runs = bat_runs_m.predict(Xb_te)
    pred_20 = bat_20_m.predict_proba(Xb_te)[:, 1]
    pred_30 = bat_30_m.predict_proba(Xb_te)[:, 1]

    print("=== BATTING MODELS (TEST SET) ===")
    print(f"Expected Runs | Naive MAE: {mean_absolute_error(b_te['runs_scored'], np.full(len(b_te), b_tr['runs_scored'].mean())):.2f} | "
          f"LGBM MAE: {mean_absolute_error(b_te['runs_scored'], pred_runs):.2f}")
    print(f"20+ Runs Prob | LogLoss: {log_loss(b_te['scored_20plus'], pred_20):.4f} | ROC-AUC: {roc_auc_score(b_te['scored_20plus'], pred_20):.3f}")
    print(f"30+ Runs Prob | LogLoss: {log_loss(b_te['scored_30plus'], pred_30):.4f} | ROC-AUC: {roc_auc_score(b_te['scored_30plus'], pred_30):.3f}")

    metrics["batting"] = {
        "runs_mae": float(mean_absolute_error(b_te["runs_scored"], pred_runs)),
        "p20_roc_auc": float(roc_auc_score(b_te["scored_20plus"], pred_20)),
        "p30_roc_auc": float(roc_auc_score(b_te["scored_30plus"], pred_30)),
    }

    # === 2. BOWLING MODELS ===
    bowl_df = df[df["did_bowl"] & (df["legal_balls_bowled"] >= 6)].copy()
    bowl_df["took_1plus_wkt"] = (bowl_df["wickets_taken"] >= 1).astype(int)
    bowl_df["match_economy"] = bowl_df["runs_conceded"] / (bowl_df["legal_balls_bowled"] / 6.0)

    w_tr, w_val, w_te = split_chrono(bowl_df)
    Xw_tr, Xw_val, Xw_te = w_tr[BOWL_FEATURES], w_val[BOWL_FEATURES], w_te[BOWL_FEATURES]

    bowl_wkts_m = fit_regressor(Xw_tr, w_tr["wickets_taken"], Xw_val, w_val["wickets_taken"])
    bowl_1p_m = fit_classifier(Xw_tr, w_tr["took_1plus_wkt"], Xw_val, w_val["took_1plus_wkt"])
    bowl_econ_m = fit_regressor(Xw_tr, w_tr["match_economy"], Xw_val, w_val["match_economy"])

    pred_wkts = bowl_wkts_m.predict(Xw_te)
    pred_1p = bowl_1p_m.predict_proba(Xw_te)[:, 1]
    pred_econ = bowl_econ_m.predict(Xw_te)

    print("\n=== BOWLING MODELS (TEST SET) ===")
    print(f"Expected Wkts | Naive MAE: {mean_absolute_error(w_te['wickets_taken'], np.full(len(w_te), w_tr['wickets_taken'].mean())):.3f} | "
          f"LGBM MAE: {mean_absolute_error(w_te['wickets_taken'], pred_wkts):.3f}")
    print(f"1+ Wicket Prob| LogLoss: {log_loss(w_te['took_1plus_wkt'], pred_1p):.4f} | ROC-AUC: {roc_auc_score(w_te['took_1plus_wkt'], pred_1p):.3f}")
    print(f"Expected Econ | Naive MAE: {mean_absolute_error(w_te['match_economy'], np.full(len(w_te), w_tr['match_economy'].mean())):.2f} | "
          f"LGBM MAE: {mean_absolute_error(w_te['match_economy'], pred_econ):.2f}")

    metrics["bowling"] = {
        "wkts_mae": float(mean_absolute_error(w_te["wickets_taken"], pred_wkts)),
        "p1w_roc_auc": float(roc_auc_score(w_te["took_1plus_wkt"], pred_1p)),
        "econ_mae": float(mean_absolute_error(w_te["match_economy"], pred_econ)),
    }

    # Save all 6 models
    joblib.dump(bat_runs_m, PLAYERS_MODELS_DIR / "bat_runs.joblib")
    joblib.dump(bat_20_m, PLAYERS_MODELS_DIR / "bat_20plus.joblib")
    joblib.dump(bat_30_m, PLAYERS_MODELS_DIR / "bat_30plus.joblib")
    joblib.dump(bowl_wkts_m, PLAYERS_MODELS_DIR / "bowl_wickets.joblib")
    joblib.dump(bowl_1p_m, PLAYERS_MODELS_DIR / "bowl_1plus_wkt.joblib")
    joblib.dump(bowl_econ_m, PLAYERS_MODELS_DIR / "bowl_economy.joblib")

    with Path("models/player_models_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved 6 player models to models/players/")


if __name__ == "__main__":
    train_player_models()