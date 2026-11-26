import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/match_features.parquet")
MODELS_DIR = Path("models")

FEATURE_COLS = [
    "t1_win_rate_5",
    "t1_win_rate_10",
    "t2_win_rate_5",
    "t2_win_rate_10",
    "diff_win_rate_10",
    "diff_avg_runs_for_10",
    "t1_xi_bat_sr_10",
    "t2_xi_bat_sr_10",
    "diff_xi_bat_sr_10",
    "t1_xi_bowl_econ_10",
    "t2_xi_bowl_econ_10",
    "diff_xi_bowl_econ_10",
    "t1_xi_total_exp",
    "t2_xi_total_exp",
    "venue_avg_1st_inn_runs",
    "venue_chase_win_rate",
    "toss_winner_is_t1",
    "toss_decision_field",
    "is_ilt20",
    "is_bbl",
]


def evaluate_probs(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    print(
        f"{name:<26} | Acc: {metrics['accuracy']:.3f} | "
        f"LogLoss: {metrics['log_loss']:.4f} | Brier: {metrics['brier_score']:.4f} | F1: {metrics['f1']:.3f}"
    )
    return metrics


def train_baselines() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH).sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # One-hot competition indicators
    df["is_ilt20"] = (df["competition"] == "ilt20").astype(int)
    df["is_bbl"] = (df["competition"] == "bbl").astype(int)

    # Chronological 70% / 15% / 15% split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("=== CHRONOLOGICAL SPLITS ===")
    print(f"Train: {len(train_df):,} matches ({train_df['match_date'].min().date()} -> {train_df['match_date'].max().date()})")
    print(f"Val  : {len(val_df):,} matches ({val_df['match_date'].min().date()} -> {val_df['match_date'].max().date()})")
    print(f"Test : {len(test_df):,} matches ({test_df['match_date'].min().date()} -> {test_df['match_date'].max().date()})")
    print(f"ILT20 matches in Test set: {(test_df['competition'] == 'ilt20').sum()}\n")

    X_train, y_train = train_df[FEATURE_COLS], train_df["team_1_win"].values
    X_val, y_val = val_df[FEATURE_COLS], val_df["team_1_win"].values
    X_test, y_test = test_df[FEATURE_COLS], test_df["team_1_win"].values

    # 1. Logistic Regression Pipeline
    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
        ]
    )
    logreg.fit(X_train, y_train)

    # 2. Random Forest Pipeline
    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)

    print("=== VALIDATION SET PERFORMANCE ===")
    evaluate_probs("Coin-Flip Baseline", y_val, np.full(len(y_val), y_train.mean()))
    val_lr = evaluate_probs("Logistic Regression", y_val, logreg.predict_proba(X_val)[:, 1])
    val_rf = evaluate_probs("Random Forest", y_val, rf.predict_proba(X_val)[:, 1])

    print("\n=== TEST SET PERFORMANCE (ALL T20s) ===")
    evaluate_probs("Coin-Flip Baseline", y_test, np.full(len(y_test), y_train.mean()))
    test_lr = evaluate_probs("Logistic Regression", y_test, logreg.predict_proba(X_test)[:, 1])
    test_rf = evaluate_probs("Random Forest", y_test, rf.predict_proba(X_test)[:, 1])

    # Evaluate specifically on ILT20 matches in the Test set
    ilt_mask = (test_df["competition"] == "ilt20").values
    if ilt_mask.sum() > 0:
        print(f"\n=== TEST SET PERFORMANCE (ILT20 ONLY, n={ilt_mask.sum()}) ===")
        evaluate_probs("Logistic Regression", y_test[ilt_mask], logreg.predict_proba(X_test)[ilt_mask, 1])
        evaluate_probs("Random Forest", y_test[ilt_mask], rf.predict_proba(X_test)[ilt_mask, 1])

    # Inspect Logistic Regression feature coefficients
    coefs = pd.Series(logreg.named_steps["clf"].coef_[0], index=FEATURE_COLS).sort_values()
    print("\n=== LOGISTIC REGRESSION TOP COEFFICIENTS (Standardized) ===")
    print(pd.concat([coefs.head(5), coefs.tail(5)]).round(4).to_string())

    # Save best baseline by validation Log Loss
    best_model = logreg if val_lr["log_loss"] <= val_rf["log_loss"] else rf
    best_name = "logistic_regression" if val_lr["log_loss"] <= val_rf["log_loss"] else "random_forest"

    joblib.dump(best_model, MODELS_DIR / "baseline_match_winner.joblib")
    with (MODELS_DIR / "baseline_metrics.json").open("w") as f:
        json.dump(
            {"best_model": best_name, "features": FEATURE_COLS, "test_logreg": test_lr, "test_rf": test_rf},
            f,
            indent=2,
        )
    print(f"\nSaved best baseline model ({best_name}) to models/baseline_match_winner.joblib")


if __name__ == "__main__":
    train_baselines()