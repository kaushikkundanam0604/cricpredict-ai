from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def build_team_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create strictly pre-match rolling team form from chronological match history."""
    t1 = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "match_date": matches["match_date"],
            "team": matches["team_1"],
            "won": (matches["winner"] == matches["team_1"]).astype(int),
            "runs_for": np.where(
                matches["innings_1_team"] == matches["team_1"],
                matches["innings_1_runs"],
                matches["innings_2_runs"],
            ),
            "runs_against": np.where(
                matches["innings_1_team"] == matches["team_1"],
                matches["innings_2_runs"],
                matches["innings_1_runs"],
            ),
        }
    )
    t2 = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "match_date": matches["match_date"],
            "team": matches["team_2"],
            "won": (matches["winner"] == matches["team_2"]).astype(int),
            "runs_for": np.where(
                matches["innings_1_team"] == matches["team_2"],
                matches["innings_1_runs"],
                matches["innings_2_runs"],
            ),
            "runs_against": np.where(
                matches["innings_1_team"] == matches["team_2"],
                matches["innings_2_runs"],
                matches["innings_1_runs"],
            ),
        }
    )

    team_df = pd.concat([t1, t2], ignore_index=True)
    team_df = team_df.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)

    g = team_df.groupby("team")
    team_df["matches_prior"] = g.cumcount()
    team_df["win_rate_5"] = g["won"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_df["win_rate_10"] = g["won"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    team_df["avg_runs_for_10"] = g["runs_for"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    team_df["avg_runs_against_10"] = g["runs_against"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )

    return team_df[
        [
            "match_id",
            "team",
            "matches_prior",
            "win_rate_5",
            "win_rate_10",
            "avg_runs_for_10",
            "avg_runs_against_10",
        ]
    ]


def build_xi_strength(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre-match Playing XI batting and bowling strength per (match_id, team)."""
    p = player_stats.copy()
    # Use experienced players (>= 3 prior innings) when available to avoid 1-match noise
    p["exp_bat_sr"] = np.where(p["pre_bat_innings"] >= 3, p["pre_bat_sr_10"], np.nan)
    p["exp_bat_avg"] = np.where(p["pre_bat_innings"] >= 3, p["pre_bat_avg_10"], np.nan)
    p["exp_bowl_econ"] = np.where(p["pre_bowl_innings"] >= 3, p["pre_bowl_econ_10"], np.nan)
    p["exp_bowl_wkts"] = np.where(p["pre_bowl_innings"] >= 3, p["pre_bowl_wkts_avg_10"], np.nan)
    p["total_prior_inn"] = p["pre_bat_innings"] + p["pre_bowl_innings"]

    xi = (
        p.groupby(["match_id", "team"], as_index=False)
        .agg(
            xi_bat_sr_10=("exp_bat_sr", "mean"),
            xi_bat_avg_10=("exp_bat_avg", "mean"),
            xi_bowl_econ_10=("exp_bowl_econ", "mean"),
            xi_bowl_wkts_10=("exp_bowl_wkts", "mean"),
            xi_total_exp=("total_prior_inn", "sum"),
        )
    )
    return xi


def build_match_dataset() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    player_stats = pd.read_parquet(PROCESSED_DIR / "player_match_stats.parquet")

    # Keep matches with a decided winner among team_1 or team_2
    valid = matches[
        matches["winner"].notna()
        & ((matches["winner"] == matches["team_1"]) | (matches["winner"] == matches["team_2"]))
    ].copy()

    valid = valid.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Normalize venue names and compute strictly pre-match venue stats
    valid["venue_clean"] = valid["venue"].fillna("Unknown").str.split(",").str[0].str.strip()
    valid["chase_won"] = (valid["winner"] == valid["innings_2_team"]).astype(float)

    valid = valid.sort_values(["venue_clean", "match_date", "match_id"]).reset_index(drop=True)
    gv = valid.groupby("venue_clean")
    valid["venue_matches_prior"] = gv.cumcount()
    valid["venue_avg_1st_inn_runs"] = gv["innings_1_runs"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    valid["venue_chase_win_rate"] = gv["chase_won"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Re-sort chronologically by match date
    valid = valid.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Merge Team 1 and Team 2 rolling form
    team_feats = build_team_rolling_features(valid)
    valid = valid.merge(
        team_feats.add_prefix("t1_"),
        left_on=["match_id", "team_1"],
        right_on=["t1_match_id", "t1_team"],
        how="left",
    ).drop(columns=["t1_match_id", "t1_team"])

    valid = valid.merge(
        team_feats.add_prefix("t2_"),
        left_on=["match_id", "team_2"],
        right_on=["t2_match_id", "t2_team"],
        how="left",
    ).drop(columns=["t2_match_id", "t2_team"])

    # Merge Playing XI strength for Team 1 and Team 2
    xi_feats = build_xi_strength(player_stats)
    valid = valid.merge(
        xi_feats.add_prefix("t1_"),
        left_on=["match_id", "team_1"],
        right_on=["t1_match_id", "t1_team"],
        how="left",
    ).drop(columns=["t1_match_id", "t1_team"])

    valid = valid.merge(
        xi_feats.add_prefix("t2_"),
        left_on=["match_id", "team_2"],
        right_on=["t2_match_id", "t2_team"],
        how="left",
    ).drop(columns=["t2_match_id", "t2_team"])

    # Differential features (Team 1 minus Team 2)
    valid["diff_win_rate_10"] = valid["t1_win_rate_10"] - valid["t2_win_rate_10"]
    valid["diff_avg_runs_for_10"] = valid["t1_avg_runs_for_10"] - valid["t2_avg_runs_for_10"]
    valid["diff_xi_bat_sr_10"] = valid["t1_xi_bat_sr_10"] - valid["t2_xi_bat_sr_10"]
    valid["diff_xi_bowl_econ_10"] = valid["t1_xi_bowl_econ_10"] - valid["t2_xi_bowl_econ_10"]

    # Toss & target columns
    valid["toss_winner_is_t1"] = (valid["toss_winner"] == valid["team_1"]).astype(int)
    valid["toss_decision_field"] = (valid["toss_decision"] == "field").astype(int)
    valid["team_1_win"] = (valid["winner"] == valid["team_1"]).astype(int)

    out_path = PROCESSED_DIR / "match_features.parquet"
    valid.to_parquet(out_path, index=False)

    print(f"Saved {out_path} ({len(valid):,} decided matches, {valid.shape[1]} columns)")
    print("\nDecided matches by competition:")
    print(valid["competition"].value_counts().to_string())
    print(f"\nILT20 date range: {valid[valid['competition']=='ilt20']['match_date'].min().date()} "
          f"to {valid[valid['competition']=='ilt20']['match_date'].max().date()}")
    print("\nMissing values in key differential features (expected only for debut teams/players):")
    print(valid[["diff_win_rate_10", "diff_xi_bat_sr_10", "venue_avg_1st_inn_runs"]].isna().sum().to_string())


if __name__ == "__main__":
    build_match_dataset()