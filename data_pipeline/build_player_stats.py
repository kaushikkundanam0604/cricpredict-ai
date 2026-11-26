from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def build_player_match_stats() -> None:
    deliveries = pd.read_parquet(PROCESSED_DIR / "deliveries.parquet")
    match_players = pd.read_parquet(PROCESSED_DIR / "match_players.parquet")

    # Exclude Super Over deliveries from standard 20-over player stats
    reg = deliveries[~deliveries["is_super_over"]].copy()
    reg["ball_faced"] = (~reg["is_wide"]).astype(int)
    reg["is_four"] = (reg["runs_batter"] == 4).astype(int)
    reg["is_six"] = (reg["runs_batter"] == 6).astype(int)
    reg["bat_dot"] = ((~reg["is_wide"]) & (reg["runs_batter"] == 0)).astype(int)
    reg["bowl_dot"] = (reg["is_legal_ball"] & (reg["runs_total"] == 0)).astype(int)

    reg["is_pp"] = reg["over"] < 6
    reg["is_mid"] = (reg["over"] >= 6) & (reg["over"] < 15)
    reg["is_death"] = reg["over"] >= 15

    # --- 1. Batting aggregation per (match_id, player_id) ---
    batting = (
        reg.groupby(["match_id", "batter_id"], as_index=False)
        .agg(
            runs_scored=("runs_batter", "sum"),
            balls_faced=("ball_faced", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            bat_dots=("bat_dot", "sum"),
        )
        .rename(columns={"batter_id": "player_id"})
    )

    # Attribute dismissals accurately (handles non-striker run outs)
    dismissals = (
        reg[reg["is_wicket"] & reg["player_out_id"].notna()]
        .groupby(["match_id", "player_out_id"], as_index=False)
        .size()
        .rename(columns={"player_out_id": "player_id", "size": "dismissals"})
    )
    batting = batting.merge(dismissals, on=["match_id", "player_id"], how="left")
    batting["dismissals"] = batting["dismissals"].fillna(0).astype(int)
    batting["did_bat"] = (batting["balls_faced"] > 0) | (batting["dismissals"] > 0)

    # --- 2. Bowling aggregation per (match_id, player_id) ---
    reg["pp_runs"] = np.where(reg["is_pp"], reg["runs_bowler"], 0)
    reg["pp_balls"] = np.where(reg["is_pp"] & reg["is_legal_ball"], 1, 0)
    reg["death_runs"] = np.where(reg["is_death"], reg["runs_bowler"], 0)
    reg["death_balls"] = np.where(reg["is_death"] & reg["is_legal_ball"], 1, 0)

    bowling = (
        reg.groupby(["match_id", "bowler_id"], as_index=False)
        .agg(
            legal_balls_bowled=("is_legal_ball", "sum"),
            runs_conceded=("runs_bowler", "sum"),
            wickets_taken=("is_bowler_wicket", "sum"),
            bowl_dots=("bowl_dot", "sum"),
            pp_runs=("pp_runs", "sum"),
            pp_balls=("pp_balls", "sum"),
            death_runs=("death_runs", "sum"),
            death_balls=("death_balls", "sum"),
        )
        .rename(columns={"bowler_id": "player_id"})
    )
    bowling["did_bowl"] = bowling["legal_balls_bowled"] > 0

    # --- 3. Merge onto Playing XI / Squad table ---
    df = match_players.merge(batting, on=["match_id", "player_id"], how="left")
    df = df.merge(bowling, on=["match_id", "player_id"], how="left")

    df["did_bat"] = df["did_bat"].fillna(False)
    df["did_bowl"] = df["did_bowl"].fillna(False)

    num_cols = [
        "runs_scored", "balls_faced", "fours", "sixes", "bat_dots", "dismissals",
        "legal_balls_bowled", "runs_conceded", "wickets_taken", "bowl_dots",
        "pp_runs", "pp_balls", "death_runs", "death_balls",
    ]
    df[num_cols] = df[num_cols].fillna(0).astype(int)

    df = df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)

    # --- 4. Compute Leakage-Free Pre-Match Rolling Stats ---
    # Batting rolling stats (computed only on innings where player actually batted)
    bat_only = df[df["did_bat"]].copy()
    g_bat = bat_only.groupby("player_id")
    bat_only["post_bat_innings"] = g_bat.cumcount() + 1
    bat_only["post_bat_runs_10"] = g_bat["runs_scored"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bat_only["post_bat_balls_10"] = g_bat["balls_faced"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bat_only["post_bat_outs_10"] = g_bat["dismissals"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bat_only["post_bat_avg_10"] = np.where(
        bat_only["post_bat_outs_10"] > 0,
        bat_only["post_bat_runs_10"] / bat_only["post_bat_outs_10"],
        bat_only["post_bat_runs_10"],
    )
    bat_only["post_bat_sr_10"] = np.where(
        bat_only["post_bat_balls_10"] > 0,
        (bat_only["post_bat_runs_10"] / bat_only["post_bat_balls_10"]) * 100.0,
        np.nan,
    )

    bat_post_cols = ["post_bat_innings", "post_bat_avg_10", "post_bat_sr_10"]
    df = df.merge(bat_only[["match_id", "player_id"] + bat_post_cols], on=["match_id", "player_id"], how="left")

    # Bowling rolling stats (computed only on innings where player actually bowled)
    bowl_only = df[df["did_bowl"]].copy()
    g_bowl = bowl_only.groupby("player_id")
    bowl_only["post_bowl_innings"] = g_bowl.cumcount() + 1
    bowl_only["post_bowl_runs_10"] = g_bowl["runs_conceded"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bowl_only["post_bowl_balls_10"] = g_bowl["legal_balls_bowled"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bowl_only["post_bowl_wkts_10"] = g_bowl["wickets_taken"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    bowl_only["post_bowl_econ_10"] = np.where(
        bowl_only["post_bowl_balls_10"] > 0,
        (bowl_only["post_bowl_runs_10"] / bowl_only["post_bowl_balls_10"]) * 6.0,
        np.nan,
    )
    bowl_only["post_bowl_wkts_avg_10"] = g_bowl["wickets_taken"].transform(lambda s: s.rolling(10, min_periods=1).mean())

    bowl_post_cols = ["post_bowl_innings", "post_bowl_econ_10", "post_bowl_wkts_avg_10"]
    df = df.merge(bowl_only[["match_id", "player_id"] + bowl_post_cols], on=["match_id", "player_id"], how="left")

    # Forward-fill then shift(1) per player so every row has strictly PRE-MATCH stats
    all_post_cols = bat_post_cols + bowl_post_cols
    df[all_post_cols] = df.groupby("player_id")[all_post_cols].ffill()
    for col in all_post_cols:
        pre_col = col.replace("post_", "pre_")
        df[pre_col] = df.groupby("player_id")[col].shift(1)

    df = df.drop(columns=all_post_cols)
    df["pre_bat_innings"] = df["pre_bat_innings"].fillna(0).astype(int)
    df["pre_bowl_innings"] = df["pre_bowl_innings"].fillna(0).astype(int)

    out_path = PROCESSED_DIR / "player_match_stats.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved {out_path} ({len(df):,} rows, {out_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Unique players tracked: {df['player_id'].nunique():,}")

    # Sanity check on a well-known ILT20 match (latest match)
    latest_match = df[df["competition"] == "ilt20"]["match_id"].max()
    sample = df[df["match_id"] == latest_match][
        ["team", "player_name", "runs_scored", "balls_faced", "pre_bat_innings", "pre_bat_sr_10", "pre_bowl_econ_10"]
    ].head(8)
    print(f"\nSample pre-match vs actual stats (ILT20 match {latest_match}):")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    build_player_match_stats()