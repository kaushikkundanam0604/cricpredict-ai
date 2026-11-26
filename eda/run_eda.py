from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("eda/outputs")


def run_eda() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    matches = pd.read_parquet(PROCESSED_DIR / "match_features.parquet")
    deliveries = pd.read_parquet(PROCESSED_DIR / "deliveries.parquet")

    print("=== 1. DATASET OVERVIEW ===")
    print(f"Total decided matches: {len(matches):,}")
    print(f"Target balance (team_1_win rate): {matches['team_1_win'].mean():.3f}")
    print("\nMatches by competition:")
    print(matches["competition"].value_counts().to_string())

    print("\n=== 2. SCORING & CHASING BY COMPETITION ===")
    comp_summary = (
        matches.groupby("competition")
        .agg(
            matches=("match_id", "count"),
            avg_1st_inn_runs=("innings_1_runs", "mean"),
            median_1st_inn_runs=("innings_1_runs", "median"),
            chase_win_pct=("chase_won", "mean"),
            toss_field_pct=("toss_decision_field", "mean"),
        )
        .round(3)
    )
    print(comp_summary.to_string())

    print("\n=== 3. ILT20 VENUE BREAKDOWN ===")
    ilt = matches[matches["competition"] == "ilt20"]
    ilt_venues = (
        ilt.groupby("venue_clean")
        .agg(
            matches=("match_id", "count"),
            avg_1st_inn_runs=("innings_1_runs", "mean"),
            chase_win_pct=("chase_won", "mean"),
        )
        .sort_values("matches", ascending=False)
        .round(3)
    )
    print(ilt_venues.to_string())

    print("\n=== 4. POWERPLAY COLLAPSE IMPACT (1ST INNINGS) ===")
    pp = deliveries[(~deliveries["is_super_over"]) & (deliveries["innings"] == 1) & (deliveries["over"] < 6)]
    pp_wkts = (
        pp.groupby("match_id", as_index=False)["is_wicket"]
        .sum()
        .rename(columns={"is_wicket": "pp_wickets_inn1"})
    )
    collapse_df = matches.merge(pp_wkts, on="match_id", how="inner")
    collapse_df["bat_1st_won"] = (collapse_df["winner"] == collapse_df["innings_1_team"]).astype(int)
    collapse_df["pp_collapse_3plus"] = collapse_df["pp_wickets_inn1"] >= 3

    collapse_summary = (
        collapse_df.groupby("pp_collapse_3plus")
        .agg(
            matches=("match_id", "count"),
            avg_1st_inn_runs=("innings_1_runs", "mean"),
            bat_1st_win_pct=("bat_1st_won", "mean"),
        )
        .round(3)
    )
    print(collapse_summary.to_string())

    print("\n=== 5. FEATURE SIGNAL SANITY CHECK ===")
    has_form = matches.dropna(subset=["diff_win_rate_10", "diff_xi_bat_sr_10"])
    print(f"T1 win rate when diff_win_rate_10 > 0 : {has_form[has_form['diff_win_rate_10'] > 0]['team_1_win'].mean():.3f}")
    print(f"T1 win rate when diff_win_rate_10 < 0 : {has_form[has_form['diff_win_rate_10'] < 0]['team_1_win'].mean():.3f}")
    print(f"T1 win rate when diff_xi_bat_sr_10 > 0: {has_form[has_form['diff_xi_bat_sr_10'] > 0]['team_1_win'].mean():.3f}")
    print(f"T1 win rate when diff_xi_bat_sr_10 < 0: {has_form[has_form['diff_xi_bat_sr_10'] < 0]['team_1_win'].mean():.3f}")

    # --- Chart 1: 1st Innings Score Distribution by Competition ---
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=matches, x="competition", y="innings_1_runs", hue="competition", legend=False, ax=ax)
    ax.set_title("First-Innings Score Distribution by Competition")
    ax.set_xlabel("Competition")
    ax.set_ylabel("1st Innings Runs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_competition_scores.png", dpi=150)
    plt.close(fig)

    # --- Chart 2: ILT20 Venue Chasing Win % ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ilt_plot = ilt_venues.reset_index()
    sns.barplot(data=ilt_plot, x="venue_clean", y="chase_win_pct", hue="venue_clean", legend=False, ax=ax)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="50% Parity")
    ax.set_title("ILT20 Chasing Win Rate by Venue")
    ax.set_xlabel("Venue")
    ax.set_ylabel("Chasing Win Rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_ilt20_venue_chasing.png", dpi=150)
    plt.close(fig)

    # --- Chart 3: Powerplay Collapse Impact ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=collapse_df, x="pp_collapse_3plus", y="innings_1_runs", hue="pp_collapse_3plus", legend=False, ax=ax)
    ax.set_title("1st Innings Total: Normal Start vs. Powerplay Collapse (3+ Wickets)")
    ax.set_xlabel("3+ Wickets Lost in Powerplay (Overs 0–5)")
    ax.set_ylabel("Final 1st Innings Runs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_powerplay_collapse_impact.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 3 EDA charts to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_eda()