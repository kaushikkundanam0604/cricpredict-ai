import json
from pathlib import Path

import pandas as pd

RAW_DATASETS = {
    "t20i": Path("data/raw/t20s_male_json"),
    "ilt20": Path("data/raw/ilt20_json"),
    "bbl": Path("data/raw/bbl_json"),
}

PROCESSED_DIR = Path("data/processed")

NON_BOWLER_WICKETS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "handled the ball",
    "hit the ball twice",
    "timed out",
}


def parse_match_file(file_path: Path, competition: str) -> tuple[dict, list[dict], list[dict]]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    match_id = file_path.stem
    info = data.get("info", {})
    registry_people = info.get("registry", {}).get("people", {})

    dates = info.get("dates", [])
    match_date = dates[0] if dates else None

    teams = info.get("teams", [])
    team_1 = teams[0] if len(teams) > 0 else None
    team_2 = teams[1] if len(teams) > 1 else None

    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    outcome_by = outcome.get("by", {})

    winner = outcome.get("winner")
    result_type = "normal"
    if not winner:
        if outcome.get("eliminator"):
            winner = outcome.get("eliminator")
            result_type = "tie_super_over"
        else:
            result_type = outcome.get("result", "no_result")

    win_method = outcome.get("method", "standard")

    # Parse squad/playing XI per team
    player_rows: list[dict] = []
    for team_name, player_list in info.get("players", {}).items():
        for idx, player_name in enumerate(player_list, start=1):
            player_rows.append(
                {
                    "match_id": match_id,
                    "competition": competition,
                    "match_date": match_date,
                    "team": team_name,
                    "player_name": player_name,
                    "player_id": registry_people.get(player_name),
                    "roster_order": idx,
                }
            )

    # Parse ball-by-ball deliveries and track main innings totals
    delivery_rows: list[dict] = []
    innings_totals: dict[int, dict[str, int | str | bool]] = {}

    for innings_idx, innings_obj in enumerate(data.get("innings", []), start=1):
        batting_team = innings_obj.get("team")
        bowling_team = team_2 if batting_team == team_1 else team_1
        is_super_over = bool(innings_obj.get("super_over", False))

        inn_runs = 0
        inn_wickets = 0
        inn_legal_balls = 0

        for over_obj in innings_obj.get("overs", []):
            over_num = int(over_obj.get("over", 0))
            legal_ball_in_over = 0

            for ball_seq, delivery in enumerate(over_obj.get("deliveries", []), start=1):
                extras = delivery.get("extras", {})
                wides = int(extras.get("wides", 0))
                noballs = int(extras.get("noballs", 0))
                byes = int(extras.get("byes", 0))
                legbyes = int(extras.get("legbyes", 0))
                penalty = int(extras.get("penalty", 0))

                is_wide = wides > 0
                is_noball = noballs > 0
                is_legal_ball = not (is_wide or is_noball)
                if is_legal_ball:
                    legal_ball_in_over += 1

                runs_obj = delivery.get("runs", {})
                runs_batter = int(runs_obj.get("batter", 0))
                runs_extras = int(runs_obj.get("extras", 0))
                runs_total = int(runs_obj.get("total", 0))
                runs_bowler = runs_batter + wides + noballs

                wickets_list = delivery.get("wickets", [])
                is_wicket = len(wickets_list) > 0
                wicket_kind = wickets_list[0].get("kind") if is_wicket else None
                player_out = wickets_list[0].get("player_out") if is_wicket else None
                is_bowler_wicket = is_wicket and (wicket_kind not in NON_BOWLER_WICKETS)

                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                non_striker = delivery.get("non_striker")

                inn_runs += runs_total
                if is_wicket:
                    inn_wickets += len(wickets_list)
                if is_legal_ball:
                    inn_legal_balls += 1

                delivery_rows.append(
                    {
                        "match_id": match_id,
                        "competition": competition,
                        "match_date": match_date,
                        "innings": innings_idx,
                        "is_super_over": is_super_over,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over_num,
                        "ball_in_over_seq": ball_seq,
                        "legal_ball_in_over": legal_ball_in_over if is_legal_ball else 0,
                        "is_legal_ball": is_legal_ball,
                        "is_wide": is_wide,
                        "is_noball": is_noball,
                        "batter": batter,
                        "batter_id": registry_people.get(batter),
                        "bowler": bowler,
                        "bowler_id": registry_people.get(bowler),
                        "non_striker": non_striker,
                        "non_striker_id": registry_people.get(non_striker),
                        "runs_batter": runs_batter,
                        "runs_extras": runs_extras,
                        "runs_total": runs_total,
                        "runs_bowler": runs_bowler,
                        "wides": wides,
                        "noballs": noballs,
                        "byes": byes,
                        "legbyes": legbyes,
                        "penalty": penalty,
                        "is_wicket": is_wicket,
                        "is_bowler_wicket": is_bowler_wicket,
                        "wicket_kind": wicket_kind,
                        "player_out": player_out,
                        "player_out_id": registry_people.get(player_out) if player_out else None,
                    }
                )

        if not is_super_over and innings_idx in (1, 2):
            innings_totals[innings_idx] = {
                "team": batting_team,
                "runs": inn_runs,
                "wickets": inn_wickets,
                "legal_balls": inn_legal_balls,
            }

    inn1 = innings_totals.get(1, {})
    inn2 = innings_totals.get(2, {})

    match_row = {
        "match_id": match_id,
        "competition": competition,
        "event_name": info.get("event", {}).get("name"),
        "match_date": match_date,
        "season": str(info.get("season", "")),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team_1": team_1,
        "team_2": team_2,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result_type": result_type,
        "win_method": win_method,
        "win_by_runs": outcome_by.get("runs"),
        "win_by_wickets": outcome_by.get("wickets"),
        "overs_limit": info.get("overs", 20),
        "innings_1_team": inn1.get("team"),
        "innings_1_runs": inn1.get("runs"),
        "innings_1_wickets": inn1.get("wickets"),
        "innings_1_legal_balls": inn1.get("legal_balls"),
        "innings_2_team": inn2.get("team"),
        "innings_2_runs": inn2.get("runs"),
        "innings_2_wickets": inn2.get("wickets"),
        "innings_2_legal_balls": inn2.get("legal_balls"),
    }

    return match_row, delivery_rows, player_rows


def run_ingestion() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_matches: list[dict] = []
    all_deliveries: list[dict] = []
    all_players: list[dict] = []
    seen_match_ids: set[str] = set()
    duplicate_count = 0
    error_count = 0

    for comp_name, folder in RAW_DATASETS.items():
        files = sorted(folder.glob("*.json"))
        print(f"Parsing {len(files)} {comp_name.upper()} files...")
        for file_path in files:
            match_id = file_path.stem
            if match_id in seen_match_ids:
                duplicate_count += 1
                continue
            seen_match_ids.add(match_id)

            try:
                m_row, d_rows, p_rows = parse_match_file(file_path, comp_name)
                all_matches.append(m_row)
                all_deliveries.extend(d_rows)
                all_players.extend(p_rows)
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
                error_count += 1
                print(f"ERROR parsing {file_path}: {exc}")

    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)
    players_df = pd.DataFrame(all_players)

    # Convert date columns and sort chronologically
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    deliveries_df["match_date"] = pd.to_datetime(deliveries_df["match_date"])
    players_df["match_date"] = pd.to_datetime(players_df["match_date"])

    matches_df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Validation checks
    print("\n--- Data Quality & Validation Report ---")
    print(f"Matches parsed      : {len(matches_df)}")
    print(f"Duplicate files     : {duplicate_count}")
    print(f"Parse errors        : {error_count}")
    print(f"Total deliveries    : {len(deliveries_df):,}")
    print(f"Match-player rows   : {len(players_df):,}")
    print(f"Date range          : {matches_df['match_date'].min().date()} to {matches_df['match_date'].max().date()}")
    print("\nMatches by competition:")
    print(matches_df["competition"].value_counts().to_string())

    # Validate run margin consistency for standard 2-innings matches won by runs
    margin_check = matches_df[
        (matches_df["result_type"] == "normal")
        & (matches_df["win_method"] == "standard")
        & (matches_df["win_by_runs"].notna())
    ].copy()
    margin_check["calc_margin"] = margin_check["innings_1_runs"] - margin_check["innings_2_runs"]
    mismatches = margin_check[margin_check["calc_margin"] != margin_check["win_by_runs"]]
    print(f"\nInnings margin validation mismatches: {len(mismatches)} / {len(margin_check)}")

    missing_player_ids = deliveries_df["batter_id"].isna().sum() + deliveries_df["bowler_id"].isna().sum()
    print(f"Missing batter/bowler registry IDs  : {missing_player_ids}")

    # Save to Parquet
    matches_df.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
    deliveries_df.to_parquet(PROCESSED_DIR / "deliveries.parquet", index=False)
    players_df.to_parquet(PROCESSED_DIR / "match_players.parquet", index=False)

    print("\nSaved Parquet files to data/processed/:")
    for pq in sorted(PROCESSED_DIR.glob("*.parquet")):
        size_mb = pq.stat().st_size / (1024 * 1024)
        print(f"  - {pq.name:<22} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    run_ingestion()