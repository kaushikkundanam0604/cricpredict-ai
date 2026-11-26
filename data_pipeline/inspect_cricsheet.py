import json
from pathlib import Path

RAW_DATASETS = {
    "t20i": Path("data/raw/t20s_male_json"),
    "ilt20": Path("data/raw/ilt20_json"),
    "bbl": Path("data/raw/bbl_json"),
}


def inspect_datasets() -> None:
    total_files = 0
    ilt20_files: list[Path] = []

    print("--- Dataset File Counts ---")
    for comp_name, folder in RAW_DATASETS.items():
        files = sorted(folder.glob("*.json"))
        total_files += len(files)
        print(f"{comp_name.upper():<6}: {len(files)} JSON files ({folder})")
        if comp_name == "ilt20":
            ilt20_files = files

    print(f"TOTAL : {total_files} JSON files\n")

    if not ilt20_files:
        print("No ILT20 JSON files found. Check data/raw/ilt20_json/")
        return

    sample_path = ilt20_files[-1]  # Inspect the most recent ILT20 match
    print(f"Inspecting sample ILT20 match: {sample_path.name}\n")

    with sample_path.open("r", encoding="utf-8") as f:
        match_data = json.load(f)

    info = match_data.get("info", {})
    print("--- ILT20 Match Info ---")
    print("Event:", info.get("event"))
    print("Dates:", info.get("dates"))
    print("Teams:", info.get("teams"))
    print("Venue:", info.get("venue"))
    print("City:", info.get("city"))
    print("Toss:", info.get("toss"))
    print("Outcome:", info.get("outcome"))
    print("Match type:", info.get("match_type"))
    print("Overs:", info.get("overs"))

    players = info.get("players", {})
    print("\nPlayers per team:", {team: len(roster) for team, roster in players.items()})
    registry_people = info.get("registry", {}).get("people", {})
    print(f"Registry player IDs mapped: {len(registry_people)}")

    innings = match_data.get("innings", [])
    print(f"Innings recorded: {len(innings)}")
    if innings and innings[0].get("overs"):
        first_ball = innings[0]["overs"][0]["deliveries"][0]
        print("\nSample delivery (1st innings, Over 0, Ball 1):")
        print(json.dumps(first_ball, indent=2))


if __name__ == "__main__":
    inspect_datasets()