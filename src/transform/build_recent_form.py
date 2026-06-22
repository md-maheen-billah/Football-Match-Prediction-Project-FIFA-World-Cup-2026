import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/matches_clean.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "team_recent_form.csv"


def build_team_match_rows(matches):
    rows = []

    for _, row in matches.iterrows():
        # Home team perspective
        rows.append({
            "date": row["date"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "goals_for": row["home_score"],
            "goals_against": row["away_score"],
            "result": "W" if row["home_score"] > row["away_score"] else "L" if row["home_score"] < row["away_score"] else "D"
        })

        # Away team perspective
        rows.append({
            "date": row["date"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "goals_for": row["away_score"],
            "goals_against": row["home_score"],
            "result": "W" if row["away_score"] > row["home_score"] else "L" if row["away_score"] < row["home_score"] else "D"
        })

    return pd.DataFrame(rows)


def build_recent_form():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(INPUT_PATH)
    matches["date"] = pd.to_datetime(matches["date"])

    print("Matches shape:", matches.shape)

    team_matches = build_team_match_rows(matches)

    recent_rows = []

    for team, group in team_matches.groupby("team"):
        last5 = group.sort_values("date").tail(5)

        recent_rows.append({
            "team": team,
            "last5_matches": len(last5),
            "last5_wins": (last5["result"] == "W").sum(),
            "last5_draws": (last5["result"] == "D").sum(),
            "last5_losses": (last5["result"] == "L").sum(),
            "last5_goals_for": last5["goals_for"].sum(),
            "last5_goals_against": last5["goals_against"].sum(),
            "last5_goal_difference": last5["goals_for"].sum() - last5["goals_against"].sum()
        })

    recent_form = pd.DataFrame(recent_rows)

    recent_form = recent_form.sort_values("team").reset_index(drop=True)

    recent_form.to_csv(OUTPUT_PATH, index=False)

    print("Recent form shape:", recent_form.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_recent_form()