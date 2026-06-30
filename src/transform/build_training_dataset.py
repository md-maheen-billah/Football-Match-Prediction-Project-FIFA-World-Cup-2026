import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/matches_clean.csv")
OUTPUT_PATH = Path("data/processed/training_dataset.csv")


def summarize_last_matches(last_matches):
    if len(last_matches) == 0:
        return {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
        }

    wins = sum(1 for match in last_matches if match["result"] == "W")
    draws = sum(1 for match in last_matches if match["result"] == "D")
    losses = sum(1 for match in last_matches if match["result"] == "L")

    goals_for = sum(match["goals_for"] for match in last_matches)
    goals_against = sum(match["goals_against"] for match in last_matches)

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
    }


def build_training_dataset(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    matches = pd.read_csv(input_path)

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    team_history = {}
    rows = []

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        if home not in team_history:
            team_history[home] = []

        if away not in team_history:
            team_history[away] = []

        home_last5 = team_history[home][-5:]
        away_last5 = team_history[away][-5:]

        home_stats = summarize_last_matches(home_last5)
        away_stats = summarize_last_matches(away_last5)

        rows.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,

            "home_recent_wins": home_stats["wins"],
            "home_recent_draws": home_stats["draws"],
            "home_recent_losses": home_stats["losses"],
            "home_recent_goals_for": home_stats["goals_for"],
            "home_recent_goals_against": home_stats["goals_against"],
            "home_recent_goal_diff": home_stats["goal_diff"],

            "away_recent_wins": away_stats["wins"],
            "away_recent_draws": away_stats["draws"],
            "away_recent_losses": away_stats["losses"],
            "away_recent_goals_for": away_stats["goals_for"],
            "away_recent_goals_against": away_stats["goals_against"],
            "away_recent_goal_diff": away_stats["goal_diff"],

            "recent_win_diff": home_stats["wins"] - away_stats["wins"],
            "recent_goal_diff_diff": home_stats["goal_diff"] - away_stats["goal_diff"],

            "neutral": match["neutral"],
            "tournament": match["tournament"],
            "target": match["match_result"],
        })

        # Update history AFTER creating features to avoid data leakage
        if match["home_score"] > match["away_score"]:
            home_result = "W"
            away_result = "L"
        elif match["home_score"] < match["away_score"]:
            home_result = "L"
            away_result = "W"
        else:
            home_result = "D"
            away_result = "D"

        team_history[home].append({
            "result": home_result,
            "goals_for": match["home_score"],
            "goals_against": match["away_score"],
        })

        team_history[away].append({
            "result": away_result,
            "goals_for": match["away_score"],
            "goals_against": match["home_score"],
        })

    training = pd.DataFrame(rows)

    training.to_csv(output_path, index=False)

    print("Training dataset shape:", training.shape)
    print(f"Saved to: {output_path}")

    print("\nColumns:")
    print(training.columns.tolist())

    print("\nMissing values:")
    print(training.isnull().sum())


if __name__ == "__main__":
    build_training_dataset()
