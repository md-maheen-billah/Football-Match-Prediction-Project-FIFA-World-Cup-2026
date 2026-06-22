import pandas as pd
from pathlib import Path

FIXTURES_PATH = Path("data/processed/fixtures_clean.csv")
ELO_PATH = Path("data/processed/elo_latest.csv")
FORM_PATH = Path("data/processed/team_recent_form.csv")

OUTPUT_PATH = Path("data/processed/upcoming_match_features.csv")


TEAM_NAME_MAP = {
    "USA": "United States",
    "Türkiye": "Turkey",
}


def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)


def build_upcoming_match_features():
    fixtures = pd.read_csv(FIXTURES_PATH)
    elo = pd.read_csv(ELO_PATH)
    form = pd.read_csv(FORM_PATH)

    form["team"] = form["team"].replace({
    "Czech Republic": "Czechia"
    })

    # Keep only group-stage matches with real teams
    fixtures = fixtures[fixtures["stage"] == "Group Stage"].copy()

    fixtures["team1_normalized"] = fixtures["team1"].apply(normalize_team_name)
    fixtures["team2_normalized"] = fixtures["team2"].apply(normalize_team_name)

    elo_rating_map = dict(zip(elo["country"], elo["rating"]))
    elo_rank_map = dict(zip(elo["country"], elo["rank"]))

    fixtures["team1_elo"] = fixtures["team1_normalized"].map(elo_rating_map)
    fixtures["team2_elo"] = fixtures["team2_normalized"].map(elo_rating_map)

    fixtures["team1_elo_rank"] = fixtures["team1_normalized"].map(elo_rank_map)
    fixtures["team2_elo_rank"] = fixtures["team2_normalized"].map(elo_rank_map)

    form_maps = {
        "last5_wins": dict(zip(form["team"], form["last5_wins"])),
        "last5_draws": dict(zip(form["team"], form["last5_draws"])),
        "last5_losses": dict(zip(form["team"], form["last5_losses"])),
        "last5_goals_for": dict(zip(form["team"], form["last5_goals_for"])),
        "last5_goals_against": dict(zip(form["team"], form["last5_goals_against"])),
        "last5_goal_difference": dict(zip(form["team"], form["last5_goal_difference"])),
    }

    for feature, mapping in form_maps.items():
        fixtures[f"team1_{feature}"] = fixtures["team1_normalized"].map(mapping)
        fixtures[f"team2_{feature}"] = fixtures["team2_normalized"].map(mapping)

    fixtures["elo_diff"] = fixtures["team1_elo"] - fixtures["team2_elo"]
    fixtures["elo_rank_diff"] = fixtures["team1_elo_rank"] - fixtures["team2_elo_rank"]

    fixtures["last5_goal_diff_difference"] = (
        fixtures["team1_last5_goal_difference"]
        - fixtures["team2_last5_goal_difference"]
    )

    fixtures["last5_win_difference"] = (
        fixtures["team1_last5_wins"]
        - fixtures["team2_last5_wins"]
    )

    fixtures.to_csv(OUTPUT_PATH, index=False)

    print("Final shape:", fixtures.shape)
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nMissing values in key feature columns:")
    print(
        fixtures[
            [
                "team1_elo",
                "team2_elo",
                "team1_last5_wins",
                "team2_last5_wins",
            ]
        ].isnull().sum()
    )


if __name__ == "__main__":
    build_upcoming_match_features()