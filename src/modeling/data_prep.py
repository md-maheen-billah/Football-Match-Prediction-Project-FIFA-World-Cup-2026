import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = Path(__file__).parent

# Name mismatches between results dataset and ELO dataset
NAME_MAP = {
    "Czech Republic": "Czechia",
}

CUTOFF_YEAR = 1970


def load_data():
    matches = pd.read_csv(PROCESSED_DIR / "training_dataset.csv", parse_dates=["date"])
    elo = pd.read_csv(RAW_DIR / "elo_ratings_wc2026.csv")
    return matches, elo


def normalize_team_names(df):
    df = df.copy()
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)
    return df


def filter_to_wc_teams(df, elo_teams):
    mask = df["home_team"].isin(elo_teams) & df["away_team"].isin(elo_teams)
    return df[mask].copy()


def join_elo(df, elo):
    elo_lookup = elo[["year", "country", "rating"]].rename(columns={"rating": "elo"})

    df = df.copy()
    df["year"] = df["date"].dt.year

    df = df.merge(
        elo_lookup.rename(columns={"country": "home_team", "elo": "home_elo"}),
        on=["year", "home_team"],
        how="left",
    )
    df = df.merge(
        elo_lookup.rename(columns={"country": "away_team", "elo": "away_elo"}),
        on=["year", "away_team"],
        how="left",
    )
    return df


def build_features(df):
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["neutral"] = df["neutral"].astype(int)
    return df


def main():
    matches, elo = load_data()

    elo_teams = set(elo["country"].unique())

    matches = normalize_team_names(matches)
    matches = matches[matches["date"].dt.year >= CUTOFF_YEAR]
    matches = filter_to_wc_teams(matches, elo_teams)
    matches = join_elo(matches, elo)

    before = len(matches)
    matches = matches.dropna(subset=["home_elo", "away_elo"])
    dropped = before - len(matches)
    if dropped:
        print(f"Dropped {dropped} rows with missing ELO coverage")

    matches = build_features(matches)

    feature_cols = [
        "date", "home_team", "away_team",
        "home_elo", "away_elo", "elo_diff",
        "home_recent_wins", "home_recent_draws", "home_recent_losses",
        "home_recent_goals_for", "home_recent_goals_against", "home_recent_goal_diff",
        "away_recent_wins", "away_recent_draws", "away_recent_losses",
        "away_recent_goals_for", "away_recent_goals_against", "away_recent_goal_diff",
        "recent_win_diff", "recent_goal_diff_diff",
        "neutral",
        "target",
    ]

    out = matches[feature_cols].reset_index(drop=True)

    out_path = OUT_DIR / "training_features.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved {len(out)} rows to {out_path}")
    print(f"Target distribution:\n{out['target'].value_counts()}")
    print(f"\nFeature sample:\n{out.head(3).to_string()}")


if __name__ == "__main__":
    main()
