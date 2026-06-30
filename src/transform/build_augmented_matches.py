from pathlib import Path

import pandas as pd


HISTORICAL_MATCHES_PATH = Path("data/processed/matches_clean.csv")
LIVE_RESULTS_PATH = Path("data/processed/worldcup_2026_live_results.csv")
OUTPUT_PATH = Path("data/processed/matches_augmented.csv")

PLACEHOLDER_PATTERNS = [
    "Winner",
    "Loser",
    "Third Place",
    "Group ",
    "Round of",
    "Quarterfinal",
    "Semifinal",
]


def match_result(home_score, away_score):
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def is_placeholder_team(team):
    if pd.isna(team):
        return True
    team = str(team)
    return any(pattern in team for pattern in PLACEHOLDER_PATTERNS)


def finished_live_results(live_results):
    live = live_results.copy()
    live = live[
        (live["is_finished"] == True)  # noqa: E712
        & live["home_score"].notna()
        & live["away_score"].notna()
    ].copy()

    live = live[
        ~live["home_team"].apply(is_placeholder_team)
        & ~live["away_team"].apply(is_placeholder_team)
    ].copy()

    live["date"] = pd.to_datetime(live["date"], utc=True).dt.date
    live["home_score"] = live["home_score"].astype(int)
    live["away_score"] = live["away_score"].astype(int)
    live["match_result"] = live.apply(
        lambda row: match_result(row["home_score"], row["away_score"]),
        axis=1,
    )

    return pd.DataFrame({
        "date": live["date"],
        "home_team": live["home_team"],
        "away_team": live["away_team"],
        "home_score": live["home_score"],
        "away_score": live["away_score"],
        "tournament": "FIFA World Cup",
        "city": live["city"].fillna(""),
        "country": live["country"].fillna(""),
        "neutral": True,
        "match_result": live["match_result"],
        "source": "espn",
        "provider_match_id": live["provider_match_id"],
    })


def remove_overlapping_worldcup_rows(historical, live_matches):
    historical = historical.copy()
    historical["date"] = pd.to_datetime(historical["date"])

    live_keys = set(
        zip(
            live_matches["home_team"],
            live_matches["away_team"],
        )
    )

    is_2026_worldcup = (
        (historical["date"].dt.year == 2026)
        & historical["tournament"].str.contains("FIFA World Cup", na=False)
    )
    same_pair = historical.apply(
        lambda row: (row["home_team"], row["away_team"]) in live_keys,
        axis=1,
    )

    return historical[~(is_2026_worldcup & same_pair)].copy()


def build_augmented_matches():
    historical = pd.read_csv(HISTORICAL_MATCHES_PATH)
    live_results = pd.read_csv(LIVE_RESULTS_PATH)

    live_matches = finished_live_results(live_results)
    historical_without_overlap = remove_overlapping_worldcup_rows(
        historical,
        live_matches,
    )

    historical_without_overlap["source"] = "historical"
    historical_without_overlap["provider_match_id"] = pd.NA

    augmented = pd.concat(
        [historical_without_overlap, live_matches],
        ignore_index=True,
        sort=False,
    )
    augmented["date"] = pd.to_datetime(augmented["date"])
    augmented = augmented.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(OUTPUT_PATH, index=False)

    print(f"Historical rows: {len(historical)}")
    print(f"Finished ESPN rows added: {len(live_matches)}")
    print(f"Augmented rows: {len(augmented)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_augmented_matches()
