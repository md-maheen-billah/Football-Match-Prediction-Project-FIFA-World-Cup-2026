import pandas as pd
from pathlib import Path
RAW_DIR = Path("data/raw")
files = {
    "results": "results.csv",
    "shootouts": "shootouts.csv",
    "goalscorers": "goalscorers.csv",
    "former_names": "former_names.csv",
    "elo": "elo_ratings_wc2026.csv",
    "fixtures_2026": "wc_2026_fixtures.csv",
    "teams_2026": "wc_2026_teams.csv",
    "wc_editions": "wc_all_editions.csv",
    "wc_matches": "wc_all_matches.csv",
    "wc_top_scorers": "wc_top_scorers.csv",
    "train": "train.csv",
    "test": "test.csv",
}

def load_raw_data():
    data = {}

    for name, file in files.items():
        path = RAW_DIR / file
        data[name] = pd.read_csv(path)
        print(f"{name}: {data[name].shape}")

    return data

if __name__ == "__main__":
    load_raw_data()