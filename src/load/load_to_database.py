import sqlite3
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
DATABASE_DIR = Path("data/database")
DATABASE_PATH = DATABASE_DIR / "football_prediction.db"


REQUIRED_TABLES = {
    "matches": "matches_clean.csv",
    "fixtures": "fixtures_clean.csv",
    "teams": "teams_clean.csv",
    "elo_latest": "elo_latest.csv",
    "team_recent_form": "team_recent_form.csv",
    "upcoming_match_features": "upcoming_match_features.csv",
    "training_dataset": "training_dataset.csv",
}

OPTIONAL_TABLES = {
    "worldcup_live_results": "worldcup_2026_live_results.csv",
    "matches_augmented": "matches_augmented.csv",
    "training_dataset_augmented": "training_dataset_augmented.csv",
}


def load_to_database():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)

    for table_name, file_name in REQUIRED_TABLES.items():
        file_path = PROCESSED_DIR / file_name

        df = pd.read_csv(file_path)

        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        print(f"Loaded {table_name}: {df.shape}")

    for table_name, file_name in OPTIONAL_TABLES.items():
        file_path = PROCESSED_DIR / file_name
        if not file_path.exists():
            print(f"Skipped optional {table_name}: {file_path} not found")
            continue

        df = pd.read_csv(file_path)

        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        print(f"Loaded optional {table_name}: {df.shape}")

    conn.close()

    print(f"\nDatabase saved to: {DATABASE_PATH}")


if __name__ == "__main__":
    load_to_database()
