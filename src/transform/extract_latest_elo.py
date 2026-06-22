import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/elo_ratings_wc2026.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "elo_latest.csv"


def extract_latest_elo():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    # Sort so latest snapshot is last for each country
    df = df.sort_values(["country", "snapshot_date"])

    # Keep latest row per country
    latest = df.groupby("country", as_index=False).tail(1)

    latest = latest.reset_index(drop=True)

    latest.to_csv(OUTPUT_PATH, index=False)

    print("Latest Elo shape:", latest.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_latest_elo()