import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/wc_2026_fixtures.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "fixtures_clean.csv"

def clean_fixtures():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values("date")

    # Reset index
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned shape:", df.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_fixtures()