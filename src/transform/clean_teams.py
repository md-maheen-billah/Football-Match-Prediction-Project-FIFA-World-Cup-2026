import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/wc_2026_teams.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "teams_clean.csv"


def clean_teams():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize text columns
    text_columns = [
        "team",
        "group",
        "confederation",
        "coach",
        "best_wc_result"
    ]

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # Reset index
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned shape:", df.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_teams()