import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/results.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "matches_clean.csv"

def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return "H"   # Home win
    elif row["home_score"] < row["away_score"]:
        return "A"   # Away win
    else:
        return "D"   # Draw
    
def clean_results():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Remove rows where score is missing
    df = df.dropna(subset=["home_score", "away_score"])

    # Convert scores from float to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Remove duplicate matches
    df = df.drop_duplicates()

    # Create result label
    df["match_result"] = df.apply(get_match_result, axis=1)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Save cleaned file
    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned shape:", df.shape)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_results()