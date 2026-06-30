from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.transform.build_training_dataset import build_training_dataset  # noqa: E402


INPUT_PATH = Path("data/processed/matches_augmented.csv")
OUTPUT_PATH = Path("data/processed/training_dataset_augmented.csv")


if __name__ == "__main__":
    build_training_dataset(INPUT_PATH, OUTPUT_PATH)
