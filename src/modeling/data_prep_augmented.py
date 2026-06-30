from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.modeling.data_prep import prepare_features  # noqa: E402


TRAINING_DATASET_PATH = BASE_DIR / "data" / "processed" / "training_dataset_augmented.csv"
OUTPUT_PATH = Path(__file__).parent / "training_features_augmented.csv"


if __name__ == "__main__":
    prepare_features(TRAINING_DATASET_PATH, OUTPUT_PATH)
