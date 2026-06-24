import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = Path(__file__).parent / "training_features.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_recent_wins", "home_recent_draws", "home_recent_losses",
    "home_recent_goals_for", "home_recent_goals_against", "home_recent_goal_diff",
    "away_recent_wins", "away_recent_draws", "away_recent_losses",
    "away_recent_goals_for", "away_recent_goals_against", "away_recent_goal_diff",
    "recent_win_diff", "recent_goal_diff_diff",
    "neutral",
]

EVAL_SPLIT_YEAR = 2018

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)


def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["target"])  # A=0, D=1, H=2
    return df, le


def temporal_split(df):
    train = df[df["date"].dt.year < EVAL_SPLIT_YEAR]
    test = df[df["date"].dt.year >= EVAL_SPLIT_YEAR]
    return train, test


def train_model(df, params=None):
    if params is None:
        params = XGB_PARAMS
    X = df[FEATURE_COLS]
    y = df["label"]
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def evaluate(model, df, le):
    X = df[FEATURE_COLS]
    y_true = df["label"]
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}  ({len(df)} samples)")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return acc


def feature_importance(model):
    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importance.to_string())


def main():
    df, le = load_data()

    train_df, test_df = temporal_split(df)
    print(f"Train: {len(train_df)} rows (pre-{EVAL_SPLIT_YEAR})")
    print(f"Test:  {len(test_df)} rows ({EVAL_SPLIT_YEAR}+)\n")

    print("--- Eval model (pre-2018 train / post-2018 test) ---")
    eval_model = train_model(train_df)
    evaluate(eval_model, test_df, le)
    feature_importance(eval_model)

    print(f"\n--- Final model (all {len(df)} rows) ---")
    final_model = train_model(df)
    print("Retrained on full dataset.")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved model → {MODEL_PATH}")
    print(f"Saved encoder → {ENCODER_PATH}")

    # Sanity check: in-sample accuracy on full data
    print("\nIn-sample accuracy (not meaningful for eval, just a sanity check):")
    evaluate(final_model, df, le)


if __name__ == "__main__":
    main()
