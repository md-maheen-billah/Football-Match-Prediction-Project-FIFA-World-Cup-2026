"""
Predict Home/Draw/Away probabilities for every World Cup 2026 match
(played and upcoming) using the trained XGBoost model.

Matches whose participants aren't determined yet (e.g. "Round of 32 8
Winner") are skipped since there's no team to look up Elo/form for.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from world_cup_montecarlo_simulation import (
    batch_probs,
    build_elo_lookup,
    build_form_lookup,
    build_ko_feature_vec,
    normalize_live_team_name,
)

MODEL_DIR = Path(__file__).parent
BASE_DIR = MODEL_DIR.parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

LIVE_RESULTS_PATH = DATA_PROCESSED / "worldcup_2026_live_results.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
OUTPUT_PATH = DATA_PROCESSED / "match_predictions_2026.csv"


def load_matches(path, elo_lookup):
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].apply(normalize_live_team_name)
    df["away_team"] = df["away_team"].apply(normalize_live_team_name)
    df["teams_known"] = df["home_team"].isin(elo_lookup) & df["away_team"].isin(elo_lookup)
    return df.sort_values("date").reset_index(drop=True)


def actual_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict H/D/A probabilities for every World Cup 2026 match."
    )
    parser.add_argument("--live-results-path", type=Path, default=LIVE_RESULTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--encoder-path", type=Path, default=ENCODER_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    with open(args.encoder_path, "rb") as f:
        le = pickle.load(f)

    elo_df = pd.read_csv(DATA_PROCESSED / "elo_latest.csv", skipinitialspace=True)
    elo_df.columns = [c.strip() for c in elo_df.columns]
    elo_df["country"] = elo_df["country"].str.strip()
    form_df = pd.read_csv(DATA_PROCESSED / "team_recent_form.csv")

    elo_lookup = build_elo_lookup(elo_df)
    form_lookup = build_form_lookup(form_df)

    matches = load_matches(args.live_results_path, elo_lookup)
    resolved = matches[matches["teams_known"]].copy()
    unresolved_count = len(matches) - len(resolved)

    feature_rows = [
        build_ko_feature_vec(row.home_team, row.away_team, elo_lookup, form_lookup)
        for row in resolved.itertuples()
    ]
    proba = batch_probs(model, le, np.stack(feature_rows))

    resolved["prob_home_win"] = proba[:, 0]
    resolved["prob_draw"] = proba[:, 1]
    resolved["prob_away_win"] = proba[:, 2]
    resolved["predicted_outcome"] = [["H", "D", "A"][i] for i in proba.argmax(axis=1)]
    resolved["actual_result"] = [
        actual_result(hs, as_) for hs, as_ in zip(resolved["home_score"], resolved["away_score"])
    ]

    cols = [
        "date", "stage", "group", "home_team", "away_team",
        "prob_home_win", "prob_draw", "prob_away_win", "predicted_outcome",
        "is_finished", "home_score", "away_score", "actual_result",
    ]
    out = resolved[cols]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print(f"Predicted {len(out)} matches ({unresolved_count} skipped: teams not yet determined).")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
