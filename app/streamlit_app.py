import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "modeling"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_recent_wins", "home_recent_draws", "home_recent_losses",
    "home_recent_goals_for", "home_recent_goals_against", "home_recent_goal_diff",
    "away_recent_wins", "away_recent_draws", "away_recent_losses",
    "away_recent_goals_for", "away_recent_goals_against", "away_recent_goal_diff",
    "recent_win_diff", "recent_goal_diff_diff",
    "neutral",
]


@st.cache_resource
def load_model_and_data():
    model_path = MODEL_DIR / "model.pkl"
    le_path = MODEL_DIR / "label_encoder.pkl"
    upcoming_path = DATA_DIR / "upcoming_match_features.csv"

    model = None
    le = None
    upcoming = None
    model_error = None
    if model_path.exists() and le_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(le_path, "rb") as f:
                le = pickle.load(f)
        except ModuleNotFoundError as e:
            model_error = e
            model = None
            le = None
        except Exception as e:
            model_error = e
            model = None
            le = None

    if upcoming_path.exists():
        upcoming = pd.read_csv(upcoming_path, parse_dates=["date"])    

    return model, le, upcoming, model_error


def build_feature_array_from_row(row):
    # Map upcoming file column names to the model FEATURE_COLS order
    arr = np.array([
        row["team1_elo"], row["team2_elo"], row["elo_diff"],
        row["team1_last5_wins"], row["team1_last5_draws"], row["team1_last5_losses"],
        row["team1_last5_goals_for"], row["team1_last5_goals_against"], row["team1_last5_goal_difference"],
        row["team2_last5_wins"], row["team2_last5_draws"], row["team2_last5_losses"],
        row["team2_last5_goals_for"], row["team2_last5_goals_against"], row["team2_last5_goal_difference"],
        row.get("last5_win_difference", 0), row.get("last5_goal_diff_difference", 0),
        1,
    ], dtype=np.float32)
    return arr


def show_match_prediction(model, le, row):
    X = build_feature_array_from_row(row).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    # Map classes -> probabilities using label encoder ordering
    classes = list(le.classes_)
    mapping = {c: proba[i] for i, c in enumerate(classes)}

    st.subheader(f"Prediction: {row['team1_normalized']} vs {row['team2_normalized']}")
    st.write(f"Date: {row['date'].date()}")

    df = pd.DataFrame([
        {"result": "Home Win (H)", "prob": mapping.get("H", 0)},
        {"result": "Draw (D)", "prob": mapping.get("D", 0)},
        {"result": "Away Win (A)", "prob": mapping.get("A", 0)},
    ])
    df["prob_pct"] = (df["prob"] * 100).round(2).astype(str) + "%"
    st.table(df.set_index("result")["prob_pct"])
    st.bar_chart(df.set_index("result")["prob"])


def import_simulator_module():
    # Dynamically import the simulation module so Streamlit can call run()
    import importlib.util

    sim_path = MODEL_DIR / "world_cup_montecarlo_simulation.py"
    if not sim_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("wcsim", str(sim_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    st.title("World Cup 2026 — Match Predictions")
    st.write("Select an upcoming match to see predicted probabilities from the XGBoost model.")

    model, le, upcoming, model_error = load_model_and_data()
    if model_error is not None:
        error_text = str(model_error).lower()
        if "libomp" in error_text or "library not loaded" in error_text or "xgboost error" in error_text:
            st.error(
                "XGBoost is installed, but the native XGBoost library failed to load. "
                "On macOS, install the OpenMP runtime with `brew install libomp`, then restart Streamlit. "
                "If that does not fix it, reinstall xgboost with `pip install --force-reinstall xgboost`."
            )
        elif "xgboost" in error_text:
            st.error("Missing required module: xgboost. Install it with `pip install xgboost` or `pip install -r requirements.txt`.")
        else:
            st.error(f"Failed to load the model: {model_error}")
        return
    if model is None or le is None:
        st.error("Model or label encoder not found. Train the model first and place `model.pkl` and `label_encoder.pkl` in `src/modeling/`.")
        return
    if upcoming is None or upcoming.empty:
        st.error("Upcoming match features not found in `data/processed/upcoming_match_features.csv`.")
        return

    upcoming["label"] = upcoming["team1_normalized"] + " vs " + upcoming["team2_normalized"] + " — " + upcoming["date"].dt.strftime("%Y-%m-%d")
    sel = st.selectbox("Choose match", upcoming["label"].tolist())
    row = upcoming[upcoming["label"] == sel].iloc[0]

    show_match_prediction(model, le, row)

    st.markdown("---")
    st.subheader("Monte Carlo Tournament Simulation")
    st.write("Optionally run a full tournament Monte Carlo simulation. This may take time.")
    n = st.slider("Number of simulations", min_value=100, max_value=20000, value=1000, step=100)
    run_sim = st.button("Run simulation")
    if run_sim:
        sim_mod = import_simulator_module()
        if sim_mod is None:
            st.error("Simulation module not found at src/modeling/world_cup_montecarlo_simulation.py")
        else:
            with st.spinner(f"Running {n} simulations... this may take a while"):
                try:
                    df = sim_mod.run(n_simulations=n)
                    st.success("Simulation complete")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode()
                    st.download_button("Download results CSV", data=csv, file_name="simulation_results.csv")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
