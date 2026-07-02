# Football Match Prediction for FIFA World Cup 2026

---

## Slide 1: Title

**Football Match Prediction for FIFA World Cup 2026**

- Machine learning dashboard for match and tournament forecasts
- XGBoost model + Monte Carlo simulation
- Streamlit interface for live predictions

Speaker note: introduce the project and explain that the goal is to predict match outcomes and tournament chances.

---

## Slide 2: Project scope

**What this system does**

- Predicts single match outcomes: home win / draw / away win
- Simulates the rest of the World Cup using those probabilities
- Displays predictions in a Streamlit dashboard

Speaker note: clarify that this is both a prediction model and a simulation framework.

---

## Slide 3: Architecture and stack

**Technical components**

- Python
- XGBoost for match prediction
- Streamlit for dashboard
- CSV / SQLite data storage
- Monte Carlo simulation for tournament probability aggregation

Speaker note: mention key files like `src/modeling/eval_model_training.py`, `src/modeling/world_cup_montecarlo_simulation.py`, and `app/streamlit_app.py`.

---

## Slide 4: Model inputs and features

**Features used by the XGBoost model**

- Home and away Elo ratings
- Elo rating difference
- Recent form for each team (last 5 matches):
  - wins, draws, losses
  - goals for, goals against
  - goal difference
- Derived differences between teams
- Neutral venue flag

Speaker note: emphasize the combination of long-term strength (Elo) and short-term momentum (recent form).

---

## Slide 5: Dashboard usage

**How to use the dashboard**

1. Run `streamlit run app/streamlit_app.py`
2. Choose an upcoming match from the dropdown
3. Read the probability table and bar chart
4. Optionally run the Monte Carlo tournament simulation

Speaker note: explain that the dropdown uses `data/processed/upcoming_match_features.csv`.

---

## Slide 6: Single match prediction

**What the prediction means**

- Home Win (H): probability the first team wins
- Draw (D): probability the match ends tied
- Away Win (A): probability the second team wins

Example:
- Curacao vs Ivory Coast → Home Win 9.42%, Draw 17.09%, Away Win 73.49%

Speaker note: the highest probability is the model’s most likely outcome, but the others show uncertainty.

---

## Slide 7: Monte Carlo Tournament Simulation

**What Monte Carlo does**

- Uses match probabilities from the XGBoost model
- Simulates the remaining tournament many times
- Randomly chooses match outcomes according to predicted odds
- Aggregates team performance across simulations

Speaker note: explain that Monte Carlo is not inside XGBoost — it is an added simulation layer.

---

## Slide 8: Simulation slider meaning

**Number of simulations**

- Controls how many tournament runs are performed
- More simulations = smoother, more stable results
- Fewer simulations = faster, more variable results
- Does not change the model’s match probabilities

Speaker note: recommend a moderate value for live demos, higher values for more accurate analysis.

---

## Slide 9: Output metrics explained

**Simulation result columns**

- `win_pct`: percent of simulations the team won the tournament
- `final_pct`: percent the team reached the final
- `semi_pct`: percent the team reached the semifinals
- `quarter_pct`: percent the team reached the quarterfinals
- `r16_pct`: percent the team reached the Round of 16
- `r32_pct`: percent the team reached the Round of 32
- `wins`: raw count of tournament wins in the simulations

Speaker note: these numbers are probabilities derived from repeated simulated tournaments.

---

## Slide 10: Summary and next steps

**Summary**

- XGBoost predicts individual match outcomes
- Monte Carlo converts match probabilities into tournament forecasts
- The dashboard lets users explore both match and tournament predictions
- This system blends statistical strength and interactive visualization

**Possible improvements**

- add bookmaker odds or injury data
- add head-to-head history
- add live updating and more match context

Speaker note: end with future work and the value of the current system.
