import pandas as pd
from pathlib import Path

FIXTURES_PATH = Path("data/processed/fixtures_clean.csv")
ELO_PATH = Path("data/processed/elo_latest.csv")
FORM_PATH = Path("data/processed/team_recent_form.csv")

fixtures = pd.read_csv(FIXTURES_PATH)
elo = pd.read_csv(ELO_PATH)
form = pd.read_csv(FORM_PATH)

fixture_teams = set(fixtures["team1"]).union(set(fixtures["team2"]))
elo_teams = set(elo["country"])
form_teams = set(form["team"])

print("\nTeams in fixtures but missing from Elo:")
print(sorted(fixture_teams - elo_teams))

print("\nTeams in fixtures but missing from recent form:")
print(sorted(fixture_teams - form_teams))