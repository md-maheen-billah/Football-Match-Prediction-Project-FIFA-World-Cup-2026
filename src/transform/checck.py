import pandas as pd

form = pd.read_csv("data/processed/team_recent_form.csv")

print(form[form["team"].str.contains("Czech", case=False, na=False)])