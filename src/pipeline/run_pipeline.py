import subprocess

scripts = [
    "src/transform/clean_results.py",
    "src/transform/clean_fixtures.py",
    "src/transform/clean_teams.py",
    "src/transform/extract_latest_elo.py",
    "src/transform/build_recent_form.py",
    "src/transform/build_upcoming_match_features.py",
    "src/load/load_to_database.py",
    "src/transform/build_training_dataset.py"
]

for script in scripts:
    print(f"\nRunning {script}")
    subprocess.run(["python", script], check=True)

print("\nPipeline completed successfully!")