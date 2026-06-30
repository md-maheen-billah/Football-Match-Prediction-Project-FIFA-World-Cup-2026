from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.scheduler.airflow_augmented_worldcup_dag import dag  # noqa: E402,F401
