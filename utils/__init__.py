from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent


DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
