"""Configuration for the employee-comment NLP project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "comments.csv"
RESULTS = ROOT / "results"

N_TOPICS = 5
TOP_TERMS = 8
RANDOM_STATE = 42
