"""Project paths shared by the notebooks, the Streamlit app and the tests."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
MEMBER1_DIR = OUTPUT_DIR / 'member1'
MEMBER2_DIR = OUTPUT_DIR / 'member2'
MEMBER3_DIR = OUTPUT_DIR / 'member3'
