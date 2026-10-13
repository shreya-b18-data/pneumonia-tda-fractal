"""Shared constants. Edit DATA_DIR if your dataset lives elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Kaggle "Chest X-Ray Images (Pneumonia)" layout: chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
DATA_DIR = ROOT / "chest_xray"
FEATURE_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"

CLASSES = ["NORMAL", "PNEUMONIA"]   # label 0 / 1
IMG_SIZE = 128                      # images are resized to IMG_SIZE x IMG_SIZE
SEED = 42
