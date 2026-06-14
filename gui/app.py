
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from src.classifier import EmailClassifier

# gui/app.py is inside the gui/ folder, parents[1] points back to
# the project root: email_classifier/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "spam.csv"

