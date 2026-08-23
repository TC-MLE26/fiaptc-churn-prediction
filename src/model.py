# src/model.py
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

class ChurnPredictor:
    def __init__(self):
        self.model_path = BASE_DIR / "models" / "champion_model.joblib"
        self.preprocessor_path = BASE_DIR / "models" / "preprocessor.joblib"
        self.colunas_path = BASE_DIR / "models" / "colunas.joblib"
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.colunas = joblib.load(self.colunas_path)