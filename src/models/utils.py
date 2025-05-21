import joblib
from pathlib import Path

def save_model(model, name: str):
    models_dir = Path(__file__).parent / 'artifacts'
    models_dir.mkdir(exist_ok=True)
    filepath = models_dir / f'{name}.pkl'
    joblib.dump(model, filepath)
    return filepath

def load_model(name: str):
    filepath = Path(__file__).parent / 'artifacts' / f'{name}.pkl'
    return joblib.load(filepath)