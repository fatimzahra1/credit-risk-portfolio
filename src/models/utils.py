import joblib
from pathlib import Path
import os

def save_model(model, name: str):
    models_dir = Path(__file__).parent / 'artifacts'
    models_dir.mkdir(exist_ok=True)
    filepath = models_dir / f'{name}.pkl'
    joblib.dump(model, filepath)
    return filepath


def load_model(name):
    filepath = f'src/models/artifacts/{name}.pkl'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)