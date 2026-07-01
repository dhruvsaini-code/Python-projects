import os

# Root directories
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

# File paths
MATCHES_PATH: str = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_PATH: str = os.path.join(DATA_DIR, "deliveries.csv")
MODEL_PATH: str = os.path.join(MODEL_DIR, "win_prediction_model.pkl")
CSS_PATH: str = os.path.join(ASSETS_DIR, "custom.css")
