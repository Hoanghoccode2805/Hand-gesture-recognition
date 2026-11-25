import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = BASE_DIR / "data" / "gesture_data.csv"

# 63 features (x0, y0, z0, ..., x20, y20, z20)
FEATURE_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
class Preprocessor:
    def normalize_landmarks(features):
        pts = np.asarray(features, dtype=np.float32).reshape(21, 3)

        # 1. Set the coordinate origin at the wrist (point 0)
        wrist = pts[0].copy()
        pts = pts - wrist

        # 2. Normalize the dimensions to the range [-1, 1]
        max_val = np.max(np.abs(pts))
        if max_val > 0:
            pts = pts / max_val

        return pts.flatten()

    def preprocess_dataFrame(df):
        df_cleaned = df.dropna(subset= FEATURE_COLS + ['label']).copy()

        X_raw = df_cleaned[FEATURE_COLS].values
        y = df_cleaned["label"].astype(int).values

        X_norm = np.array([Preprocessor.normalize_landmarks(row) for row in X_raw], dtype= np.float32)

        return X_norm, y

    def load_and_split_data(csv_path=DEFAULT_CSV_PATH, test_size=0.2, random_state=42):
        df = pd.read_csv(csv_path)
        X, y = Preprocessor.preprocess_dataFrame(df)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_size, random_state = random_state, stratify=y)

        return X_train, X_test, y_train, y_test

    