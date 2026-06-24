"""
preprocess.py — Load, clean, and split the landmark CSV dataset.

Outputs
-------
    dataset/X_train.npy
    dataset/X_test.npy
    dataset/y_train.npy
    dataset/y_test.npy
    models/label_encoder.pkl

Usage
-----
    python preprocess.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import FEATURE_DIM, GESTURE_LABELS, NUM_LANDMARKS, get_logger

logger = get_logger("preprocess")

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
CSV_PATH    = DATASET_DIR / "landmarks.csv"
MODELS_DIR  = Path("models")

# ── main ─────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """
    Load the landmark CSV and apply basic cleaning:
    • Drop rows with any NaN feature values.
    • Drop rows whose label is not in GESTURE_LABELS.
    • Report class distribution.
    """
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw shape: %s", df.shape)

    # Keep only known gesture labels
    original_len = len(df)
    df = df[df["label"].isin(GESTURE_LABELS)].copy()
    removed = original_len - len(df)
    if removed:
        logger.warning("Dropped %d rows with unknown labels.", removed)

    # Drop rows that have NaN in any feature column
    feature_cols = [c for c in df.columns if c != "label"]
    df.dropna(subset=feature_cols, inplace=True)
    logger.info("Shape after cleaning: %s", df.shape)

    # Class distribution
    dist = df["label"].value_counts()
    logger.info("Class distribution:\n%s", dist.to_string())

    return df


def split_and_save(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Encode labels, split into train/test, and persist as .npy + pickle.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != "label"]

    # Validate feature count
    if len(feature_cols) != FEATURE_DIM:
        logger.warning(
            "Expected %d feature columns, found %d.", FEATURE_DIM, len(feature_cols)
        )

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    # Encode string labels → integers
    le = LabelEncoder()
    le.fit(GESTURE_LABELS)        # deterministic ordering aligned with utils.py
    y = le.transform(y_raw)

    logger.info("Classes: %s", list(le.classes_))

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Train: %d  |  Test: %d  |  Features: %d  |  Classes: %d",
        len(X_train), len(X_test), X.shape[1], len(le.classes_),
    )

    # Save arrays
    np.save(DATASET_DIR / "X_train.npy", X_train)
    np.save(DATASET_DIR / "X_test.npy",  X_test)
    np.save(DATASET_DIR / "y_train.npy", y_train)
    np.save(DATASET_DIR / "y_test.npy",  y_test)
    logger.info("Saved X/y train-test arrays to %s/", DATASET_DIR)

    # Save label encoder
    le_path = MODELS_DIR / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    logger.info("Saved LabelEncoder to %s", le_path)


# ── entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not CSV_PATH.exists():
        logger.error(
            "Dataset CSV not found at '%s'. "
            "Run collect_data.py first to gather samples.",
            CSV_PATH,
        )
        sys.exit(1)

    df = load_and_clean(CSV_PATH)

    if len(df) < 50:
        logger.error(
            "Too few samples (%d). Collect at least 50 samples before preprocessing.",
            len(df),
        )
        sys.exit(1)

    split_and_save(df)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
