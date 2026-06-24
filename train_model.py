"""
train_model.py — Train an MLP classifier on hand-landmark features.
Uses scikit-learn MLPClassifier (works on Python 3.13+).

Usage
-----
    python train_model.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

logger = get_logger("train_model")

DATASET_DIR = Path("dataset")
MODELS_DIR  = Path("models")
MODEL_PATH  = MODELS_DIR / "gesture_model.pkl"
LE_PATH     = MODELS_DIR / "label_encoder.pkl"


def train() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    for p in [DATASET_DIR / f for f in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")]:
        if not p.exists():
            logger.error("Missing %s — run preprocess.py first.", p)
            sys.exit(1)

    X_train = np.load(DATASET_DIR / "X_train.npy")
    X_test  = np.load(DATASET_DIR / "X_test.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")
    y_test  = np.load(DATASET_DIR / "y_test.npy")

    with open(LE_PATH, "rb") as f:
        import pickle
        le = pickle.load(f)

    logger.info("Train: %d  Test: %d  Classes: %d", len(X_train), len(X_test), len(le.classes_))

    # ── Build and train model ─────────────────────────────────────────────────
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=True,
    )

    logger.info("Training MLP classifier...")
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    accuracy = model.score(X_test, y_test)
    logger.info("Test accuracy: %.4f", accuracy)

    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    logger.info("Precision: %.4f  Recall: %.4f  F1: %.4f", precision, recall, f1)

    print("\n" + "=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("=" * 60 + "\n")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Gesture Recognition")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix.")

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Training complete.")


if __name__ == "__main__":
    train()