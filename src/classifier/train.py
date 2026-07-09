"""Trains the complexity-tier classifier (logistic regression) and reports held-out accuracy."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classifier.build_dataset import DATA_PATH, build
from src.classifier.features import extract_features

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "complexity_classifier.joblib"


def _load_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        build()
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines()]
    X = np.vstack([extract_features(r["prompt"]) for r in rows])
    y = np.array([r["tier"] for r in rows])
    return X, y


def train() -> float:
    X, y = _load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, preds)
    print(f"Held-out accuracy: {accuracy:.1%}")
    print("Confusion matrix (rows=true tier 1/2/3, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return accuracy


if __name__ == "__main__":
    train()
