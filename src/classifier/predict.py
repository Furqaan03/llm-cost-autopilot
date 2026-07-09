"""Runtime complexity prediction + tier-to-model routing lookup."""
from __future__ import annotations

from pathlib import Path

import joblib
import yaml

from src.classifier.features import extract_features
from src.classifier.train import MODEL_PATH, train

ROUTING_CONFIG_PATH = Path(__file__).resolve().parent / "routing_config.yaml"

_model_cache: dict | None = None


def _load_model() -> dict:
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            train()
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict_tier(prompt: str) -> int:
    bundle = _load_model()
    features = extract_features(prompt)
    scaled = bundle["scaler"].transform(features)
    return int(bundle["clf"].predict(scaled)[0])


def route_to_model_name(prompt: str) -> tuple[int, str]:
    tier = predict_tier(prompt)
    routing = yaml.safe_load(ROUTING_CONFIG_PATH.read_text(encoding="utf-8"))
    return tier, routing[tier]
