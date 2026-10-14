"""
engines/thp/model.py
---------------------
XGBoost surrogate model wrapper for THP prediction.
Loads via the central ml.registry instead of an ad-hoc joblib.load() +
relative path, which is what caused prior Railway deployment crashes.
"""

import numpy as np

from ml.registry import load_model

FEATURES = ["QL", "WOR", "GLR", "oil_sg", "water_sg", "gas_sg",
            "T", "API", "Bw", "d", "depth", "Pwf"]

PARAM_RANGES = {
    "QL": (100, 1500), "WOR": (0.0, 5.0), "GLR": (100, 800),
    "oil_sg": (0.80, 0.90), "water_sg": (1.03, 1.07), "gas_sg": (0.65, 0.80),
    "T": (100, 200), "API": (25, 45), "Bw": (1.00, 1.02),
    "d": (1.995, 3.476), "depth": (2000, 8000), "Pwf": (200, 2000)
}


def check_ranges(inputs: dict) -> list:
    """Return list of input keys that fall outside the model's trained range."""
    out = []
    for k, (lo, hi) in PARAM_RANGES.items():
        if not (lo <= inputs[k] <= hi):
            out.append(k)
    return out


def confidence_from_ranges(out_of_range: list) -> str:
    if not out_of_range:
        return "high"
    if len(out_of_range) <= 2:
        return "medium"
    return "low"


def predict_thp(inputs: dict) -> float:
    """
    Run the XGBoost surrogate model on a validated well-input dict.
    Returns THP in psia.
    """
    model = load_model("thp_xgboost")
    X = np.array([[inputs[f] for f in FEATURES]])
    return float(model.predict(X)[0])
