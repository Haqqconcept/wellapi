"""
ml/registry.py
---------------
Central registry for all trained ML model artifacts used across WellAPI's
engines (THP today; nodal/VLP, gas lift, etc. in later phases).

Why this exists
----------------
Each engine used to hardcode its own relative path to its .joblib/.json
file, which is what caused the Railway deployment crashes documented in
WellAPI's build notes (relative paths break depending on the process's
working directory). This module solves that once, in one place, using
__file__-relative absolute paths — every engine calls `load_model(key)`
instead of managing its own path logic.

Usage
-----
    from ml.registry import load_model

    model = load_model("thp_xgboost")
"""

import os
import threading
import joblib

# Absolute path to the repo root (this file lives in <root>/ml/registry.py)
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model registry ────────────────────────────────────────────────────────────
# Add one entry per trained model artifact. Path is relative to repo root.
MODEL_REGISTRY = {
    "thp_xgboost": {
        "path": os.path.join(_ROOT_DIR, "engines", "thp", "artifacts", "thp_model.joblib"),
        "loader": joblib.load,
    },
    # Future models register here, e.g.:
    # "vlp_curve_xgboost": {
    #     "path": os.path.join(_ROOT_DIR, "engines", "nodal", "artifacts", "vlp_model.joblib"),
    #     "loader": joblib.load,
    # },
}

_cache = {}
_lock = threading.Lock()


def load_model(key: str):
    """
    Load (and cache) a model by its registry key. Thread-safe, loads once.

    Raises
    ------
    KeyError    if the key isn't registered in MODEL_REGISTRY
    FileNotFoundError  if the registered path doesn't exist on disk
    """
    if key in _cache:
        return _cache[key]

    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{key}'. Registered keys: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[key]
    path = entry["path"]

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model artifact for '{key}' not found at {path}. "
            f"Check that the file was committed/deployed correctly."
        )

    with _lock:
        # Re-check in case another thread loaded it while we waited on the lock
        if key not in _cache:
            _cache[key] = entry["loader"](path)

    return _cache[key]


def registered_models() -> list:
    """Utility: list all registered model keys (useful for a /health or /debug endpoint)."""
    return list(MODEL_REGISTRY.keys())
