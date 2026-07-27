"""
save_model.py
-------------
Retrains the final XGBoost model on the full dataset and saves it.
Also saves the feature metadata for the prediction API.
"""

import sys, os, joblib, json
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv("synthetic_dataset.csv")

FEATURES = ["QL", "WOR", "GLR", "oil_sg", "water_sg", "gas_sg",
            "T", "API", "Bw", "d", "depth", "Pwf"]
TARGET = "THP"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Best params from tuning ───────────────────────────────────────────────────

best_params = {
    "n_estimators"    : 700,
    "max_depth"       : 5,
    "learning_rate"   : 0.08,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state"    : 42,
    "verbosity"       : 0,
}

# Train on full training set
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Final model  MAE : {mean_absolute_error(y_test, preds):.2f} psia")
print(f"Final model  R²  : {r2_score(y_test, preds):.4f}")

# ── Save model and metadata ───────────────────────────────────────────────────

joblib.dump(model, "thp_model.joblib")

meta = {
    "features"   : FEATURES,
    "target"     : TARGET,
    "n_train"    : int(len(X_train)),
    "n_test"     : int(len(X_test)),
    "mae_psia"   : round(float(mean_absolute_error(y_test, preds)), 2),
    "r2"         : round(float(r2_score(y_test, preds)), 4),
    "THP_min"    : float(y.min()),
    "THP_max"    : float(y.max()),
    "param_ranges": {
        "QL"      : [100, 1500],
        "WOR"     : [0.0, 5.0],
        "GLR"     : [100, 800],
        "oil_sg"  : [0.80, 0.90],
        "water_sg": [1.03, 1.07],
        "gas_sg"  : [0.65, 0.80],
        "T"       : [100, 200],
        "API"     : [25, 45],
        "Bw"      : [1.00, 1.02],
        "d"       : [1.995, 3.476],
        "depth"   : [2000, 8000],
        "Pwf"     : [200, 2000],
    }
}

with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved → thp_model.joblib")
print("Saved → model_meta.json")
