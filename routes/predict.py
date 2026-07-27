"""
routes/predict.py
-----------------
/predict/ml    — XGBoost surrogate prediction (1 credit)
/predict/physics — Full gradient-traverse calculation (4 credits)
"""

import time, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..', 'thp_engine'))

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.auth import require_credits
from core.config import settings
from models.db import User, UsageLog
from models.schemas import (
    WellInput, MLPredictionResponse, PhysicsPredictionResponse, TraverseStep
)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Load ML model at startup (not per request)
import joblib
import numpy as np

_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'thp_engine', 'thp_model.joblib')
_model      = joblib.load(_model_path)

FEATURES = ["QL", "WOR", "GLR", "oil_sg", "water_sg", "gas_sg",
            "T", "API", "Bw", "d", "depth", "Pwf"]

PARAM_RANGES = {
    "QL": (100, 1500), "WOR": (0.0, 5.0), "GLR": (100, 800),
    "oil_sg": (0.80, 0.90), "water_sg": (1.03, 1.07), "gas_sg": (0.65, 0.80),
    "T": (100, 200), "API": (25, 45), "Bw": (1.00, 1.02),
    "d": (1.995, 3.476), "depth": (2000, 8000), "Pwf": (200, 2000)
}


def _check_ranges(inputs: dict) -> list:
    out = []
    for k, (lo, hi) in PARAM_RANGES.items():
        if not (lo <= inputs[k] <= hi):
            out.append(k)
    return out


def _confidence(out_of_range: list) -> str:
    if not out_of_range:      return "high"
    if len(out_of_range) <= 2: return "medium"
    return "low"


def _log_usage(db, user, endpoint, credits_used, credits_before,
               inputs, thp, confidence, response_ms):
    log = UsageLog(
        user_id       = user.id,
        endpoint      = endpoint,
        credits_used  = credits_used,
        credits_before= credits_before,
        credits_after = user.credits,
        response_ms   = response_ms,
        input_params  = json.dumps(inputs),
        thp_result    = thp,
        confidence    = confidence,
    )
    db.add(log)
    user.total_calls += 1
    db.commit()


# ── ML Surrogate ─────────────────────────────────────────────────────────────

@router.post("/ml", response_model=MLPredictionResponse,
             summary="Fast THP prediction via XGBoost surrogate",
             description="Returns THP in < 10ms. Costs 1 credit.")
def predict_ml(
    well: WellInput,
    db  : Session = Depends(get_db),
    user: User    = Depends(require_credits(settings.ML_CALL_COST))
):
    t0     = time.perf_counter()
    inputs = well.model_dump()

    X      = np.array([[inputs[f] for f in FEATURES]])
    THP    = float(_model.predict(X)[0])

    out_of_range = _check_ranges(inputs)
    confidence   = _confidence(out_of_range)
    response_ms  = round((time.perf_counter() - t0) * 1000, 3)

    # Deduct credits
    credits_before   = user.credits
    user.credits    -= settings.ML_CALL_COST

    _log_usage(db, user, "ml", settings.ML_CALL_COST,
               credits_before, inputs, THP, confidence, response_ms)

    return MLPredictionResponse(
        THP               = round(THP, 1),
        confidence        = confidence,
        out_of_range      = out_of_range,
        credits_used      = settings.ML_CALL_COST,
        credits_remaining = user.credits,
        response_ms       = response_ms,
    )


# ── Physics Engine ────────────────────────────────────────────────────────────

@router.post("/physics", response_model=PhysicsPredictionResponse,
             summary="Full gradient-traverse THP calculation",
             description="Returns THP + full pressure-depth profile. Costs 4 credits.")
def predict_physics(
    well            : WellInput,
    include_traverse: bool = True,
    db              : Session = Depends(get_db),
    user            : User    = Depends(require_credits(settings.PHYSICS_CALL_COST))
):
    t0     = time.perf_counter()
    inputs = well.model_dump()

    try:
        from traverse import WellInput as WI, GradientTraverse
        wi     = WI(**inputs)
        engine = GradientTraverse(wi)
        table, THP = engine.run()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Physics engine error: {str(e)}")

    out_of_range = _check_ranges(inputs)
    confidence   = _confidence(out_of_range)
    response_ms  = round((time.perf_counter() - t0) * 1000, 3)

    credits_before   = user.credits
    user.credits    -= settings.PHYSICS_CALL_COST

    _log_usage(db, user, "physics", settings.PHYSICS_CALL_COST,
               credits_before, inputs, THP, confidence, response_ms)

    traverse = None
    if include_traverse:
        traverse = [TraverseStep(**row) for _, row in table.iterrows()]

    return PhysicsPredictionResponse(
        THP               = round(THP, 1),
        confidence        = confidence,
        out_of_range      = out_of_range,
        traverse_steps    = len(table),
        traverse_table    = traverse,
        credits_used      = settings.PHYSICS_CALL_COST,
        credits_remaining = user.credits,
        response_ms       = response_ms,
    )
