"""
engines/thp/router.py
-----------------------
/predict/ml       — XGBoost surrogate prediction (1 credit)
/predict/physics  — Full gradient-traverse calculation (4 credits)

This is the THP engine's self-contained router. Mounted onto the main app
in api/main.py via app.include_router(thp_router, prefix="/predict").
"""

import time
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.auth import require_credits
from core.config import settings
from models.db import User, UsageLog

from .schema import (
    WellInput, MLPredictionResponse, PhysicsPredictionResponse, TraverseStep
)
from .model import predict_thp, check_ranges, confidence_from_ranges
from .traverse import WellInput as PhysicsWellInput, GradientTraverse

router = APIRouter(prefix="/predict", tags=["Predictions"])


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

    THP = predict_thp(inputs)

    out_of_range = check_ranges(inputs)
    confidence   = confidence_from_ranges(out_of_range)
    response_ms  = round((time.perf_counter() - t0) * 1000, 3)

    credits_before = user.credits
    user.credits -= settings.ML_CALL_COST

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
        wi     = PhysicsWellInput(**inputs)
        engine = GradientTraverse(wi)
        table, THP = engine.run()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Physics engine error: {str(e)}")

    out_of_range = check_ranges(inputs)
    confidence   = confidence_from_ranges(out_of_range)
    response_ms  = round((time.perf_counter() - t0) * 1000, 3)

    credits_before = user.credits
    user.credits -= settings.PHYSICS_CALL_COST

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
