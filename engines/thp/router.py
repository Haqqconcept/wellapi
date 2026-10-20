"""
engines/thp/router.py
---------------------
POST /predict/ml       surrogate prediction (1 credit)
POST /predict/physics  full gradient traverse (4 credits)

Out-of-range behaviour: tree models cannot extrapolate (they go flat outside
their training data), so when ANY input falls outside the range the surrogate
was trained on, /predict/ml answers with the physics engine instead and says
so (method = "physics_fallback", confidence = "low", plus a note). The caller
is still charged the ML price. Pass  ?fallback=false  to force the raw
surrogate output anyway (flagged with out_of_range, as before).
"""
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.auth import require_credits
from core.config import settings
from core.privacy import input_log
from models.db import User, UsageLog

from .schema import WellInput, MLPredictionResponse, PhysicsPredictionResponse, TraverseStep
from .model import predict_thp, check_ranges, confidence_from_ranges
from .traverse import WellInput as PhysicsWellInput, GradientTraverse

router = APIRouter(prefix="/predict", tags=["Predictions"])

_charge_ml = require_credits(settings.ML_CALL_COST)
_charge_physics = require_credits(settings.PHYSICS_CALL_COST)


def _run_physics(inputs: dict):
    return GradientTraverse(PhysicsWellInput(**inputs)).run()


def _log_usage(db, user, endpoint, credits_used, credits_before, inputs, thp, confidence, response_ms):
    db.add(UsageLog(
        user_id=user.id, endpoint=endpoint, credits_used=credits_used,
        credits_before=credits_before, credits_after=user.credits, response_ms=response_ms,
        input_params=input_log(inputs),          # None unless STORE_REQUEST_INPUTS=true
        thp_result=thp, confidence=confidence))
    user.total_calls += 1
    db.commit()


@router.post("/ml", response_model=MLPredictionResponse,
             summary="Fast THP prediction via XGBoost surrogate",
             description="Costs 1 credit. Inputs outside the trained range are solved with the "
                         "physics engine instead (see method / note in the response).")
def predict_ml(well: WellInput, fallback: bool = True,
               db: Session = Depends(get_db), user: User = Depends(_charge_ml)):
    t0 = time.perf_counter()
    inputs = well.model_dump()
    out_of_range = check_ranges(inputs)
    method, note = "ml_surrogate", None

    if out_of_range and fallback:
        try:
            _, THP = _run_physics(inputs)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Inputs outside the surrogate's trained range ({', '.join(out_of_range)}) "
                       f"and the physics engine could not solve them either: {e}")
        method, confidence = "physics_fallback", "low"
        note = (f"Inputs outside the surrogate's trained range ({', '.join(out_of_range)}). "
                "Solved with the physics engine instead; results beyond the validated range are "
                "extrapolations of the correlations.")
    else:
        THP = predict_thp(inputs)
        confidence = confidence_from_ranges(out_of_range)

    ms = round((time.perf_counter() - t0) * 1000, 3)
    before = user.credits
    user.credits -= settings.ML_CALL_COST
    _log_usage(db, user, "ml", settings.ML_CALL_COST, before, inputs, THP, confidence, ms)

    return MLPredictionResponse(
        THP=round(THP, 1), method=method, confidence=confidence, out_of_range=out_of_range,
        note=note, credits_used=settings.ML_CALL_COST, credits_remaining=user.credits, response_ms=ms)


@router.post("/physics", response_model=PhysicsPredictionResponse,
             summary="Full gradient-traverse THP calculation",
             description="Returns THP plus the full pressure-depth profile. Costs 4 credits.")
def predict_physics(well: WellInput, include_traverse: bool = True,
                    db: Session = Depends(get_db), user: User = Depends(_charge_physics)):
    t0 = time.perf_counter()
    inputs = well.model_dump()
    try:
        table, THP = _run_physics(inputs)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Physics engine error: {e}")

    out_of_range = check_ranges(inputs)
    confidence = confidence_from_ranges(out_of_range)
    ms = round((time.perf_counter() - t0) * 1000, 3)
    before = user.credits
    user.credits -= settings.PHYSICS_CALL_COST
    _log_usage(db, user, "physics", settings.PHYSICS_CALL_COST, before, inputs, THP, confidence, ms)

    traverse = [TraverseStep(**row) for _, row in table.iterrows()] if include_traverse else None
    return PhysicsPredictionResponse(
        THP=round(THP, 1), confidence=confidence, out_of_range=out_of_range,
        traverse_steps=len(table), traverse_table=traverse,
        credits_used=settings.PHYSICS_CALL_COST, credits_remaining=user.credits, response_ms=ms)
