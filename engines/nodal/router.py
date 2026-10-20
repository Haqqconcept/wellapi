"""POST /nodal/analyze - nodal analysis (IPR x VLP operating point)."""
import json, time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.auth import require_credits
from core.config import settings
from core.privacy import input_log
from models.db import User, UsageLog

from .schema import NodalRequest, NodalResponse
from .ipr import build_ipr
from .matcher import find_operating_point

router = APIRouter(prefix="/nodal", tags=["Nodal Analysis"])

NODAL_COST = getattr(settings, "NODAL_CALL_COST", 8)   # add NODAL_CALL_COST to core/config.py to tune
_charge = require_credits(NODAL_COST)

# The friction correlation is only trusted inside the range the surrogate and
# physics engine were built for. Outside it we still answer, but warn.
VALIDATED = {"d": (1.995, 3.476), "depth": (2000, 8000), "GLR": (100, 800),
             "WOR": (0.0, 5.0), "API": (25, 45), "T": (100, 200)}
Q_VALIDATED_MAX = 1500


def _build_ipr(s):
    try:
        if s.model == "vogel":
            if s.Qmax is None: raise ValueError("Vogel IPR needs Qmax")
            return build_ipr("vogel", Pr=s.Pr, Qmax=s.Qmax)
        if s.model == "darcy":
            if s.PI is None: raise ValueError("Darcy IPR needs PI")
            return build_ipr("darcy", Pr=s.Pr, PI=s.PI)
        if s.C is not None:
            return build_ipr("fetkovich", Pr=s.Pr, C=s.C, n=s.n)
        if s.Pwf_test is not None and s.Q_test is not None:
            from .ipr import FetkovichIPR
            return FetkovichIPR.from_test_point(s.Pr, s.Pwf_test, s.Q_test, s.n)
        raise ValueError("Fetkovich IPR needs either C, or both Pwf_test and Q_test")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/analyze", response_model=NodalResponse,
             summary="Nodal analysis: IPR/VLP operating point",
             description=f"Costs {NODAL_COST} credits.")
def analyze(req: NodalRequest, db: Session = Depends(get_db), user: User = Depends(_charge)):
    t0 = time.perf_counter()
    if req.q_max <= req.q_min:
        raise HTTPException(status_code=422, detail="q_max must be greater than q_min")

    warnings = []
    w = req.well.model_dump()
    for k, (lo, hi) in VALIDATED.items():
        if not (lo <= w[k] <= hi):
            warnings.append(f"{k}={w[k]} is outside the validated range {lo}-{hi}")
    if req.q_max > Q_VALIDATED_MAX:
        warnings.append(f"q_max={req.q_max} exceeds the validated rate range (up to {Q_VALIDATED_MAX} STB/day): "
                        "the friction correlation underestimates losses at higher rates")
    if req.ipr.Pr > 2000 * 2:
        warnings.append("Reservoir pressure is far above the bottomhole range the physics engine was tested on")

    ipr = _build_ipr(req.ipr)
    try:
        result = find_operating_point(ipr, w, req.target_thp, req.q_min, req.q_max, req.n_points)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Nodal solver error: {e}")

    if any(not p["converged"] for p in result.vlp_curve):
        warnings.append("Some VLP points did not converge within the search bracket; treat the curve with caution")

    ms = round((time.perf_counter() - t0) * 1000, 1)
    before = user.credits
    user.credits -= NODAL_COST
    db.add(UsageLog(user_id=user.id, endpoint="nodal", credits_used=NODAL_COST,
                    credits_before=before, credits_after=user.credits, response_ms=ms,
                    input_params=input_log(req.model_dump()), thp_result=req.target_thp,
                    confidence="low" if warnings else "high"))
    user.total_calls += 1
    db.commit()

    op = result.operating_point
    return NodalResponse(
        operating_point={"found": op.found, "Q": op.Q, "Pwf": op.Pwf, "note": op.note},
        ipr_curve=result.ipr_curve, vlp_curve=result.vlp_curve, warnings=warnings,
        credits_used=NODAL_COST, credits_remaining=user.credits, response_ms=ms)
