"""Operating point: where inflow (IPR) meets required outflow (VLP)."""
from dataclasses import dataclass
from typing import List, Optional
from .vlp import build_vlp_curve


@dataclass
class OperatingPoint:
    found: bool
    Q: float = 0.0
    Pwf: float = 0.0
    note: Optional[str] = None


@dataclass
class NodalResult:
    operating_point: OperatingPoint
    ipr_curve: List[dict]
    vlp_curve: List[dict]


def find_operating_point(ipr, fixed, target_thp, q_min, q_max, n_points=12) -> NodalResult:
    pts = build_vlp_curve(fixed, target_thp, q_min, q_max, n_points)
    ipr_curve, vlp_curve, diffs = [], [], []
    for p in pts:
        ipr_pwf = ipr.pwf_at_q(p.Q)
        ipr_curve.append({"Q": round(p.Q, 1), "Pwf": round(ipr_pwf, 1)})
        vlp_curve.append({"Q": round(p.Q, 1), "Pwf": round(p.Pwf_required, 1), "converged": p.converged})
        diffs.append(ipr_pwf - p.Pwf_required)

    for i in range(len(diffs) - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        if d0 == 0 or (d0 > 0) != (d1 > 0):
            f = 0.0 if d0 == 0 else d0 / (d0 - d1)
            q = pts[i].Q + f * (pts[i + 1].Q - pts[i].Q)
            pwf = ipr_curve[i]["Pwf"] + f * (ipr_curve[i + 1]["Pwf"] - ipr_curve[i]["Pwf"])
            return NodalResult(OperatingPoint(True, round(q, 1), round(pwf, 1)), ipr_curve, vlp_curve)

    if all(d > 0 for d in diffs):
        note = "Reservoir can deliver more than the tubing needs across this whole range. Try a higher q_max."
    elif all(d < 0 for d in diffs):
        note = "Tubing requires more pressure than the reservoir provides across this whole range: the well cannot flow at this target THP."
    else:
        note = "No single crossing found. Check inputs."
    return NodalResult(OperatingPoint(False, note=note), ipr_curve, vlp_curve)
