"""Vertical Lift Performance (outflow) curve: Pwf required to deliver a
target THP at each flow rate, by bisection on the validated traverse engine."""
from dataclasses import dataclass
from typing import List
from engines.thp.traverse import WellInput, GradientTraverse


@dataclass
class VLPPoint:
    Q: float
    Pwf_required: float
    converged: bool


def _thp(pwf, q, fixed):
    p = dict(fixed); p["QL"] = q; p["Pwf"] = pwf
    return GradientTraverse(WellInput(**p)).run()[1]


def solve_pwf_for_target_thp(q, target_thp, fixed, lo=50.0, hi=6000.0, tol=1.0, max_iter=40):
    t_lo, t_hi = _thp(lo, q, fixed), _thp(hi, q, fixed)
    if target_thp <= t_lo:
        return VLPPoint(q, lo, False)
    if target_thp >= t_hi:
        return VLPPoint(q, hi, False)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        t = _thp(mid, q, fixed)
        if abs(t - target_thp) <= tol:
            return VLPPoint(q, round(mid, 1), True)
        lo, hi = (mid, hi) if t < target_thp else (lo, mid)
    return VLPPoint(q, round((lo + hi) / 2, 1), False)


def build_vlp_curve(fixed, target_thp, q_min, q_max, n_points=12) -> List[VLPPoint]:
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    step = (q_max - q_min) / (n_points - 1)
    return [solve_pwf_for_target_thp(q_min + i * step, target_thp, fixed) for i in range(n_points)]
