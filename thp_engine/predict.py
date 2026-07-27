"""
predict.py
----------
Clean prediction API for THP estimation.
Wraps both the ML surrogate and the physics engine in one interface.

Usage
-----
    from predict import predict_THP

    result = predict_THP(
        QL=330, WOR=1.5, GLR=273, oil_sg=0.85, water_sg=1.05,
        gas_sg=0.75, T=150, API=35, Bw=1.01, d=2.441,
        depth=5000, Pwf=1000, method="ml"
    )
    print(result)
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

_DIR   = os.path.dirname(__file__)
_MODEL = None
_META  = None

FEATURES = ["QL", "WOR", "GLR", "oil_sg", "water_sg", "gas_sg",
            "T", "API", "Bw", "d", "depth", "Pwf"]


def _load():
    global _MODEL, _META
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(_DIR, "thp_model.joblib"))
    if _META is None:
        with open(os.path.join(_DIR, "model_meta.json")) as f:
            _META = json.load(f)


def predict_THP(
    QL, WOR, GLR, oil_sg, water_sg, gas_sg,
    T, API, Bw, d, depth, Pwf,
    method="ml"
):
    """
    Predict Tubing Head Pressure (THP) for a vertical oil well.

    Parameters
    ----------
    QL        : Total liquid rate, STB/day
    WOR       : Water-oil ratio, STB/STB
    GLR       : Gas-liquid ratio, scf/STB
    oil_sg    : Oil specific gravity (60/60°F)
    water_sg  : Water specific gravity
    gas_sg    : Gas specific gravity (air = 1)
    T         : Average wellbore temperature, °F
    API       : Oil API gravity
    Bw        : Water formation volume factor, bbl/STB
    d         : Tubing inside diameter, inches
    depth     : Well depth, ft
    Pwf       : Bottom-hole flowing pressure, psia
    method    : "ml" for surrogate model, "physics" for gradient traverse

    Returns
    -------
    dict with keys:
        THP         : predicted tubing head pressure, psia
        method      : which method was used
        confidence  : "high" / "medium" / "low" based on input range check
    """
    _load()

    inputs = dict(
        QL=QL, WOR=WOR, GLR=GLR, oil_sg=oil_sg, water_sg=water_sg,
        gas_sg=gas_sg, T=T, API=API, Bw=Bw, d=d, depth=depth, Pwf=Pwf
    )

    # Range check
    ranges = _META["param_ranges"]
    out_of_range = []
    for feat, val in inputs.items():
        lo, hi = ranges[feat]
        if not (lo <= val <= hi):
            out_of_range.append(feat)

    confidence = "high" if not out_of_range else (
        "medium" if len(out_of_range) <= 2 else "low"
    )

    if method == "ml":
        X = np.array([[inputs[f] for f in FEATURES]])
        THP = float(_MODEL.predict(X)[0])
    else:
        from traverse import WellInput, GradientTraverse
        well = WellInput(**inputs)
        _, THP = GradientTraverse(well).run()

    return {
        "THP"          : round(THP, 1),
        "method"       : method,
        "confidence"   : confidence,
        "out_of_range" : out_of_range,
    }


def compare_methods(QL, WOR, GLR, oil_sg, water_sg, gas_sg,
                    T, API, Bw, d, depth, Pwf):
    """Run both methods and return side-by-side comparison."""
    ml  = predict_THP(QL,WOR,GLR,oil_sg,water_sg,gas_sg,T,API,Bw,d,depth,Pwf, method="ml")
    phy = predict_THP(QL,WOR,GLR,oil_sg,water_sg,gas_sg,T,API,Bw,d,depth,Pwf, method="physics")
    diff = abs(ml["THP"] - phy["THP"])
    return {
        "ML surrogate (psia)" : ml["THP"],
        "Physics engine (psia)": phy["THP"],
        "Difference (psia)"   : round(diff, 1),
        "Agreement"           : "good" if diff < 20 else "moderate" if diff < 50 else "check inputs",
        "confidence"          : ml["confidence"],
    }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("── Single prediction (ML surrogate) ──────────────────────")
    r = predict_THP(
        QL=330, WOR=1.5, GLR=273, oil_sg=0.85, water_sg=1.05,
        gas_sg=0.75, T=150, API=35, Bw=1.01, d=2.441,
        depth=5000, Pwf=1000, method="ml"
    )
    print(f"  THP         : {r['THP']} psia")
    print(f"  Confidence  : {r['confidence']}")
    print(f"  Out-of-range: {r['out_of_range'] or 'none'}")

    print("\n── Comparison (ML vs Physics) ────────────────────────────")
    c = compare_methods(
        QL=330, WOR=1.5, GLR=273, oil_sg=0.85, water_sg=1.05,
        gas_sg=0.75, T=150, API=35, Bw=1.01, d=2.441,
        depth=5000, Pwf=1000
    )
    for k, v in c.items():
        print(f"  {k:<28}: {v}")

    print("\n── Speed test (1000 ML predictions) ─────────────────────")
    t0 = time.perf_counter()
    for _ in range(1000):
        predict_THP(500, 1.0, 350, 0.85, 1.05, 0.72, 150, 35, 1.01, 2.441, 5000, 800)
    ms = (time.perf_counter() - t0)
    print(f"  1000 predictions in {ms*1000:.1f} ms ({ms:.3f} ms each)")
