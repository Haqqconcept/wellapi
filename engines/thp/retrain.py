"""
retrain.py
----------
Regenerates the synthetic dataset with the CORRECTED physics engine
(traverse.py with the Ef / K-bar fix) and retrains the XGBoost surrogate.

Run from the folder that contains traverse.py and pvt.py:

    python retrain.py                       # defaults: 12,000 cases, continuous sampling
    python retrain.py --n 6000              # quicker test run
    python retrain.py --out-dir artifacts   # refactored layout (engines/thp/artifacts)

Differences from the original generate_dataset.py + save_model.py:
  * Continuous Latin-hypercube sampling for the continuous inputs, instead of
    picking from fixed grid values. The old grid sampling gave good scores on
    grid inputs but ~6x worse error on arbitrary inputs.
  * Tubing ID stays on the four standard sizes (it is a catalogue value).
  * Evaluation reports two views: in-distribution test split, and an
    independent off-grid check, so the numbers are not flattering by design.
  * The previous model is backed up as thp_model_previous.joblib.
"""

import argparse, json, os, shutil, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

try:
    from .traverse import WellInput, GradientTraverse
except ImportError:
    from traverse import WellInput, GradientTraverse

FEATURES = ["QL", "WOR", "GLR", "oil_sg", "water_sg", "gas_sg",
            "T", "API", "Bw", "d", "depth", "Pwf"]

# (low, high) for continuous inputs
CONTINUOUS = {
    "QL": (100, 1500), "WOR": (0.0, 5.0), "GLR": (100, 800),
    "oil_sg": (0.80, 0.90), "water_sg": (1.03, 1.07), "gas_sg": (0.65, 0.80),
    "T": (100, 200), "API": (25, 45), "Bw": (1.00, 1.02),
    "depth": (2000, 8000), "Pwf": (200, 2000),
}
TUBING_IDS = [1.995, 2.441, 2.992, 3.476]   # standard sizes, inches

PARAM_RANGES = {**{k: list(v) for k, v in CONTINUOUS.items()},
                "d": [1.995, 3.476]}


def run_physics(params):
    well = WellInput(**params)
    _, thp = GradientTraverse(well).run(delta_p=5.0, fine_step=1.0, fine_threshold=50.0)
    return thp


def sample(n, seed, discrete_d=True):
    """Latin hypercube over the continuous inputs; d from standard sizes."""
    keys = list(CONTINUOUS)
    lhs = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    lo = np.array([CONTINUOUS[k][0] for k in keys])
    hi = np.array([CONTINUOUS[k][1] for k in keys])
    vals = qmc.scale(lhs, lo, hi)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for i in range(n):
        p = {k: float(vals[i, j]) for j, k in enumerate(keys)}
        p["d"] = float(rng.choice(TUBING_IDS)) if discrete_d else float(rng.uniform(1.995, 3.476))
        rows.append(p)
    return rows


def label(rows, tag=""):
    """Run the physics engine on each row; drop unphysical results."""
    out, skipped = [], 0
    for i, p in enumerate(rows):
        try:
            thp = run_physics(p)
            if thp <= 0 or thp >= p["Pwf"]:
                skipped += 1
                continue
            out.append({**p, "THP": thp})
        except Exception:
            skipped += 1
        if (i + 1) % 2000 == 0:
            print(f"  {tag}{i + 1}/{len(rows)} processed, {skipped} skipped")
    return pd.DataFrame(out)


def score(model, df):
    pred = model.predict(df[FEATURES].values)
    return (round(float(mean_absolute_error(df["THP"], pred)), 2),
            round(float(r2_score(df["THP"], pred)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000, help="synthetic cases to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] Generating {args.n} cases with the corrected physics engine...")
    df = label(sample(args.n, args.seed), "train ")
    print(f"      {len(df)} valid cases in {time.time() - t0:.0f}s, "
          f"THP {df['THP'].min():.1f}-{df['THP'].max():.1f} psia")
    df.to_csv(os.path.join(args.out_dir, "synthetic_dataset.csv"), index=False)

    print("[2/4] Training XGBoost surrogate...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES].values, df["THP"].values, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    te = pd.DataFrame(X_te, columns=FEATURES); te["THP"] = y_te
    mae, r2 = score(model, te)
    print(f"      Test split ({len(te)} cases):  MAE {mae} psia   R2 {r2}")

    print("[3/4] Independent checks on fresh cases the model never saw...")
    checks = {}
    for name, rows in {
        "continuous_inputs": sample(400, args.seed + 100, discrete_d=True),
        "arbitrary_tubing_id": sample(400, args.seed + 200, discrete_d=False),
    }.items():
        chk = label(rows)
        c_mae, c_r2 = score(model, chk)
        checks[name] = {"n": int(len(chk)), "mae_psia": c_mae, "r2": c_r2}
        print(f"      {name:20s} n={len(chk):4d}  MAE {c_mae} psia   R2 {c_r2}")

    print("[4/4] Saving...")
    model_path = os.path.join(args.out_dir, "thp_model.joblib")
    if os.path.exists(model_path):
        shutil.copy(model_path, os.path.join(args.out_dir, "thp_model_previous.joblib"))
    joblib.dump(model, model_path)
    meta = {
        "features": FEATURES, "target": "THP",
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "mae_psia": mae, "r2": r2,
        "THP_min": float(df["THP"].min()), "THP_max": float(df["THP"].max()),
        "sampling": "latin_hypercube_continuous; tubing ID on standard sizes",
        "physics_engine": "Poettmann-Carpenter, Ef/K-bar per Ahmed (Ef coeff 4.9884)",
        "independent_checks": checks,
        "param_ranges": PARAM_RANGES,
    }
    with open(os.path.join(args.out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Saved thp_model.joblib and model_meta.json "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
