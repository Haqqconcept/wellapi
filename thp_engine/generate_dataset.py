"""
generate_dataset.py
-------------------
Generates a synthetic dataset for ML surrogate model training.
Each row = one well configuration → THP computed by the physics engine.

Parameter ranges are based on realistic field values from published literature.
Ref: Tarek Ahmed (2010), Brill & Mukherjee (1986), SPE field data ranges.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from itertools import product
from traverse import WellInput, GradientTraverse

# ── Parameter ranges ──────────────────────────────────────────────────────────

PARAM_RANGES = {
    "QL"       : (100, 1500, 8),    # Total liquid rate, STB/day
    "WOR"      : (0.0, 5.0,  6),    # Water-oil ratio
    "GLR"      : (100, 800,  8),    # Gas-liquid ratio, scf/STB
    "oil_sg"   : (0.80, 0.90, 3),   # Oil specific gravity
    "water_sg" : [1.03, 1.05, 1.07],# Water specific gravity (fixed options)
    "gas_sg"   : (0.65, 0.80, 4),   # Gas specific gravity
    "T"        : (100, 200, 5),     # Avg wellbore temperature, °F
    "API"      : (25, 45, 5),       # API gravity
    "Bw"       : [1.00, 1.01, 1.02],# Water FVF (fixed options)
    "d"        : [1.995, 2.441, 2.992, 3.476],  # Tubing ID (standard sizes), in
    "depth"    : (2000, 8000, 7),   # Well depth, ft
    "Pwf"      : (200, 2000, 10),   # Bottom-hole flowing pressure, psia
}


def _linspace_or_list(spec):
    """Helper: convert (min, max, n) tuples to arrays, pass lists as-is."""
    if isinstance(spec, list):
        return np.array(spec)
    lo, hi, n = spec
    return np.linspace(lo, hi, n)


def generate(n_samples=5000, seed=42, delta_p=5.0):
    """
    Latin Hypercube-style random sampling across parameter space.

    Parameters
    ----------
    n_samples : number of well cases to generate
    seed      : random seed for reproducibility
    delta_p   : pressure step for gradient traverse, psi

    Returns
    -------
    DataFrame with input parameters and computed THP
    """
    rng = np.random.default_rng(seed)

    # Build discrete option arrays for each parameter
    options = {k: _linspace_or_list(v) for k, v in PARAM_RANGES.items()}

    records = []
    skipped = 0

    print(f"Generating {n_samples} synthetic well cases...")

    for i in range(n_samples):
        # Sample one value per parameter
        params = {k: rng.choice(v) for k, v in options.items()}

        # Basic physical consistency checks before running traverse
        # 1. GLR must be > Rs(Pwf) * fo, otherwise no free gas at wellbore
        #    (not strictly required but avoids degenerate cases)
        # 2. Pwf must give physically meaningful conditions

        try:
            well = WellInput(**params)
            engine = GradientTraverse(well)
            _, THP = engine.run(delta_p=delta_p, fine_step=1.0, fine_threshold=50.0)

            # Filter out unphysical results
            if THP <= 0 or THP >= params["Pwf"]:
                skipped += 1
                continue

            record = dict(params)
            record["THP"] = THP
            records.append(record)

        except Exception:
            skipped += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_samples} cases processed, {skipped} skipped...")

    df = pd.DataFrame(records)
    print(f"\nDone. {len(df)} valid cases generated, {skipped} skipped.")
    return df


if __name__ == "__main__":
    df = generate(n_samples=5000, seed=42)

    print(f"\nDataset shape : {df.shape}")
    print(f"THP range     : {df['THP'].min():.1f} — {df['THP'].max():.1f} psia")
    print(f"THP mean      : {df['THP'].mean():.1f} psia")
    print(f"THP std       : {df['THP'].std():.1f} psia")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    out_path = os.path.join(os.path.dirname(__file__), "synthetic_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
