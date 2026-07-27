"""
validate_example716.py
----------------------
Validates the physics engine against Example 7-16 from:
    Tarek Ahmed, Reservoir Engineering Handbook, Chapter 7.

Expected result from textbook: THP ≈ 110 psi
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pvt import PVT
from traverse import WellInput, GradientTraverse
from tabulate import tabulate


# ── Input data from Example 7-16 ──────────────────────────────────────────────

well = WellInput(
    QL       = 330,    # STB/day
    WOR      = 1.5,    # STB/STB
    GLR      = 273,    # scf/STB  (gas-liquid ratio)
    oil_sg   = 0.85,
    water_sg = 1.05,
    gas_sg   = 0.75,
    T        = 150,    # °F
    API      = 35,
    Bw       = 1.01,   # bbl/STB
    d        = 2.441,  # inches
    depth    = 5000,   # ft
    Pwf      = 1000,   # psia
)


# ── Preliminary calculations (Step 2 in textbook) ────────────────────────────

engine  = GradientTraverse(well)
T_R     = well.T + 460
Tpc, Ppc = PVT.pseudo_critical(well.gas_sg)
z0, Tpr, Ppr0 = PVT.z_factor(well.Pwf, T_R, well.gas_sg)

print("=" * 65)
print("  VALIDATION: Example 7-16 — Reservoir Engineering Handbook")
print("=" * 65)

print("\n── Preliminary Calculations ──────────────────────────────────")
print(f"  fw  = {well.fw:.4f}          (textbook: 0.6)")
print(f"  fo  = {well.fo:.4f}          (textbook: 0.4)")
print(f"  Tpc = {Tpc:.1f} °R        (textbook: 405 °R)")
print(f"  Tpr = {Tpr:.4f}         (textbook: 1.51)")
print(f"  Ppc = {Ppc:.1f} psi       (textbook: 667 psi)")
print(f"  M   = {engine.M:.4f} lbm/STB (textbook: 355.45)")
print(f"  Dvp = {engine.Dvp:.4f}         (textbook: 8.498)")
print(f"  f   = {engine.f:.6f}")
print(f"  K   = {engine.K:.4e}")


# ── Run the traverse ──────────────────────────────────────────────────────────

table, THP = engine.run(delta_p=5.0, fine_step=1.0, fine_threshold=50.0)

print(f"\n── Traverse Table (first 8 rows and last 5 rows) ─────────────")

display_cols = ["P_assumed", "P_avg", "Rs", "Bo", "Ppr", "z", "Bg", "Avg_rho", "delta_h", "depth_to_top"]

head = table[display_cols].head(8)
tail = table[display_cols].tail(5)

print(tabulate(head, headers=display_cols, tablefmt="rounded_outline", floatfmt=".4f", showindex=False))
print("  ...")
print(tabulate(tail, headers=display_cols, tablefmt="rounded_outline", floatfmt=".4f", showindex=False))

print(f"\n── Result ────────────────────────────────────────────────────")
print(f"  Computed THP : {THP} psia")
print(f"  Textbook THP : ~110 psia")
print(f"  Difference   : {abs(THP - 110):.1f} psia")
print(f"  Total steps  : {len(table)}")

if abs(THP - 110) <= 10:
    print("\n  ✓ Result is within acceptable range of textbook answer.")
else:
    print("\n  ✗ Result deviates from textbook — check correlations.")

print("=" * 65)
