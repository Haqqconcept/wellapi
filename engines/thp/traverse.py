"""
traverse.py
-----------
Poettmann-Carpenter gradient-traverse method for tubing head pressure (THP)
calculation, as described in:

    Tarek Ahmed, Reservoir Engineering Handbook, Chapter 7 (Example 7-16)

The method steps upward through the well in small pressure increments,
computing the depth increment at each step until the total well depth
is accounted for. The pressure at that point is the THP.
"""

import numpy as np
import pandas as pd
try:
    from .pvt import PVT      # package layout: engines/thp/
except ImportError:
    from pvt import PVT       # flat layout: thp_engine/


class WellInput:
    """
    All input parameters required for a THP traverse calculation.
    """

    def __init__(
        self,
        QL,          # Total liquid rate, STB/day
        WOR,         # Water-oil ratio, STB/STB
        GLR,         # Gas-liquid ratio, scf/STB
        oil_sg,      # Oil specific gravity (60°F/60°F)
        water_sg,    # Water specific gravity
        gas_sg,      # Gas specific gravity (air = 1)
        T,           # Average wellbore temperature, °F
        API,         # Oil API gravity
        Bw,          # Water FVF, bbl/STB
        d,           # Tubing inside diameter, inches
        depth,       # Well depth (measured), ft
        Pwf,         # Bottom-hole flowing pressure, psia
    ):
        self.QL       = QL
        self.WOR      = WOR
        self.GLR      = GLR
        self.oil_sg   = oil_sg
        self.water_sg = water_sg
        self.gas_sg   = gas_sg
        self.T        = T
        self.API      = API
        self.Bw       = Bw
        self.d        = d
        self.depth    = depth
        self.Pwf      = Pwf

        # Derived fractions
        self.fw = WOR / (WOR + 1)
        self.fo = 1.0 - self.fw


class GradientTraverse:
    """
    Implements the average-pressure gradient traverse (Poettmann-Carpenter).

    Usage
    -----
        well = WellInput(...)
        engine = GradientTraverse(well)
        table, THP = engine.run()
    """

    def __init__(self, well: WellInput):
        self.w = well
        self.T_R = well.T + 460.0  # Convert to Rankine

        Tpc, Ppc = PVT.pseudo_critical(well.gas_sg)
        self.Tpc = Tpc
        self.Ppc = Ppc

        # Mixture molecular weight term (lbm/STB)
        # M = 350.376[fo*γo + fw*γw] + 0.0763*γg*GLR
        self.M = (
            350.376 * (well.fo * well.oil_sg + well.fw * well.water_sg)
            + 0.0763 * well.gas_sg * well.GLR
        )

        # Velocity number (dimensionless)
        # Dvp = 176.844e-6 * M * QL / d
        self.Dvp = 176.844e-6 * self.M * well.QL / well.d

        # Energy-loss factor E_f (Ahmed, Reservoir Engineering Handbook,
        # Poettmann-Carpenter method).
        # NOTE: the textbook prints the leading coefficient as 46.0115.
        # That is a misprint: the correct value is 4.9884. Verified by
        # reproducing the book's own worked example exactly:
        #   Dvp = 8.498  ->  E_f = 0.006678
        self.Ef = self._pc_energy_loss_factor(self.Dvp)

        # K-bar = 3.3567e-6 * QL^2 * M^2 * E_f / d^5   (d in inches)
        # QL appears explicitly, which is what gives THP its dependence
        # on flow rate (the previous formula had none).
        self.K = 3.3567e-6 * (well.QL ** 2) * (self.M ** 2) * self.Ef / (well.d ** 5)

    def _pc_energy_loss_factor(self, Dvp):
        """E_f = 4.9884*Dvp^-3.092368 + 60.37e-6*(Dvp^-1 + 0.00524355)"""
        if Dvp <= 0:
            return 0.0
        return 4.9884 * Dvp ** (-3.092368) + 60.37e-6 * (Dvp ** -1 + 0.00524355)

    def _mixture_density(self, P_avg):
        """
        Average mixture density at P_avg using PVT correlations.
        Returns rho (lbm/ft³) and intermediate PVT values.
        """
        w = self.w

        Rs_val  = PVT.Rs(P_avg, w.T, w.API, w.gas_sg)
        Bo_val  = PVT.Bo(P_avg, w.T, w.API, w.gas_sg, Rs_val)
        z, _, _ = PVT.z_factor(P_avg, self.T_R, w.gas_sg)
        Bg_val  = PVT.Bg(P_avg, self.T_R, z)

        # Free gas remaining in gas phase
        free_gas = max(w.GLR - w.fo * Rs_val, 0.0)

        # Volume occupied per STB of liquid
        vol = 5.614 * (w.fo * Bo_val + w.fw * w.Bw) + Bg_val * free_gas

        rho = self.M / vol
        return rho, Rs_val, Bo_val, z, Bg_val

    def _depth_increment(self, rho_avg, delta_p):
        """
        Depth increment (ft) corresponding to a pressure drop of delta_p (psi).

        Δh = 144.9 × Δp / (ρ̄ + K/ρ̄)
        """
        return 144.9 * delta_p / (rho_avg + self.K / rho_avg)

    def run(self, delta_p=5.0, fine_step=1.0, fine_threshold=50.0):
        """
        Run the traverse from Pwf upward to the surface.

        Parameters
        ----------
        delta_p        : normal pressure step, psi (default 5)
        fine_step      : reduced step near surface, psi (default 1)
        fine_threshold : switch to fine step when depth_remaining < this, ft

        Returns
        -------
        table : pandas DataFrame with one row per step
        THP   : tubing head pressure, psia
        """
        w = self.w
        rows = []

        P              = w.Pwf
        depth_remaining = w.depth

        while depth_remaining > 0 and P > 0:

            # Use finer steps near the surface
            dp = fine_step if depth_remaining < fine_threshold else delta_p

            P_next  = max(P - dp, 0.0)
            P_avg   = (P + P_next) / 2.0

            rho, Rs_val, Bo_val, z, Bg_val = self._mixture_density(P_avg)
            dh = self._depth_increment(rho, dp)

            rows.append({
                "P_assumed"    : round(P, 3),
                "P_avg"        : round(P_avg, 3),
                "Rs"           : round(Rs_val, 3),
                "Bo"           : round(Bo_val, 4),
                "Ppr"          : round(P_avg / self.Ppc, 4),
                "z"            : round(z, 6),
                "Bg"           : round(Bg_val, 6),
                "Avg_rho"      : round(rho, 4),
                "delta_h"      : round(dh, 3),
                "depth_to_top" : round(depth_remaining, 3),
            })

            depth_remaining -= dh
            P = P_next

            if P_next == 0:
                break

        table = pd.DataFrame(rows)
        THP   = round(P, 1)
        return table, THP
