"""
pvt.py  —  PVT correlations
Correlations:
    Rs  : Standing (1947)
    Bo  : Dake (1978) form of Standing — matches textbook (Ahmed) values
    z   : Hall-Yarborough (1974) — more accurate than Standing-Katz chart reading
    Bg  : Standard gas law form
"""
import numpy as np


class PVT:

    @staticmethod
    def pseudo_critical(gas_sg):
        """Standing's pseudo-critical T (°R) and P (psia)."""
        Tpc = 168 + 325 * gas_sg - 12.5 * gas_sg ** 2
        Ppc = 677 + 15.0 * gas_sg - 37.5 * gas_sg ** 2
        return Tpc, Ppc

    @staticmethod
    def z_factor(P, T_R, gas_sg):
        """
        Hall-Yarborough (1974) z-factor — iterative Newton-Raphson.
        Returns z, Tpr, Ppr.
        Note: produces z ~0.7% higher than Standing-Katz chart at moderate Ppr,
        which is expected and documented. HY is analytically more accurate.
        """
        Tpc, Ppc = PVT.pseudo_critical(gas_sg)
        Tpr = T_R / Tpc
        Ppr = P / Ppc

        t  = 1.0 / Tpr
        t2 = t * t
        t3 = t2 * t

        C  = 0.06125 * Ppr * t * np.exp(-1.2 * (1.0 - t) ** 2)
        B1 = 14.76 * t  - 9.76  * t2 + 4.58  * t3
        B2 = 90.7  * t  - 242.2 * t2 + 42.4  * t3
        B3 = 2.18  + 2.82 * t

        def F(y):
            return (-C
                    + (y + y**2 + y**3 - y**4) / (1.0 - y)**3
                    - B1 * y**2
                    + B2 * y**B3)

        def dF(y):
            return ((1.0 + 4*y + 4*y**2 - 4*y**3 + y**4) / (1.0 - y)**4
                    - 2.0 * B1 * y
                    + B2 * B3 * y**(B3 - 1.0))

        y = max(C, 1e-6)   # initial guess: z ≈ 1 → y ≈ C

        for _ in range(200):
            fv  = F(y)
            dfv = dF(y)
            if abs(dfv) < 1e-15:
                break
            y_new = y - fv / dfv
            y_new = max(y_new, 1e-10)
            if abs(y_new - y) < 1e-10:
                y = y_new
                break
            y = y_new

        return C / y, Tpr, Ppr      # z = C/y (always positive)

    @staticmethod
    def Bg(P, T_R, z):
        """Gas formation volume factor, ft³/scf."""
        return 0.02827 * z * T_R / P

    @staticmethod
    def Rs(P, T_F, API, gas_sg):
        """Standing (1947) solution GOR, scf/STB."""
        x = 0.0125 * API - 0.00091 * T_F
        return gas_sg * ((P / 18.2 + 1.4) * 10 ** x) ** 1.2048

    @staticmethod
    def Bo(P, T_F, API, gas_sg, Rs_val):
        """
        Dake (1978) form of Standing — matches Ahmed textbook values.
        Bo = 0.9759 + 0.000120 * F^1.2
        where F = Rs*(γg/γo)^0.5 + 1.25*T
        """
        oil_sg = 141.5 / (131.5 + API)
        F = Rs_val * (gas_sg / oil_sg) ** 0.5 + 1.25 * T_F
        return 0.9759 + 0.000120 * F ** 1.2
