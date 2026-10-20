"""Inflow Performance Relationship (IPR) models: Vogel, Fetkovich, Darcy.
Each exposes q_at_pwf(pwf) and pwf_at_q(q) in closed form."""
import math


class VogelIPR:
    """Q/Qmax = 1 - 0.2(Pwf/Pr) - 0.8(Pwf/Pr)^2"""
    def __init__(self, Pr, Qmax):
        if Pr <= 0 or Qmax <= 0:
            raise ValueError("Pr and Qmax must be positive")
        self.Pr, self.Qmax = Pr, Qmax

    def q_at_pwf(self, pwf):
        r = max(0.0, min(pwf, self.Pr)) / self.Pr
        return self.Qmax * (1 - 0.2 * r - 0.8 * r * r)

    def pwf_at_q(self, q):
        q = max(0.0, min(q, self.Qmax))
        c = -(1 - q / self.Qmax)
        disc = max(0.04 - 4 * 0.8 * c, 0.0)
        x = (-0.2 + math.sqrt(disc)) / 1.6
        return max(0.0, min(x, 1.0)) * self.Pr


class FetkovichIPR:
    """Q = C (Pr^2 - Pwf^2)^n"""
    def __init__(self, Pr, C, n=1.0):
        if Pr <= 0 or C <= 0 or n <= 0:
            raise ValueError("Pr, C and n must be positive")
        self.Pr, self.C, self.n = Pr, C, n

    @classmethod
    def from_test_point(cls, Pr, Pwf_test, Q_test, n=1.0):
        if not (0 <= Pwf_test < Pr):
            raise ValueError("Pwf_test must be between 0 and Pr")
        return cls(Pr, Q_test / ((Pr ** 2 - Pwf_test ** 2) ** n), n)

    def q_at_pwf(self, pwf):
        pwf = max(0.0, min(pwf, self.Pr))
        return self.C * max(self.Pr ** 2 - pwf ** 2, 0.0) ** self.n

    def pwf_at_q(self, q):
        d = (max(q, 0.0) / self.C) ** (1 / self.n)
        return math.sqrt(max(self.Pr ** 2 - d, 0.0))


class DarcyIPR:
    """Q = PI (Pr - Pwf)"""
    def __init__(self, Pr, PI):
        if Pr <= 0 or PI <= 0:
            raise ValueError("Pr and PI must be positive")
        self.Pr, self.PI = Pr, PI

    def q_at_pwf(self, pwf):
        return self.PI * (self.Pr - max(0.0, min(pwf, self.Pr)))

    def pwf_at_q(self, q):
        return max(self.Pr - max(q, 0.0) / self.PI, 0.0)


IPR_MODELS = {"vogel": VogelIPR, "fetkovich": FetkovichIPR, "darcy": DarcyIPR}


def build_ipr(model, **kw):
    model = model.lower()
    if model not in IPR_MODELS:
        raise ValueError(f"Unknown IPR model '{model}'. Choose from {list(IPR_MODELS)}")
    return IPR_MODELS[model](**kw)
