from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class NodalWell(BaseModel):
    """Well and fluid inputs. QL and Pwf are solved for, so they are not inputs."""
    WOR     : float = Field(..., ge=0, description="Water-oil ratio, STB/STB")
    GLR     : float = Field(..., gt=0, description="Gas-liquid ratio, scf/STB")
    oil_sg  : float = Field(..., gt=0)
    water_sg: float = Field(..., gt=0)
    gas_sg  : float = Field(..., gt=0)
    T       : float = Field(..., gt=0, description="Avg wellbore temperature, F")
    API     : float = Field(..., gt=0)
    Bw      : float = Field(..., gt=0)
    d       : float = Field(..., gt=0, description="Tubing inside diameter, in")
    depth   : float = Field(..., gt=0, description="Well depth, ft")


class IPRSpec(BaseModel):
    model : Literal["vogel", "fetkovich", "darcy"]
    Pr    : float = Field(..., gt=0, description="Reservoir pressure, psia")
    Qmax  : Optional[float] = Field(None, gt=0, description="Vogel: AOF, STB/day")
    PI    : Optional[float] = Field(None, gt=0, description="Darcy: productivity index, STB/day/psi")
    C     : Optional[float] = Field(None, gt=0, description="Fetkovich coefficient")
    n     : float = Field(1.0, gt=0, description="Fetkovich exponent")
    Pwf_test: Optional[float] = Field(None, ge=0, description="Fetkovich: calibrate C from a test point")
    Q_test  : Optional[float] = Field(None, gt=0)


class NodalRequest(BaseModel):
    well      : NodalWell
    ipr       : IPRSpec
    target_thp: float = Field(..., gt=0, description="Wellhead pressure the well must deliver against, psia")
    q_min     : float = Field(100, gt=0)
    q_max     : float = Field(1500, gt=0)
    n_points  : int   = Field(10, ge=4, le=25)

    class Config:
        json_schema_extra = {"example": {
            "well": {"WOR": 1.5, "GLR": 273, "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
                     "T": 150, "API": 35, "Bw": 1.01, "d": 2.441, "depth": 5000},
            "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 1800},
            "target_thp": 200, "q_min": 100, "q_max": 1500, "n_points": 10}}


class CurvePoint(BaseModel):
    Q: float
    Pwf: float
    converged: Optional[bool] = None


class OperatingPointOut(BaseModel):
    found: bool
    Q: float
    Pwf: float
    note: Optional[str] = None


class NodalResponse(BaseModel):
    operating_point: OperatingPointOut
    ipr_curve: List[CurvePoint]
    vlp_curve: List[CurvePoint]
    warnings: List[str]
    credits_used: int
    credits_remaining: int
    response_ms: float
