"""
engines/thp/schema.py
----------------------
Pydantic request/response models for the THP engine.
Moved out of the old shared models/schemas.py so this engine is self-contained.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class WellInput(BaseModel):
    QL      : float = Field(..., gt=0,   description="Total liquid rate, STB/day")
    WOR     : float = Field(..., ge=0,   description="Water-oil ratio, STB/STB")
    GLR     : float = Field(..., gt=0,   description="Gas-liquid ratio, scf/STB")
    oil_sg  : float = Field(..., gt=0,   description="Oil specific gravity (60/60\u00b0F)")
    water_sg: float = Field(..., gt=0,   description="Water specific gravity")
    gas_sg  : float = Field(..., gt=0,   description="Gas specific gravity (air=1)")
    T       : float = Field(..., gt=0,   description="Avg wellbore temperature, \u00b0F")
    API     : float = Field(..., gt=0,   description="Oil API gravity")
    Bw      : float = Field(..., gt=0,   description="Water formation volume factor, bbl/STB")
    d       : float = Field(..., gt=0,   description="Tubing inside diameter, inches")
    depth   : float = Field(..., gt=0,   description="Well depth, ft")
    Pwf     : float = Field(..., gt=0,   description="Bottom-hole flowing pressure, psia")

    class Config:
        json_schema_extra = {
            "example": {
                "QL": 330, "WOR": 1.5, "GLR": 273,
                "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
                "T": 150, "API": 35, "Bw": 1.01,
                "d": 2.441, "depth": 5000, "Pwf": 1000
            }
        }


class MLPredictionResponse(BaseModel):
    THP               : float
    unit              : str = "psia"
    method            : str = "ml_surrogate"
    confidence        : str
    out_of_range      : List[str]
    credits_used      : int
    credits_remaining : int
    response_ms       : float


class TraverseStep(BaseModel):
    P_assumed   : float
    P_avg       : float
    Rs          : float
    Bo          : float
    Ppr         : float
    z           : float
    Bg          : float
    Avg_rho     : float
    delta_h     : float
    depth_to_top: float


class PhysicsPredictionResponse(BaseModel):
    THP               : float
    unit              : str = "psia"
    method            : str = "physics_engine"
    confidence        : str
    out_of_range      : List[str]
    traverse_steps    : int
    traverse_table    : Optional[List[TraverseStep]] = None
    credits_used      : int
    credits_remaining : int
    response_ms       : float
