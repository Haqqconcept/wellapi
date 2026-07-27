"""
models/schemas.py
-----------------
Pydantic models for request validation and response serialization.
"""

from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)
    organisation: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    organisation: Optional[str]
    credits: int
    total_calls: int
    created_at: datetime

    class Config:
        from_attributes = True


# ── API Keys ──────────────────────────────────────────────────────────────────

class APIKeyOut(BaseModel):
    id: str
    key: str
    label: str
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime]

    class Config:
        from_attributes = True


class CreateAPIKey(BaseModel):
    label: str = "Default key"


# ── Predictions ───────────────────────────────────────────────────────────────

class WellInput(BaseModel):
    QL      : float = Field(..., gt=0,   description="Total liquid rate, STB/day")
    WOR     : float = Field(..., ge=0,   description="Water-oil ratio, STB/STB")
    GLR     : float = Field(..., gt=0,   description="Gas-liquid ratio, scf/STB")
    oil_sg  : float = Field(..., gt=0,   description="Oil specific gravity (60/60°F)")
    water_sg: float = Field(..., gt=0,   description="Water specific gravity")
    gas_sg  : float = Field(..., gt=0,   description="Gas specific gravity (air=1)")
    T       : float = Field(..., gt=0,   description="Avg wellbore temperature, °F")
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


# ── Account ───────────────────────────────────────────────────────────────────

class UsageLogOut(BaseModel):
    id           : str
    endpoint     : str
    credits_used : int
    thp_result   : Optional[float]
    confidence   : Optional[str]
    response_ms  : Optional[float]
    created_at   : datetime

    class Config:
        from_attributes = True


class BundleInfo(BaseModel):
    bundle_id   : str
    credits     : int
    price_usd   : float
    description : str


class TopupRequest(BaseModel):
    bundle      : str = Field(..., description="Bundle ID: starter, basic, professional, enterprise, academic")
    gateway     : str = Field(..., description="Payment gateway: stripe | paystack")


class TopupResponse(BaseModel):
    payment_url : str
    reference   : str
    bundle      : str
    credits     : int
    amount_usd  : float


class AccountSummary(BaseModel):
    user         : UserOut
    credits      : int
    total_calls  : int
    api_keys     : List[APIKeyOut]
    recent_usage : List[UsageLogOut]
