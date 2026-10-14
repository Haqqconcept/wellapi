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


# NOTE: THP-specific schemas (WellInput, MLPredictionResponse, TraverseStep,
# PhysicsPredictionResponse) moved to engines/thp/schema.py — this file now
# holds only schemas shared across engines (auth, account, billing).

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
