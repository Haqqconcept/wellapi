"""
core/auth.py
------------
API key authentication middleware and password hashing.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.config import settings
from core.database import get_db
from models.db import User, APIKey

# ── Password hashing ──────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    secret = password.encode("utf-8")[:72]
    return pwd_context.hash(secret)


def verify_password(plain: str, hashed: str) -> bool:
    secret = plain.encode("utf-8")[:72]
    return pwd_context.verify(secret, hashed)


# ── JWT tokens (for web dashboard) ───────────────────────────────────────────

ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24


def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": user_id, "exp": expire}, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ── API Key authentication ────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user_by_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency — validates X-API-Key header and returns the User.
    Raises 401 if missing or invalid, 403 if inactive, 402 if no credits.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    key_record = db.query(APIKey).filter(
        APIKey.key == api_key,
        APIKey.is_active == True
    ).first()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key."
        )

    user = key_record.user
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact support."
        )

    # Update last used timestamp
    key_record.last_used = datetime.utcnow()
    db.commit()

    return user


def require_credits(cost: int):
    """
    Dependency factory — checks user has enough credits before prediction.
    Usage: Depends(require_credits(1))
    """
    def check(user: User = Depends(get_current_user_by_api_key)):
        if user.credits < cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. You have {user.credits} credit(s) but this endpoint costs {cost}. Top up at wellapi.io/topup"
            )
        return user
    return check
