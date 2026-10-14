"""
routes/auth.py
--------------
POST /auth/register  — create account (50 free credits on signup)
POST /auth/login     — get JWT token
POST /auth/keys      — create a new API key
GET  /auth/keys      — list all API keys
DELETE /auth/keys/{id} — deactivate an API key
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime

from core.database import get_db
from core.config import settings
from core.auth import hash_password, verify_password, create_token, decode_token
from models.db import User, APIKey
from models.schemas import (
    UserRegister, TokenResponse, UserOut,
    APIKeyOut, CreateAPIKey
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user_jwt(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserOut, status_code=201,
             summary="Create a new WellAPI account")
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        email         = payload.email,
        name          = payload.name,
        password_hash = hash_password(payload.password),
        organisation  = payload.organisation,
        credits       = settings.FREE_CREDITS_ON_SIGNUP,
    )
    db.add(user)
    db.flush()

    # Auto-create a default API key
    key = APIKey(user_id=user.id, label="Default key")
    db.add(key)
    db.commit()
    db.refresh(user)

    return user


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse,
             summary="Login and receive a JWT access token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account inactive.")

    return TokenResponse(access_token=create_token(user.id))


# ── API Key management ────────────────────────────────────────────────────────

@router.get("/keys", response_model=list[APIKeyOut],
            summary="List all your API keys")
def list_keys(user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    return db.query(APIKey).filter(APIKey.user_id == user.id).all()


@router.post("/keys", response_model=APIKeyOut, status_code=201,
             summary="Generate a new API key")
def create_key(
    payload: CreateAPIKey,
    user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    key = APIKey(user_id=user.id, label=payload.label)
    db.add(key)
    db.commit()
    db.refresh(key)
    return key


@router.delete("/keys/{key_id}", status_code=204,
               summary="Deactivate an API key")
def delete_key(
    key_id: str,
    user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    key = db.query(APIKey).filter(
        APIKey.id == key_id, APIKey.user_id == user.id
    ).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found.")
    key.is_active = False
    db.commit()
