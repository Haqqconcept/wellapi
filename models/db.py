"""
models/db.py
------------
SQLAlchemy ORM models.
Tables: users, api_keys, transactions, usage_logs
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


def gen_api_key():
    """Generate a WellAPI key in the format wapi_xxxxxxxxxxxx"""
    return "wapi_" + uuid.uuid4().hex[:24]


class User(Base):
    __tablename__ = "users"

    id             = Column(String, primary_key=True, default=gen_uuid)
    email          = Column(String, unique=True, nullable=False, index=True)
    name           = Column(String, nullable=False)
    password_hash  = Column(String, nullable=False)
    organisation   = Column(String, nullable=True)
    is_active      = Column(Boolean, default=True)
    is_verified    = Column(Boolean, default=False)
    credits        = Column(Integer, default=50)         # free credits on signup
    total_calls    = Column(Integer, default=0)
    created_at     = Column(DateTime, default=datetime.utcnow)

    api_keys       = relationship("APIKey", back_populates="user")
    transactions   = relationship("Transaction", back_populates="user")
    usage_logs     = relationship("UsageLog", back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"

    id          = Column(String, primary_key=True, default=gen_uuid)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    key         = Column(String, unique=True, nullable=False, default=gen_api_key, index=True)
    label       = Column(String, default="Default key")
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    last_used   = Column(DateTime, nullable=True)

    user        = relationship("User", back_populates="api_keys")


class Transaction(Base):
    __tablename__ = "transactions"

    id              = Column(String, primary_key=True, default=gen_uuid)
    user_id         = Column(String, ForeignKey("users.id"), nullable=False)
    bundle          = Column(String, nullable=False)        # e.g. "professional"
    credits_added   = Column(Integer, nullable=False)
    amount_usd      = Column(Float, nullable=False)
    payment_ref     = Column(String, nullable=True)         # Stripe/Paystack reference
    payment_gateway = Column(String, nullable=True)         # "stripe" | "paystack"
    status          = Column(String, default="pending")     # pending | completed | failed
    created_at      = Column(DateTime, default=datetime.utcnow)

    user            = relationship("User", back_populates="transactions")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id              = Column(String, primary_key=True, default=gen_uuid)
    user_id         = Column(String, ForeignKey("users.id"), nullable=False)
    endpoint        = Column(String, nullable=False)        # "ml" | "physics"
    credits_used    = Column(Integer, nullable=False)
    credits_before  = Column(Integer, nullable=False)
    credits_after   = Column(Integer, nullable=False)
    response_ms     = Column(Float, nullable=True)
    # Store input params as JSON string for audit purposes
    input_params    = Column(Text, nullable=True)
    thp_result      = Column(Float, nullable=True)
    confidence      = Column(String, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    user            = relationship("User", back_populates="usage_logs")
