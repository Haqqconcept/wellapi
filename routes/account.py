"""
routes/account.py
-----------------
GET  /account/me       — profile + balance
GET  /account/usage    — usage history
GET  /account/bundles  — available credit bundles
POST /account/topup    — initiate payment (Stripe or Paystack)
POST /account/topup/verify — verify payment and credit account
"""

import stripe, requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.config import settings
from models.db import User, UsageLog, Transaction
from models.schemas import (
    AccountSummary, UsageLogOut, BundleInfo,
    TopupRequest, TopupResponse, UserOut
)
from routes.auth import get_current_user_jwt

router = APIRouter(prefix="/account", tags=["Account"])


# ── Profile & balance ─────────────────────────────────────────────────────────

@router.get("/me", response_model=AccountSummary,
            summary="Your account summary, balance, and recent usage")
def get_account(user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    recent = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id)
        .order_by(UsageLog.created_at.desc())
        .limit(20)
        .all()
    )
    return AccountSummary(
        user         = UserOut.model_validate(user),
        credits      = user.credits,
        total_calls  = user.total_calls,
        api_keys     = [k for k in user.api_keys if k.is_active],
        recent_usage = recent,
    )


# ── Usage history ─────────────────────────────────────────────────────────────

@router.get("/usage", response_model=list[UsageLogOut],
            summary="Full usage history (most recent first)")
def get_usage(
    limit: int = 50,
    user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    return (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id)
        .order_by(UsageLog.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )


# ── Available bundles ─────────────────────────────────────────────────────────

@router.get("/bundles", response_model=list[BundleInfo],
            summary="Available credit bundles and pricing")
def get_bundles():
    descriptions = {
        "starter"     : "100 ML calls — ideal for students and exploration",
        "basic"       : "500 ML calls — perfect for independent consultants",
        "professional": "2,000 ML calls — for active consulting projects",
        "enterprise"  : "8,000 ML calls — for companies and research groups",
        "academic"    : "1,200 mixed calls — university departments (per semester)",
    }
    return [
        BundleInfo(
            bundle_id   = bid,
            credits     = credits,
            price_usd   = price,
            description = descriptions.get(bid, "")
        )
        for bid, (credits, price) in settings.BUNDLES.items()
    ]


# ── Top-up via Stripe ─────────────────────────────────────────────────────────

@router.post("/topup", response_model=TopupResponse,
             summary="Initiate a credit purchase — Stripe (global) or Paystack (Africa)")
def topup(
    payload: TopupRequest,
    user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    if payload.bundle not in settings.BUNDLES:
        raise HTTPException(status_code=400, detail=f"Unknown bundle: {payload.bundle}")

    credits, price_usd = settings.BUNDLES[payload.bundle]

    # Create pending transaction
    txn = Transaction(
        user_id         = user.id,
        bundle          = payload.bundle,
        credits_added   = credits,
        amount_usd      = price_usd,
        payment_gateway = payload.gateway,
        status          = "pending",
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    if payload.gateway == "stripe":
        if not settings.STRIPE_SECRET_KEY:
            raise HTTPException(status_code=503, detail="Stripe not configured yet.")
        stripe.api_key = settings.STRIPE_SECRET_KEY
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"WellAPI {payload.bundle.title()} Bundle ({credits} credits)"},
                    "unit_amount": int(price_usd * 100),
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=f"https://wellapi.io/success?ref={txn.id}",
            cancel_url ="https://wellapi.io/topup",
            metadata   ={"transaction_id": txn.id, "user_id": user.id},
        )
        txn.payment_ref = session.id
        db.commit()
        return TopupResponse(
            payment_url = session.url,
            reference   = txn.id,
            bundle      = payload.bundle,
            credits     = credits,
            amount_usd  = price_usd,
        )

    elif payload.gateway == "paystack":
        if not settings.PAYSTACK_SECRET_KEY:
            raise HTTPException(status_code=503, detail="Paystack not configured yet.")
        resp = requests.post(
            "https://api.paystack.co/transaction/initialize",
            headers={"Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}"},
            json={
                "email"    : user.email,
                "amount"   : int(price_usd * 100 * 1600),  # approx NGN at ₦1600/USD
                "currency" : "NGN",
                "reference": txn.id,
                "metadata" : {"bundle": payload.bundle, "credits": credits},
            }
        ).json()
        if not resp.get("status"):
            raise HTTPException(status_code=502, detail="Paystack initialization failed.")
        txn.payment_ref = resp["data"]["reference"]
        db.commit()
        return TopupResponse(
            payment_url = resp["data"]["authorization_url"],
            reference   = txn.id,
            bundle      = payload.bundle,
            credits     = credits,
            amount_usd  = price_usd,
        )

    raise HTTPException(status_code=400, detail="Gateway must be 'stripe' or 'paystack'.")


# ── Payment verification (webhook / callback) ─────────────────────────────────

@router.post("/topup/verify/{transaction_id}",
             summary="Verify a completed payment and credit your account")
def verify_topup(
    transaction_id: str,
    db: Session = Depends(get_db)
):
    txn = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    if txn.status == "completed":
        return {"message": "Already credited.", "credits_added": txn.credits_added}

    # Verify with Paystack
    if txn.payment_gateway == "paystack" and settings.PAYSTACK_SECRET_KEY:
        resp = requests.get(
            f"https://api.paystack.co/transaction/verify/{txn.payment_ref}",
            headers={"Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}"}
        ).json()
        if resp.get("data", {}).get("status") != "success":
            raise HTTPException(status_code=402, detail="Payment not yet confirmed.")

    # Credit the user
    user = db.query(User).filter(User.id == txn.user_id).first()
    user.credits += txn.credits_added
    txn.status    = "completed"
    db.commit()

    return {
        "message"      : f"✓ {txn.credits_added} credits added to your account.",
        "credits_added": txn.credits_added,
        "new_balance"  : user.credits,
    }
