"""
main.py
-------
WellAPI — Pay-Per-Use Well Performance Calculation API
Entry point for the FastAPI application.

Run with:
    uvicorn main:app --reload
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'thp_engine'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.database import engine, Base
from routes import predict, auth, account

# ── Create database tables ────────────────────────────────────────────────────

Base.metadata.create_all(bind=engine)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.APP_NAME,
    version     = settings.APP_VERSION,
    description = """
## WellAPI — Tubing Head Pressure Prediction API

Pay-per-use API for vertical oil well performance calculations.
Built on the Poettmann-Carpenter gradient-traverse method with an XGBoost surrogate model (R² = 0.9991).

### Quick Start

1. Register at `/auth/register` — you get **50 free credits** immediately
2. Copy your API key from the response
3. Include `X-API-Key: wapi_xxxxx` in every prediction request
4. Call `/predict/ml` (1 credit) or `/predict/physics` (4 credits)
5. Top up credits at `/account/topup` via Stripe or Paystack

### Pricing
| Endpoint | Credits | Speed |
|---|---|---|
| `/predict/ml` | 1 credit | < 10 ms |
| `/predict/physics` | 4 credits | 50–200 ms |

### Credit Bundles
| Bundle | Credits | Price |
|---|---|---|
| Starter | 100 | $5 |
| Basic | 500 | $20 |
| Professional | 2,000 | $75 |
| Enterprise | 8,000 | $250 |
| Academic | 1,200 | $30/semester |
    """,
    contact = {
        "name" : "Abdulhakeem Aljazari",
        "email": "abdulhaqqabdulhakeem3@gmail.com",
        "url"  : "https://wellapi.io",
    },
    license_info = {
        "name": "MIT License",
        "url" : "https://opensource.org/licenses/MIT",
    }
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(account.router)

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="API status and version")
def root():
    return {
        "service"    : settings.APP_NAME,
        "version"    : settings.APP_VERSION,
        "status"     : "running",
        "docs"       : "/docs",
        "endpoints"  : {
            "register"       : "POST /auth/register",
            "login"          : "POST /auth/login",
            "predict_ml"     : "POST /predict/ml",
            "predict_physics": "POST /predict/physics",
            "account"        : "GET  /account/me",
            "bundles"        : "GET  /account/bundles",
            "topup"          : "POST /account/topup",
        }
    }


@app.get("/health", tags=["Health"], summary="Liveness check")
def health():
    return {"status": "ok"}
