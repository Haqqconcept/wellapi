# WellAPI — Backend

Pay-per-use REST API for Tubing Head Pressure (THP) prediction in vertical oil wells.

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill environment config
cp .env.example .env

# 3. Run the server
uvicorn main:app --reload
```

API will be live at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

## Endpoints

| Method | Path | Auth | Cost | Description |
|---|---|---|---|---|
| GET | `/` | None | Free | Health check |
| POST | `/auth/register` | None | Free | Create account (50 free credits) |
| POST | `/auth/login` | Password | Free | Get JWT token |
| GET | `/auth/keys` | JWT | Free | List API keys |
| POST | `/auth/keys` | JWT | Free | Create new API key |
| DELETE | `/auth/keys/{id}` | JWT | Free | Deactivate API key |
| POST | `/predict/ml` | API Key | 1 credit | XGBoost surrogate prediction |
| POST | `/predict/physics` | API Key | 4 credits | Full gradient-traverse calculation |
| GET | `/account/me` | JWT | Free | Account summary + balance |
| GET | `/account/usage` | JWT | Free | Usage history |
| GET | `/account/bundles` | None | Free | Available credit bundles |
| POST | `/account/topup` | JWT | Free | Initiate payment |
| POST | `/account/topup/verify/{id}` | None | Free | Confirm payment + credit account |

---

## Making a prediction

```bash
# 1. Register and get your API key
curl -X POST https://api.wellapi.io/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","name":"Your Name","password":"yourpassword"}'

# 2. Call the ML endpoint
curl -X POST https://api.wellapi.io/predict/ml \
  -H "X-API-Key: wapi_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "QL": 330, "WOR": 1.5, "GLR": 273,
    "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
    "T": 150, "API": 35, "Bw": 1.01,
    "d": 2.441, "depth": 5000, "Pwf": 1000
  }'
```

Response:
```json
{
  "THP": 88.3,
  "unit": "psia",
  "method": "ml_surrogate",
  "confidence": "high",
  "out_of_range": [],
  "credits_used": 1,
  "credits_remaining": 49,
  "response_ms": 4.4
}
```

---

## Project structure

```
wellapi/
├── main.py              # FastAPI app entry point
├── core/
│   ├── config.py        # Settings from .env
│   ├── database.py      # SQLAlchemy session management
│   └── auth.py          # API key auth + JWT helpers
├── models/
│   ├── db.py            # ORM models (User, APIKey, Transaction, UsageLog)
│   └── schemas.py       # Pydantic request/response schemas
├── routes/
│   ├── auth.py          # Register, login, API key management
│   ├── predict.py       # /predict/ml and /predict/physics
│   └── account.py       # Balance, usage, bundles, payments
├── requirements.txt
└── .env.example
```

The `thp_engine/` folder (physics engine + ML model) must sit one level up from `wellapi/`.

---

## Deployment (Railway/Render)

```bash
# Set these environment variables in your hosting dashboard:
DATABASE_URL=postgresql://...
SECRET_KEY=your-random-secret
STRIPE_SECRET_KEY=sk_live_...
PAYSTACK_SECRET_KEY=sk_live_...
```

Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
