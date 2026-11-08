# WellAPI

**Physics-informed machine learning for well performance.** Predict tubing head pressure (THP) for vertical oil wells and run nodal analysis through a simple API, without a software licence.

> **Status: research prototype.** All results below compare the machine-learning model with its own physics engine on synthetic data. The model has **not** been validated against field data or commercial software. Do not use it for critical engineering decisions yet.

Presented as a poster at **BUILD26: Africa's Shared Compute and AI Infrastructure Conference** (Abuja, October 2026).

---

## Why this exists

THP is a basic quantity in production engineering, used to monitor wells, size equipment and plan artificial lift. The tools that calculate it (PIPESIM, PROSPER) are licensed, desktop-bound and expensive, which puts them out of reach for many independent operators, consultants and students in Africa.

WellAPI explores a different route: build a physics engine from published correlations, use it to generate training data (so no proprietary field data is needed), train a fast surrogate model, and serve both behind an open, auditable API.

## How it works

```
 TRAINING (offline)
 12 inputs ──▶ Physics engine ──▶ 12,000 Latin-hypercube cases ──▶ 11,300 valid ──▶ XGBoost surrogate
                (Poettmann-Carpenter)                              (700 discarded)

 EVERY API REQUEST
 12 inputs ──▶ Range check ──┬─ all inputs inside trained range ──▶ XGBoost surrogate ──▶ THP
                             └─ any input outside  ───────────────▶ Physics engine   ──▶ THP (flagged low-confidence)
```

**Inputs (12):** liquid rate, water-oil ratio, gas-liquid ratio, oil / water / gas specific gravity, temperature, oil API gravity, water formation volume factor, tubing inside diameter, well depth, bottomhole flowing pressure.

## Results

Reference for every number below is the project's own physics engine, not real wells.

| Evaluation set | Cases | MAE (psia) | R² |
|---|---:|---:|---:|
| Held-out test split | 2,260 | 10.42 | 0.9978 |
| Fresh cases, continuous inputs | 375 | 11.39 | 0.9971 |
| Fresh cases, arbitrary tubing diameter | 378 | 10.92 | 0.9971 |

The two "fresh" sets use random seeds never used in training. The surrogate is an XGBoost regressor (700 trees, depth 5, learning rate 0.08) trained on 9,040 cases.

**Speed** (author's laptop, Windows, Python 3.13; varies with hardware):

| | Physics engine | Surrogate | Gain |
|---|---:|---:|---:|
| One well | 11.6 ms | 2.9 ms | about 4x |
| 10,000 wells (batch) | about 116 s | 89 ms | about 1,300x |

For a single well the gain is small. It matters for batch screening, sweeps and nodal analysis.

## Outside the trained range

Tree models cannot extrapolate: beyond their training data, predictions go flat and can be badly wrong while still looking plausible. Mean absolute error against the physics engine, with a neural network (MLP) trained on the same data for comparison:

| Input | Trained range | Tested range | XGBoost | Neural net |
|---|---|---|---:|---:|
| (in range) | | | 10.42 | 10.17 |
| Liquid rate (STB/day) | 100 - 1,500 | 1,600 - 2,200 | 11.9 | 15.0 |
| Liquid rate (STB/day) | 100 - 1,500 | 2,200 - 3,000 | 12.0 | 22.2 |
| Well depth (ft) | 2,000 - 8,000 | 8,500 - 10,000 | 57.3 | 10.6 |
| Bottomhole pressure (psia) | 200 - 2,000 | 2,100 - 2,600 | 214.3 | 30.5 |
| Bottomhole pressure (psia) | 200 - 2,000 | 2,600 - 3,200 | 620.7 | 91.9 |
| Gas-liquid ratio (scf/STB) | 100 - 800 | 850 - 1,100 | 55.2 | 14.8 |
| Gas-liquid ratio (scf/STB) | 100 - 800 | 1,100 - 1,500 | 115.6 | 42.8 |

(Reproduce with `engines/thp/panel_experiment.py`.)

**What the API does about it:** every request is range-checked. If any input is outside the trained range, `/predict/ml` answers with the physics engine instead and says so (`method: "physics_fallback"`, `confidence: "low"`, plus a plain-language `note`). Add `?fallback=false` to force the raw surrogate output. The physics engine itself is still only as good as its correlations outside the range it was checked against.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Haqqconcept/wellapi.git
cd wellapi
pip install -r requirements.txt

python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API documentation. A SQLite database is created on first run. The first prediction after the server starts loads the model and can take several seconds; later calls take milliseconds.

**Web frontend (Streamlit)** in a second terminal:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The frontend talks to the API over HTTP, so it exercises real login, API keys and credits. Point it at another server with the `WELLAPI_URL` environment variable.

## Using the API

Predictions authenticate with an `X-API-Key` header; account actions use a login token.

```python
import requests

BASE = "http://127.0.0.1:8000"

# 1. Register (50 free credits) and log in. Login is form-encoded.
requests.post(f"{BASE}/auth/register",
              json={"email": "you@example.com", "name": "Your Name", "password": "a-strong-password"})
token = requests.post(f"{BASE}/auth/login",
                      data={"username": "you@example.com", "password": "a-strong-password"}).json()["access_token"]
auth = {"Authorization": f"Bearer {token}"}

# 2. Get an API key (use an existing one from GET /auth/keys, or create one)
keys = requests.get(f"{BASE}/auth/keys", headers=auth).json()
key = (keys[0] if keys else requests.post(f"{BASE}/auth/keys", headers=auth, json={"label": "demo"}).json())["key"]

# 3. Predict
well = {"QL": 330, "WOR": 1.5, "GLR": 273, "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
        "T": 150, "API": 35, "Bw": 1.01, "d": 2.441, "depth": 5000, "Pwf": 1000}
headers = {"X-API-Key": key}

print(requests.post(f"{BASE}/predict/ml", headers=headers, json=well).json())
print(requests.post(f"{BASE}/predict/physics", headers=headers, json=well).json()["THP"])
```

`/predict/ml` returns `THP`, `method`, `confidence`, `out_of_range`, an optional `note`, `credits_remaining` and `response_ms`. `/predict/physics` also returns the full pressure-depth `traverse_table` (disable with `?include_traverse=false`).

### Endpoints

| Endpoint | Auth | Credits | Purpose |
|---|---|---:|---|
| `POST /predict/ml` | API key | 1 | Surrogate prediction (physics fallback if out of range) |
| `POST /predict/physics` | API key | 4 | Full gradient traverse and pressure profile |
| `POST /nodal/analyze` | API key | 8 | Nodal analysis: IPR x VLP operating point |
| `POST /auth/register`, `POST /auth/login` | none | - | Create account, log in |
| `GET/POST/DELETE /auth/keys` | login | - | Manage API keys |
| `GET /account/me`, `/account/usage`, `/account/bundles` | login / none | - | Balance, history, bundles |
| `POST /account/topup` | login | - | Buy credits (Paystack or Stripe) |
| `GET /health` | none | - | Server status |

## Nodal analysis

`POST /nodal/analyze` finds where reservoir inflow (IPR: Vogel, Fetkovich or Darcy) meets the tubing's required outflow (VLP, solved on the physics engine) for a target wellhead pressure.

```python
request = {
    "well": {"WOR": 1.5, "GLR": 273, "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
             "T": 150, "API": 35, "Bw": 1.01, "d": 2.441, "depth": 5000},
    "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 1800},
    "target_thp": 200, "q_min": 100, "q_max": 1500, "n_points": 10,
}
result = requests.post(f"{BASE}/nodal/analyze", headers=headers, json=request).json()
print(result["operating_point"])   # for this example: about 1,400 STB/day at about 1,195 psia
```

The response includes both curves for plotting and a `warnings` list. At moderate rates hydrostatic pressure dominates, so the VLP curve is nearly flat; results are **most reliable up to about 1,500 STB/day**, and the API warns beyond that.

## Credits and pricing

New accounts receive 50 free credits. Bundles (credits for USD): starter 100 for $5, basic 500 for $20, professional 2,000 for $75, enterprise 8,000 for $250, academic 1,200 for $30. Purchases need Paystack or Stripe keys (see below); without them, `/account/topup` returns an error.

## Configuration

Set in the environment or a `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | development key | Signs login tokens. **Must be changed in production.** |
| `DEBUG` | `true` | With `false`, the server refuses to start on the default `SECRET_KEY` |
| `DATABASE_URL` | `sqlite:///./wellapi.db` | Use PostgreSQL in production (SQLite files are lost on most hosts' redeploys) |
| `STORE_REQUEST_INPUTS` | `false` | Save each request's well inputs in the usage log |
| `FREE_CREDITS_ON_SIGNUP` | `50` | Starting credits |
| `ML_CALL_COST`, `PHYSICS_CALL_COST` | `1`, `4` | Credits per call |
| `NODAL_CALL_COST` | `8` | Credits per nodal analysis (optional) |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `PAYSTACK_SECRET_KEY` | empty | Payments |
| `WELLAPI_URL` | `http://127.0.0.1:8000` | Server address used by the Streamlit frontend |

Generate a key with `python -c "import secrets; print(secrets.token_hex(32))"`.

## Privacy and security

- **Request inputs are not stored by default.** Only billing metadata is kept (endpoint, credits, timing, result). Inputs are generic engineering numbers with no well names or coordinates.
- **Self-hosting:** the code is open, so a company can run the whole system on its own infrastructure and never send data out.
- **Startup check:** with `DEBUG=false`, the server will not start using the default `SECRET_KEY`.
- **Not done yet:** rate limiting, email verification at signup (so free credits can be farmed with many accounts), hashing of stored API keys, and restricting CORS (currently open to all origins). Address these before opening the service to the public.

## The physics engine

A gradient traverse along the tubing using the Poettmann-Carpenter correlation, with Hall-Yarborough z-factor and Standing / Dake PVT correlations. Depth step per pressure step:

```
dh = 144.9 * dp / ( rho_avg + K / rho_avg )
K  = 3.3567e-6 * QL^2 * M^2 * Ef / d^5                     (d in inches)
Ef = 4.9884 * Dvp^-3.092368 + 60.37e-6 * ( 1/Dvp + 0.00524355 )
```

**A coefficient correction.** The textbook (T. Ahmed, *Reservoir Engineering Handbook*, 4th ed., 2010) prints the leading coefficient of `Ef` as 46.0115. That value does not reproduce the book's own worked answer (`Ef = 0.006678` at `Dvp = 8.498`); 4.9884 does. We therefore believe the printed value is a misprint (a page reference will be added once confirmed). With the uncorrected term, THP barely responded to flow rate; with the corrected term it does. The valid range of the `Ef` correlation has not been confirmed against the original source.

**Validation status.** Intermediate quantities (`M`, `Dvp`, `Bo`, `Rs`, `fw`, `fo`, `Tpc`, `Ppc`) match Ahmed's Example 7-16. The engine's final THP is about 145 psia against the book's roughly 110 psia, a difference of about 35 psia attributed to the z-factor method (analytical Hall-Yarborough versus manual chart reading). See `engines/thp/validation_note.md`. This is the only validation case so far.

## Reproducing the results

All scripts run from `engines/thp/`:

```bash
python retrain.py --out-dir artifacts   # regenerate data with the physics engine and retrain (about 3 minutes)
python crossplot_data.py                # 48 fresh cases, physics vs surrogate
python benchmark_speed.py               # speed on your machine
python panel_experiment.py              # XGBoost vs neural network, in range and outside it
```

Tests (from the repository root): `python -m tests.test_nodal` and `python -m tests.test_safety`.

## Project structure

```
main.py                 FastAPI app
core/                   config, auth, database, privacy and startup safeguards
routes/                 /auth and /account
engines/thp/            physics engine, surrogate model, /predict routes, training scripts
engines/nodal/          IPR models, VLP solver, operating-point matcher, /nodal route
ml/registry.py          central model loader
models/                 database models and schemas
frontend/               Streamlit web app and API client
tests/                  automated tests
```

## Limitations

- Training data is synthetic and comes from a single correlation. The surrogate cannot be more accurate than that physics.
- Validated against one textbook example only; no field data and no comparison with commercial software yet.
- The friction correlation is trusted only up to about 1,500 STB/day, and its valid range is not yet confirmed.
- PVT correlations are not calibrated to any particular basin's fluids (for example the Niger Delta).
- A steady-state snapshot: it does not forecast changes in gas ratio or water cut over time.
- Vertical wells only.

## Roadmap

Validate against field data and commercial software; calibrate PVT to regional fluid samples; additional flow correlations (for example Hagedorn-Brown); artificial lift (gas lift, ESP); uncertainty estimates and explainability; batch endpoint; evaluate a neural-network or hybrid surrogate for better behaviour outside the trained range.

## References

- Ahmed, T. (2010). *Reservoir Engineering Handbook* (4th ed.). Gulf Professional Publishing.
- Poettmann, F. H. & Carpenter, P. G. (1952). The multiphase flow of gas, oil and water through vertical flow strings. *Drilling and Production Practice*, API.
- Hall, K. R. & Yarborough, L. (1973). A new equation of state for Z-factor calculations. *Oil & Gas Journal*, 71(25), 82-92.
- Standing, M. B. (1947). A pressure-volume-temperature correlation for mixtures of California oils and gases. *Drilling and Production Practice*, API.
- Dake, L. P. (1978). *Fundamentals of Reservoir Engineering*. Elsevier.
- Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD*.

## Author and licence

Abdulhaqq Abdulhakeem, Federal University of Technology, Minna, Nigeria.

Released under the MIT License (see `LICENSE`).
