"""Run from the project root:  python -m tests.test_safety
Covers: out-of-range fallback, privacy switch for stored inputs, SECRET_KEY safeguard."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from engines.thp import router as T
from engines.nodal import router as N
from core import privacy


class FakeUser:
    id = "u1"; credits = 100; total_calls = 0


class FakeDB:
    def __init__(self): self.logs = []
    def add(self, x): self.logs.append(x)
    def commit(self): pass


DB = FakeDB()
app = FastAPI()
app.include_router(T.router); app.include_router(N.router)
for dep in (T._charge_ml, T._charge_physics, N._charge):
    app.dependency_overrides[dep] = lambda: FakeUser()
app.dependency_overrides[T.get_db] = lambda: DB
c = TestClient(app)

WELL = dict(QL=330, WOR=1.5, GLR=273, oil_sg=0.85, water_sg=1.05, gas_sg=0.75, T=150, API=35, Bw=1.01,
            d=2.441, depth=5000, Pwf=1000)


def test_in_range_uses_surrogate():
    j = c.post("/predict/ml", json=WELL).json()
    assert j["method"] == "ml_surrogate" and j["note"] is None and j["confidence"] == "high"


def test_out_of_range_falls_back_to_physics():
    far = {**WELL, "Pwf": 2800}
    j = c.post("/predict/ml", json=far).json()
    phys = c.post("/predict/physics?include_traverse=false", json=far).json()
    assert j["method"] == "physics_fallback" and j["confidence"] == "low"
    assert j["out_of_range"] == ["Pwf"] and "physics engine" in j["note"]
    assert abs(j["THP"] - phys["THP"]) < 0.11            # same number the physics endpoint gives
    assert j["credits_used"] == 1                         # still charged the ML price


def test_fallback_can_be_disabled():
    j = c.post("/predict/ml?fallback=false", json={**WELL, "Pwf": 2800}).json()
    assert j["method"] == "ml_surrogate" and j["out_of_range"] == ["Pwf"]


def test_fallback_failure_is_422_and_not_charged():
    original = T._run_physics
    T._run_physics = lambda inputs: (_ for _ in ()).throw(ValueError("boom"))
    try:
        n_before = len(DB.logs)
        r = c.post("/predict/ml", json={**WELL, "Pwf": 2800})
        assert r.status_code == 422 and "could not solve" in r.json()["detail"]
        assert len(DB.logs) == n_before                   # nothing logged, nothing charged
    finally:
        T._run_physics = original


def test_inputs_not_stored_by_default():
    privacy.privacy.STORE_REQUEST_INPUTS = False
    c.post("/predict/ml", json=WELL)
    assert DB.logs[-1].input_params is None
    assert DB.logs[-1].thp_result is not None             # billing metadata is still kept


def test_inputs_stored_when_enabled():
    privacy.privacy.STORE_REQUEST_INPUTS = True
    try:
        c.post("/predict/ml", json=WELL)
        assert '"Pwf": 1000' in DB.logs[-1].input_params
    finally:
        privacy.privacy.STORE_REQUEST_INPUTS = False


def test_nodal_also_respects_privacy():
    body = {"well": {k: WELL[k] for k in ("WOR","GLR","oil_sg","water_sg","gas_sg","T","API","Bw","d","depth")},
            "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 1800}, "target_thp": 200}
    assert c.post("/nodal/analyze", json=body).status_code == 200
    assert DB.logs[-1].endpoint == "nodal" and DB.logs[-1].input_params is None


def test_default_secret_blocked_in_production_allowed_in_dev():
    try:
        privacy.check_production_settings(secret_key=privacy.DEFAULT_SECRET, debug=False)
        assert False, "should have refused to start"
    except RuntimeError as e:
        assert "SECRET_KEY" in str(e)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        privacy.check_production_settings(secret_key=privacy.DEFAULT_SECRET, debug=True)
        assert any("SECRET_KEY" in str(x.message) for x in w)
    privacy.check_production_settings(secret_key="a" * 64, debug=False)   # real key: silent


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
