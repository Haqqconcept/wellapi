"""Run from the repo root:  python -m tests.test_nodal
Stubs auth and the database, so no login or credits are needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from engines.nodal import router as R


class FakeUser:
    id = "u1"; credits = 100; total_calls = 0


class FakeDB:
    def add(self, x): pass
    def commit(self): pass


app = FastAPI(); app.include_router(R.router)
app.dependency_overrides[R._charge] = lambda: FakeUser()
app.dependency_overrides[R.get_db] = lambda: FakeDB()
c = TestClient(app)

WELL = {"WOR": 1.5, "GLR": 273, "oil_sg": 0.85, "water_sg": 1.05, "gas_sg": 0.75,
        "T": 150, "API": 35, "Bw": 1.01, "d": 2.441, "depth": 5000}


def post(body):
    return c.post("/nodal/analyze", json=body)


def test_vogel_finds_operating_point():
    j = post({"well": WELL, "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 1800}, "target_thp": 200}).json()
    assert j["operating_point"]["found"] and 1000 < j["operating_point"]["Q"] < 1500


def test_missing_vogel_param_is_422():
    assert post({"well": WELL, "ipr": {"model": "vogel", "Pr": 3000}, "target_thp": 200}).status_code == 422


def test_bad_rate_range_is_422():
    r = post({"well": WELL, "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 1800},
              "target_thp": 200, "q_min": 500, "q_max": 400})
    assert r.status_code == 422


def test_high_rate_warns():
    j = post({"well": WELL, "ipr": {"model": "vogel", "Pr": 3000, "Qmax": 4000},
              "target_thp": 200, "q_max": 3000}).json()
    assert any("validated rate range" in w for w in j["warnings"])


def test_weak_reservoir_reports_no_flow():
    j = post({"well": WELL, "ipr": {"model": "vogel", "Pr": 1000, "Qmax": 500}, "target_thp": 200}).json()
    assert not j["operating_point"]["found"] and "cannot flow" in j["operating_point"]["note"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
