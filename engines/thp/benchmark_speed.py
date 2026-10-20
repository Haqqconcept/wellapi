"""
benchmark_speed.py - measures the speed numbers shown on the poster, on YOUR machine.

Run from the engines\\thp folder:
    python benchmark_speed.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import joblib, pandas as pd
from retrain import sample, run_physics, FEATURES

model = joblib.load(os.path.join("artifacts", "thp_model.joblib"))
rows = sample(10000, seed=5)
X = pd.DataFrame(rows)[FEATURES].values

n = 300
t = time.perf_counter(); [run_physics(r) for r in rows[:n]]; physics = (time.perf_counter() - t) / n
t = time.perf_counter(); [model.predict(X[i:i + 1]) for i in range(n)]; single = (time.perf_counter() - t) / n
t = time.perf_counter(); model.predict(X); batch = time.perf_counter() - t

print(f"\nOne well:       physics {physics*1000:.1f} ms   surrogate {single*1000:.2f} ms   ({physics/single:.0f}x)")
print(f"10,000 wells:   physics ~{physics*10000:.0f} s   surrogate {batch*1000:.0f} ms   ({physics*10000/batch:.0f}x)")
print("\nSend these two lines to Claude so the poster can be updated to your numbers.")
