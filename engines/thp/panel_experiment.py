import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from retrain import sample, label, FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

t=time.time()
df = label(sample(12000, 42))
print(f"dataset: {len(df)} cases ({time.time()-t:.0f}s)")
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES].values, df["THP"].values, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=5, random_state=42, verbosity=0).fit(Xtr, ytr)
mlp = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,128,64), early_stopping=True,
                            max_iter=600, random_state=42)),
    transformer=StandardScaler()).fit(Xtr, ytr)

def rep(name, y, px, pm):
    print(f"{name:34s} XGBoost MAE {mean_absolute_error(y,px):8.1f}   DNN(MLP) MAE {mean_absolute_error(y,pm):8.1f}   n={len(y)}")

print("\n--- IN-RANGE TEST SPLIT")
px, pm = xgb.predict(Xte), mlp.predict(Xte)
print(f"XGBoost  MAE {mean_absolute_error(yte,px):.2f}  R2 {r2_score(yte,px):.4f}")
print(f"DNN(MLP) MAE {mean_absolute_error(yte,pm):.2f}  R2 {r2_score(yte,pm):.4f}")

print("\n--- EXTRAPOLATION (physics engine = reference; training range shown in brackets)")
tests = {"QL": [(1600,2200),(2200,3000)], "depth": [(8500,10000)], "Pwf": [(2100,2600),(2600,3200)], "GLR": [(850,1100),(1100,1500)]}
tr_rng = {"QL":"100-1500","depth":"2000-8000","Pwf":"200-2000","GLR":"100-800"}
for var, bands in tests.items():
    for lo, hi in bands:
        rows = sample(220, 900 + int(lo))
        rng = np.random.default_rng(int(lo))
        for r in rows: r[var] = float(rng.uniform(lo, hi))
        d = label(rows)
        if len(d) < 20: print(f"{var} {lo}-{hi}: too few valid cases"); continue
        X = d[FEATURES].values
        rep(f"{var} {lo}-{hi} [{tr_rng[var]}]", d["THP"].values, xgb.predict(X), mlp.predict(X))
