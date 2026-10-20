"""
crossplot_data.py
Samples fresh well cases the model has never seen, runs the corrected physics
engine and YOUR trained model on each, and prints the pairs for the poster.

Run from the engines\\thp folder:
    python crossplot_data.py
    python crossplot_data.py --model artifacts\\thp_model.joblib
"""
import argparse, json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import joblib, numpy as np
from retrain import sample, label, FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join("artifacts", "thp_model.joblib"))
ap.add_argument("--n", type=int, default=48)
args = ap.parse_args()

model = joblib.load(args.model)
df = label(sample(args.n + 20, seed=777)).head(args.n)      # seed 777: never used in training
pred = model.predict(df[FEATURES].values)
err = np.abs(pred - df["THP"].values)

pairs = [[round(float(a), 1), round(float(p), 1)] for a, p in zip(df["THP"], pred)]
print(f"\nCases: {len(pairs)}   MAE on these cases: {err.mean():.2f} psia   max error: {err.max():.1f} psia")
print("\nCOPY EVERYTHING BELOW THIS LINE AND SEND IT TO CLAUDE:\n")
print(json.dumps(pairs))
