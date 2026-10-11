"""Recompute the paper's headline numbers from the saved CSVs.

No GPU, no competition data, no model needed. Run from the repo root:
    python scripts/verify_results.py
Exits non-zero if any number disagrees with the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-4
failures = []


def check(name, got, expected, tol=TOL):
    ok = abs(got - expected) <= tol
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<40} {got:.4f}   (paper {expected:.4f})")
    if not ok:
        failures.append(name)


def rate(df, family, amp, thr):
    s = df[(df.family == family) & np.isclose(df.amplitude, amp)]
    if len(s) != 300:
        failures.append(f"{family}@{amp} n={len(s)}")
    return (s.score > thr).mean()


# 1. Frozen baseline: AUC, 1%-FPR threshold, recall (validation fold, out-of-fold preds)
oof = pd.read_csv(ROOT / "data/oof_df.csv")
y, p = oof.target.values, oof.preds.values
thr = np.percentile(p[y == 0], 99)
check("baseline AUC", roc_auc_score(y, p), 0.8545)
check("threshold (99th pct of val negatives)", thr, 0.1309)
check("baseline recall at threshold", (p[y == 1] > thr).mean(), 0.5525)

# 2. Clean controls (300 uninjected backgrounds)
clean = pd.read_csv(ROOT / "results/clean_control_scores.csv")
check("clean-control FPR", (clean.score > thr).mean(), 4 / 300)

# 3. Injection scores (family/amplitude live in the manifest, not the scores file)
scores = pd.read_csv(ROOT / "results/phase3_scores.csv")
man = pd.read_csv(ROOT / "data/injection_manifest.csv")
if len(scores) != 15000 or scores.sample_id.nunique() != 15000:
    failures.append("phase3_scores row count / duplicates")
df = scores.merge(man[["sample_id", "family", "amplitude"]], on="sample_id")
check("D1-1_extreme_drift @0.5 recall", rate(df, "D1-1_extreme_drift", 0.5, thr), 0.4100)
check("D1-3_s_shaped @4.0 recall", rate(df, "D1-3_s_shaped", 4.0, thr), 0.9000)
check("D2-5_high_drift @0.5 FPR", rate(df, "D2-5_high_drift", 0.5, thr), 0.6267)

print()
if failures:
    print("MISMATCH:", ", ".join(failures))
    sys.exit(1)
print("All headline numbers reproduce.")
