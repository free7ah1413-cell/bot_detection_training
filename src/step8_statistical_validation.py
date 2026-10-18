"""
Step 8: Statistical and robustness validation.

  * Bootstrap 95% confidence intervals for F1 / PR-AUC on each test partition.
  * McNemar's test between model pairs evaluated on the same test partition.
  * Seed-stability check: the tree ensembles are refitted under several random
    seeds to quantify variance attributable to initialisation.
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    MODELS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    SPLIT_STRATEGIES,
    ensure_dirs,
    feature_columns,
)

N_BOOTSTRAP = 1000
SEEDS = [42, 7, 2024, 31337, 101]
MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Isolation Forest": "isolation_forest",
}


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def model_scores(model, X, model_name):
    if model_name == "Isolation Forest":
        y_pred = (model.predict(X) == -1).astype(int)
        s = -model.decision_function(X)
        lo, hi = s.min(), s.max()
        y_prob = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1s, prs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1s.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        prs.append(average_precision_score(y_true[idx], y_prob[idx]))
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_ci_low": float(np.percentile(f1s, 2.5)),
        "f1_ci_high": float(np.percentile(f1s, 97.5)),
        "pr_auc_mean": float(np.mean(prs)),
        "pr_auc_ci_low": float(np.percentile(prs, 2.5)),
        "pr_auc_ci_high": float(np.percentile(prs, 97.5)),
    }


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on the discordant pairs of two classifiers."""
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    n01 = int(np.sum(a_right & ~b_right))
    n10 = int(np.sum(~a_right & b_right))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p_value": float(p)}


def seed_stability(strategy):
    """Refit tree ensembles under several seeds and report F1 spread."""
    train = pd.read_parquet(SPLITS_DIR / f"{strategy}_train.parquet")
    test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
    cols = feature_columns(train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(clean_matrix(train[cols].values))
    X_test = scaler.transform(clean_matrix(test[cols].values))
    y_train = train["is_bot"].values.astype(int)
    y_test = test["is_bot"].values.astype(int)
    n_pos = max(int(y_train.sum()), 1)
    scale_pos = (len(y_train) - n_pos) / n_pos

    out = {}
    for name in ["Random Forest", "XGBoost", "LightGBM"]:
        scores = []
        for seed in SEEDS:
            if name == "Random Forest":
                m = RandomForestClassifier(
                    n_estimators=300, max_depth=20, min_samples_leaf=1, min_samples_split=5,
                    class_weight="balanced", random_state=seed, n_jobs=-1,
                )
            elif name == "LightGBM":
                m = lgb.LGBMClassifier(
                    n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8,
                    subsample_freq=1, scale_pos_weight=scale_pos,
                    random_state=seed, n_jobs=-1, verbose=-1,
                )
            else:
                m = xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                    scale_pos_weight=scale_pos, eval_metric="logloss",
                    tree_method="hist", random_state=seed, n_jobs=-1,
                )
            m.fit(X_train, y_train)
            scores.append(f1_score(y_test, m.predict(X_test), zero_division=0))
        out[name] = {
            "seeds": SEEDS,
            "f1_scores": [round(float(s), 5) for s in scores],
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        print(f"    {name}: F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return out


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 8: Statistical and Robustness Validation")
    print("=" * 70)

    ci_rows = []
    mcnemar_rows = []
    stability = {}

    for strategy in SPLIT_STRATEGIES:
        print(f"\n{'=' * 60}\nSPLIT: {strategy.upper()}\n{'=' * 60}")
        test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
        cols = feature_columns(test)
        scaler = joblib.load(MODELS_DIR / f"scaler_{strategy}.joblib")
        X_test = scaler.transform(clean_matrix(test[cols].values))
        y_test = test["is_bot"].values.astype(int)

        preds = {}
        print("\n  Bootstrap 95% confidence intervals (1000 resamples):")
        for model_name, slug in MODEL_SLUGS.items():
            path = MODELS_DIR / f"{slug}_{strategy}.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)
            y_pred, y_prob = model_scores(model, X_test, model_name)
            preds[model_name] = y_pred
            ci = bootstrap_ci(y_test, y_pred, y_prob)
            ci.update({"model": model_name, "split_strategy": strategy})
            ci_rows.append(ci)
            print(f"    {model_name:<20} F1={ci['f1_mean']:.4f} "
                  f"[{ci['f1_ci_low']:.4f}, {ci['f1_ci_high']:.4f}]")

        print("\n  McNemar tests (pairwise, same test partition):")
        for a, b in combinations(preds, 2):
            res = mcnemar(y_test, preds[a], preds[b])
            res.update({"model_a": a, "model_b": b, "split_strategy": strategy,
                        "significant_at_0.05": bool(res["p_value"] < 0.05)})
            mcnemar_rows.append(res)
            flag = "significant" if res["p_value"] < 0.05 else "not significant"
            print(f"    {a} vs {b}: p={res['p_value']:.3e} ({flag})")

        print("\n  Seed stability (5 seeds):")
        stability[strategy] = seed_stability(strategy)

    pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)
    (RESULTS_DIR / "seed_stability.json").write_text(
        json.dumps(stability, indent=2), encoding="utf-8"
    )

    # Do the random-split and entity-split confidence intervals overlap?
    ci = pd.DataFrame(ci_rows)
    print("\n" + "=" * 70)
    print("LEAKAGE SIGNIFICANCE (non-overlapping 95% CIs => reliable difference)")
    print("=" * 70)
    overlap_rows = []
    for model_name in ci["model"].unique():
        sub = ci[ci["model"] == model_name].set_index("split_strategy")
        if not {"random", "temporal", "entity", "ua_entity"}.issubset(sub.index):
            continue
        r = sub.loc["random"]
        for other in ["temporal", "entity", "ua_entity"]:
            o = sub.loc[other]
            overlaps = not (r["f1_ci_low"] > o["f1_ci_high"] or o["f1_ci_low"] > r["f1_ci_high"])
            overlap_rows.append({
                "model": model_name, "comparison": f"random vs {other}",
                "random_f1": r["f1_mean"], "other_f1": o["f1_mean"],
                "delta": r["f1_mean"] - o["f1_mean"], "ci_overlap": overlaps,
            })
            print(f"  {model_name:<20} random vs {other:<9} "
                  f"delta={r['f1_mean'] - o['f1_mean']:+.4f}  "
                  f"CI overlap={'yes' if overlaps else 'NO'}")
    pd.DataFrame(overlap_rows).to_csv(RESULTS_DIR / "leakage_significance.csv", index=False)

    print(f"\n[OK] Saved statistical validation to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
