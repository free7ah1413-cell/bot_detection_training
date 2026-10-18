"""
Step 7: Feature-set ablation.

Trains the supervised learners on four feature sets under every splitting
strategy:

  Request-level   -- descriptors of individual requests, averaged per session
  Session-level   -- behavioural aggregates of the session as a whole
  Combined        -- the feature set used for every reported result
  Combined + UA   -- adds the User-Agent descriptors. Because the bot label is
                     itself derived from the User-Agent string, this variant is
                     circular; it is included only to quantify how much a
                     spurious label-correlated feature inflates the metrics.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    REQUEST_LEVEL_FEATURES,
    RESULTS_DIR,
    RANDOM_SEED,
    SESSION_LEVEL_FEATURES,
    SPLITS_DIR,
    SPLIT_STRATEGIES,
    UA_DERIVED_COLS,
    ensure_dirs,
    feature_columns,
)


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def build_models(y_train):
    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=1, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, scale_pos_weight=n_neg / n_pos,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
    }


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 7: Feature-Set Ablation")
    print("=" * 70)

    rows = []
    for strategy in SPLIT_STRATEGIES:
        train = pd.read_parquet(SPLITS_DIR / f"{strategy}_train.parquet")
        test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
        available = set(feature_columns(train))
        with_ua = feature_columns(train, include_ua_derived=True)

        feature_sets = {
            "Request-level": [c for c in REQUEST_LEVEL_FEATURES if c in available],
            "Session-level": [c for c in SESSION_LEVEL_FEATURES if c in available],
            "Combined": sorted(available),
            "Combined + UA (circular)": sorted(with_ua),
        }

        y_train = train["is_bot"].values.astype(int)
        y_test = test["is_bot"].values.astype(int)

        print(f"\n{'=' * 60}\nSPLIT: {strategy.upper()}\n{'=' * 60}")
        for set_name, cols in feature_sets.items():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(clean_matrix(train[cols].values))
            X_test = scaler.transform(clean_matrix(test[cols].values))

            print(f"\n  Feature set: {set_name} ({len(cols)} features)")
            for model_name, model in build_models(y_train).items():
                t0 = time.time()
                model.fit(X_train, y_train)
                elapsed = time.time() - t0
                metrics = evaluate(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
                metrics.update({
                    "split_strategy": strategy,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "model": model_name,
                    "train_time_sec": round(elapsed, 2),
                })
                rows.append(metrics)
                print(f"    {model_name:<20} F1={metrics['f1_score']:.4f} "
                      f"PR-AUC={metrics['pr_auc']:.4f} FPR={metrics['fpr']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    pivot = df.pivot_table(values="f1_score", index=["model", "feature_set"], columns="split_strategy")
    pivot.to_csv(RESULTS_DIR / "ablation_pivot_f1.csv")

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (F1-score)")
    print("=" * 70)
    print(pivot.round(4).to_string())

    print("\nCIRCULARITY INFLATION (Combined + UA minus Combined):")
    circ_rows = []
    for (model_name, strategy), grp in df.groupby(["model", "split_strategy"]):
        base = grp[grp["feature_set"] == "Combined"]["f1_score"]
        circ = grp[grp["feature_set"] == "Combined + UA (circular)"]["f1_score"]
        if base.empty or circ.empty:
            continue
        delta = float(circ.iloc[0] - base.iloc[0])
        circ_rows.append({
            "model": model_name, "split_strategy": strategy,
            "f1_combined": float(base.iloc[0]),
            "f1_with_ua": float(circ.iloc[0]),
            "inflation": delta,
        })
        print(f"  {model_name:<20} {strategy:<9} {base.iloc[0]:.4f} -> {circ.iloc[0]:.4f} "
              f"({delta:+.4f})")
    pd.DataFrame(circ_rows).to_csv(RESULTS_DIR / "circularity_inflation.csv", index=False)

    print(f"\n[OK] Saved ablation results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
