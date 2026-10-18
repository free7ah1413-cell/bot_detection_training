"""
Step 4: Train 4 models x 3 split strategies and evaluate the binary bot task.

Models: Logistic Regression, Random Forest, XGBoost, Isolation Forest.
Hyperparameters are selected on the validation partition only; the test
partition is used exactly once, for the final reported metrics.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SPLIT_STRATEGIES,
    RANDOM_SEED,
    ensure_dirs,
    feature_columns,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob)) if y_prob is not None else 0.0
    except ValueError:
        metrics["pr_auc"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Hyperparameter grids. Each candidate is fitted on the training partition and
# scored on the validation partition (F1); the winner is refitted and evaluated
# once on the held-out test partition.
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "Logistic Regression": [
        {"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0},
    ],
    "Random Forest": [
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
    ],
    "XGBoost": [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 400, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.9},
    ],
    "Isolation Forest": [
        {"n_estimators": 200, "max_samples": 256},
        {"n_estimators": 300, "max_samples": 512},
        {"n_estimators": 300, "max_samples": "auto"},
    ],
}


def build_logistic_regression(params, y_train):
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
        solver="lbfgs", n_jobs=-1, **params,
    )


def build_random_forest(params, y_train):
    return RandomForestClassifier(
        min_samples_split=5, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1, **params,
    )


def build_xgboost(params, y_train):
    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    return xgb.XGBClassifier(
        scale_pos_weight=n_neg / n_pos, random_state=RANDOM_SEED,
        eval_metric="logloss", n_jobs=-1, tree_method="hist", **params,
    )


def build_isolation_forest(params, y_train):
    contamination = float(np.clip(y_train.mean(), 0.01, 0.5))
    return IsolationForest(
        contamination=contamination, random_state=RANDOM_SEED, n_jobs=-1, **params,
    )


MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "Random Forest": build_random_forest,
    "XGBoost": build_xgboost,
    "Isolation Forest": build_isolation_forest,
}


def fit_model(model, model_name, X_train, y_train):
    if model_name == "Isolation Forest":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    return model


def tune_model(model_name, X_train, y_train, X_val, y_val):
    """Grid search scored on the validation partition. Returns (model, params, log)."""
    builder = MODEL_BUILDERS[model_name]
    best_score, best_params, best_model = -1.0, None, None
    log = []
    for params in PARAM_GRIDS[model_name]:
        model = fit_model(builder(params, y_train), model_name, X_train, y_train)
        y_pred, _ = predict_with_model(model, X_val, model_name)
        score = f1_score(y_val, y_pred, zero_division=0)
        log.append({"params": params, "val_f1": round(float(score), 5)})
        if score > best_score:
            best_score, best_params, best_model = score, params, model
    return best_model, best_params, best_score, log


def predict_with_model(model, X, model_name):
    if model_name == "Isolation Forest":
        raw = model.predict(X)
        y_pred = (raw == -1).astype(int)
        scores = -model.decision_function(X)
        smin, smax = scores.min(), scores.max()
        y_prob = (scores - smin) / (smax - smin) if smax > smin else np.full_like(scores, 0.5, dtype=float)
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 4: Model Training & Evaluation")
    print("=" * 70)

    all_results = []
    tuning_log = {}

    for strategy in SPLIT_STRATEGIES:
        print(f"\n{'=' * 60}\nSPLIT STRATEGY: {strategy.upper()}\n{'=' * 60}")
        train = pd.read_parquet(SPLITS_DIR / f"{strategy}_train.parquet")
        val = pd.read_parquet(SPLITS_DIR / f"{strategy}_val.parquet")
        test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
        cols = feature_columns(train)
        print(f"Features: {len(cols)} | Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

        X_train = clean_matrix(train[cols].values)
        y_train = train["is_bot"].values.astype(int)
        X_val = clean_matrix(val[cols].values)
        y_val = val["is_bot"].values.astype(int)
        X_test = clean_matrix(test[cols].values)
        y_test = test["is_bot"].values.astype(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        joblib.dump(scaler, MODELS_DIR / f"scaler_{strategy}.joblib")

        for model_name in MODEL_BUILDERS:
            print(f"\n  Tuning + training: {model_name}...")
            t0 = time.time()
            try:
                model, best_params, best_val_f1, log = tune_model(
                    model_name, X_train_s, y_train, X_val_s, y_val
                )
                train_time = time.time() - t0
                tuning_log[f"{model_name}|{strategy}"] = {
                    "best_params": best_params,
                    "best_val_f1": round(float(best_val_f1), 5),
                    "candidates": log,
                }
                print(f"    best params: {best_params} (val F1={best_val_f1:.4f})")
                slug = model_name.lower().replace(" ", "_")
                joblib.dump(model, MODELS_DIR / f"{slug}_{strategy}.joblib")

                y_pred, y_prob = predict_with_model(model, X_test_s, model_name)
                metrics = compute_metrics(y_test, y_pred, y_prob)
                metrics["model"] = model_name
                metrics["split_strategy"] = strategy
                metrics["train_time_sec"] = round(train_time, 2)

                y_tr_pred, y_tr_prob = predict_with_model(model, X_train_s, model_name)
                tr = compute_metrics(y_train, y_tr_pred, y_tr_prob)
                metrics["train_f1"] = tr["f1_score"]
                metrics["train_roc_auc"] = tr["roc_auc"]
                all_results.append(metrics)

                print(
                    f"    {train_time:.1f}s | P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                    f"F1={metrics['f1_score']:.4f} ROC={metrics['roc_auc']:.4f} PR={metrics['pr_auc']:.4f}"
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"model": model_name, "split_strategy": strategy, "error": str(e)})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    (RESULTS_DIR / "hyperparameter_search.json").write_text(
        json.dumps(tuning_log, indent=2), encoding="utf-8"
    )
    print("\n" + results_df.to_string(index=False))

    for metric_name in ["f1_score", "precision", "recall", "roc_auc", "pr_auc", "fpr"]:
        if metric_name in results_df.columns:
            results_df.pivot_table(values=metric_name, index="model", columns="split_strategy").to_csv(
                RESULTS_DIR / f"pivot_{metric_name}.csv"
            )

    print("\nLEAKAGE IMPACT (Random F1 - other):")
    for model_name in results_df["model"].dropna().unique():
        m = results_df[results_df["model"] == model_name]
        def _f1(s):
            v = m[m["split_strategy"] == s]["f1_score"].values
            return float(v[0]) if len(v) else float("nan")
        r, t, e = _f1("random"), _f1("temporal"), _f1("entity")
        print(f"  {model_name}: random={r:.4f} temporal={t:.4f} (d{r-t:+.4f}) entity={e:.4f} (d{r-e:+.4f})")

    for strategy in SPLIT_STRATEGIES:
        train = pd.read_parquet(SPLITS_DIR / f"{strategy}_train.parquet")
        cols = feature_columns(train)
        for model_name in ["Random Forest", "XGBoost"]:
            path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_{strategy}.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)
            if hasattr(model, "feature_importances_"):
                pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values(
                    "importance", ascending=False
                ).to_csv(
                    RESULTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}_{strategy}.csv",
                    index=False,
                )

    print(f"\n[OK] Results: {RESULTS_DIR}")
    print(f"[OK] Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
