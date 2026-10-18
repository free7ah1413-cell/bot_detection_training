"""
Step 6: Three-class detection - Human / Bot / Credential Stuffing.

The features that define the credential-stuffing labelling rule (login attempt
counts, login failures, CMS-probe counts) are withheld from the model input, so
the classifier has to recover the attack class from behavioural evidence rather
than by inverting the labelling rule.
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    CLASS_NAMES,
    LABEL_DEFINING_COLS,
    MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    SPLIT_STRATEGIES,
    ensure_dirs,
    feature_columns,
)

TARGET = "traffic_class"


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def build_models(y_train):
    """Class-weighted models; XGBoost receives explicit per-sample weights."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", multi_class="multinomial",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=1, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            objective="multi:softprob", num_class=len(CLASS_NAMES),
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, objective="multiclass", class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
    }


def sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weight_of = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight_of[v] for v in y])


def per_class_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    rows = []
    for idx, name in enumerate(CLASS_NAMES):
        rows.append({
            "class": name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
        })
    return rows


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 6: Three-Class Detection (Human / Bot / Credential Stuffing)")
    print("=" * 70)

    summary_rows = []
    per_class_rows = []
    confusions = {}

    for strategy in SPLIT_STRATEGIES:
        print(f"\n{'=' * 60}\nSPLIT STRATEGY: {strategy.upper()}\n{'=' * 60}")
        train = pd.read_parquet(SPLITS_DIR / f"{strategy}_train.parquet")
        val = pd.read_parquet(SPLITS_DIR / f"{strategy}_val.parquet")
        test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")

        cols = feature_columns(train, exclude_label_defining=True)
        print(f"Features: {len(cols)} (withheld: {', '.join(LABEL_DEFINING_COLS)})")
        for name, part in [("Train", train), ("Val", val), ("Test", test)]:
            dist = part[TARGET].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
            print(f"  {name:<6}: {len(part):,}  " + "  ".join(
                f"{CLASS_NAMES[i]}={int(dist[i]):,}" for i in range(len(CLASS_NAMES))
            ))

        X_train, y_train = clean_matrix(train[cols].values), train[TARGET].values.astype(int)
        X_val, y_val = clean_matrix(val[cols].values), val[TARGET].values.astype(int)
        X_test, y_test = clean_matrix(test[cols].values), test[TARGET].values.astype(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        joblib.dump(scaler, MODELS_DIR / f"mc_scaler_{strategy}.joblib")

        for model_name, model in build_models(y_train).items():
            print(f"\n  Training: {model_name}...")
            t0 = time.time()
            if model_name == "XGBoost":
                model.fit(X_train_s, y_train, sample_weight=sample_weights(y_train))
            else:
                model.fit(X_train_s, y_train)
            elapsed = time.time() - t0

            slug = model_name.lower().replace(" ", "_")
            joblib.dump(model, MODELS_DIR / f"mc_{slug}_{strategy}.joblib")

            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)
            val_f1 = f1_score(y_val, model.predict(X_val_s), average="macro", zero_division=0)

            try:
                ovr_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except ValueError:
                ovr_auc = float("nan")

            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            rows = per_class_metrics(y_test, y_pred, y_prob)
            for r in rows:
                r.update({"model": model_name, "split_strategy": strategy})
            per_class_rows.extend(rows)

            cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_NAMES)))
            confusions[f"{model_name}|{strategy}"] = cm.tolist()

            stuffing = rows[2]
            summary_rows.append({
                "model": model_name,
                "split_strategy": strategy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "macro_roc_auc_ovr": ovr_auc,
                "val_macro_f1": val_f1,
                "stuffing_precision": stuffing["precision"],
                "stuffing_recall": stuffing["recall"],
                "stuffing_f1": stuffing["f1_score"],
                "stuffing_support": stuffing["support"],
                "train_time_sec": round(elapsed, 2),
            })

            print(f"    {elapsed:.1f}s | macro-F1={macro_f1:.4f} weighted-F1={weighted_f1:.4f} "
                  f"OvR-AUC={ovr_auc:.4f}")
            print(f"    Credential Stuffing: P={stuffing['precision']:.4f} "
                  f"R={stuffing['recall']:.4f} F1={stuffing['f1_score']:.4f} "
                  f"(n={stuffing['support']})")

        # Detailed report for the strongest configuration on this split
        print("\n  Classification report (Random Forest):")
        rf = joblib.load(MODELS_DIR / f"mc_random_forest_{strategy}.joblib")
        print(classification_report(
            y_test, rf.predict(X_test_s), target_names=CLASS_NAMES, zero_division=0, digits=4
        ))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "multiclass_results.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "multiclass_per_class.csv", index=False)
    (RESULTS_DIR / "multiclass_confusion.json").write_text(
        json.dumps(confusions, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("THREE-CLASS SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\n[OK] Saved multiclass results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
