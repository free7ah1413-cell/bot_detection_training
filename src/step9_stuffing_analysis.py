"""
Step 9: Credential-stuffing uncertainty and rule-adjacent ablation (IP-entity split).

Produces
  results/stuffing_bootstrap_ci.csv   bootstrap 95% CI (1000 resamples) of the stuffing-class F1
  results/stuffing_ablation.csv       stuffing-class F1 with and without the behavioural
                                      features adjacent to the stuffing rule
The five label-defining counters stay withheld in both settings.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, RANDOM_SEED, SPLITS_DIR, ensure_dirs, feature_columns
from step6_multiclass import TARGET, build_models, clean_matrix, sample_weights

SPLIT = "entity"
STUFF = 2
RULE_ADJACENT = ["requests_per_minute", "auth_area_count", "auth_area_ratio",
                 "n_4xx", "error_rate_4xx"]


def stuff_f1(y, p):
    return f1_score(y == STUFF, p == STUFF, zero_division=0)


def fit_predict(train, test, cols):
    Xtr = clean_matrix(train[cols].values); ytr = train[TARGET].values.astype(int)
    Xte = clean_matrix(test[cols].values)
    sc = StandardScaler().fit(Xtr)
    out = {}
    for name, model in build_models(ytr).items():
        if name == "XGBoost":
            model.fit(sc.transform(Xtr), ytr, sample_weight=sample_weights(ytr))
        else:
            model.fit(sc.transform(Xtr), ytr)
        out[name] = model.predict(sc.transform(Xte))
    return out


def main():
    ensure_dirs()
    train = pd.read_parquet(SPLITS_DIR / f"{SPLIT}_train.parquet")
    test = pd.read_parquet(SPLITS_DIR / f"{SPLIT}_test.parquet")
    y = test[TARGET].values.astype(int)
    full_cols = feature_columns(train, exclude_label_defining=True)
    red_cols = [c for c in full_cols if c not in RULE_ADJACENT]
    print(f"Full features: {len(full_cols)} | without rule-adjacent: {len(red_cols)}")

    full = fit_predict(train, test, full_cols)
    red = fit_predict(train, test, red_cols)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    ci_rows, ab_rows = [], []
    for m in full:
        boots = np.array([stuff_f1(y[i], full[m][i]) for i in idx])
        f_full, f_red = stuff_f1(y, full[m]), stuff_f1(y, red[m])
        ci_rows.append({"model": m, "f1": f_full, "ci_low": np.percentile(boots, 2.5),
                        "ci_high": np.percentile(boots, 97.5), "n_stuffing": int((y == STUFF).sum())})
        ab_rows.append({"model": m, "full_f1": f_full, "without_rule_adjacent_f1": f_red,
                        "delta": f_red - f_full})
        print(f"{m:<20} full={f_full:.3f}  CI=[{ci_rows[-1]['ci_low']:.3f},{ci_rows[-1]['ci_high']:.3f}]"
              f"  without={f_red:.3f}  delta={f_red - f_full:+.3f}")
    pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "stuffing_bootstrap_ci.csv", index=False)
    pd.DataFrame(ab_rows).to_csv(RESULTS_DIR / "stuffing_ablation.csv", index=False)


if __name__ == "__main__":
    main()
