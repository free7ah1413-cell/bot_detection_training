"""
Empirical invariant checks on the artefacts the pipeline actually produced.

Reading the code proves what it intends to do; this proves what it did.
Run after the pipeline completes:  python ml_project/validate_pipeline.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

SPLITS = HERE / "output" / "splits"
failures: list[str] = []
checks = 0


def check(label, condition, detail=""):
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        failures.append(f"{label}: {detail}")


def load(strategy):
    return {part: pd.read_parquet(SPLITS / f"{strategy}_{part}.parquet")
            for part in ("train", "val", "test")}


def main():
    print("=" * 72)
    print("PIPELINE INVARIANT VALIDATION")
    print("=" * 72)

    sessions = pd.read_parquet(HERE / "output" / "session_features.parquet")

    print("\n[1] Session table integrity")
    check("session_id is unique", sessions["session_id"].is_unique,
          f"{sessions['session_id'].duplicated().sum()} duplicates")
    check("no session is empty", (sessions["n_requests"] > 0).all())
    check("session duration is non-negative", (sessions["session_duration_sec"] >= 0).all())
    check("requests_per_minute is finite",
          np.isfinite(sessions["requests_per_minute"]).all())

    feats = config.feature_columns(sessions)
    check("no UA-derived feature reaches the binary models",
          not set(feats) & set(config.UA_DERIVED_COLS),
          f"leaked: {sorted(set(feats) & set(config.UA_DERIVED_COLS))}")
    check("no identifier or label reaches the models",
          not set(feats) & set(config.EXCLUDE_COLS),
          f"leaked: {sorted(set(feats) & set(config.EXCLUDE_COLS))}")

    mc_feats = config.feature_columns(sessions, exclude_label_defining=True)
    check("no label-defining counter reaches the three-class model",
          not set(mc_feats) & set(config.LABEL_DEFINING_COLS),
          f"leaked: {sorted(set(mc_feats) & set(config.LABEL_DEFINING_COLS))}")

    X = sessions[feats]
    bad = [c for c in feats if not np.isfinite(X[c].to_numpy(dtype=float)).all()]
    check("every model feature is finite (no NaN or inf)", not bad, f"offending: {bad}")

    print("\n[2] Labels are consistent between the binary and three-class targets")
    tc, ib = sessions["traffic_class"], sessions["is_bot"]
    check("every credential-stuffing session is also flagged as automated",
          bool(((tc == config.CLASS_STUFFING) <= (tc != config.CLASS_HUMAN)).all()))
    check("traffic_class only takes the three defined values",
          set(tc.unique()) <= {config.CLASS_HUMAN, config.CLASS_BOT, config.CLASS_STUFFING},
          f"found {sorted(tc.unique())}")
    check("is_bot is binary", set(ib.unique()) <= {0, 1})

    print("\n[3] Partitions are exhaustive, disjoint and correctly sized")
    n_total = len(sessions)
    for strategy in config.SPLIT_STRATEGIES:
        parts = load(strategy)
        sizes = {k: len(v) for k, v in parts.items()}
        check(f"{strategy}: partitions cover every session",
              sum(sizes.values()) == n_total, f"{sum(sizes.values())} vs {n_total}")

        uids = {k: set(v["session_id"]) for k, v in parts.items()}
        check(f"{strategy}: no session appears in two partitions",
              not (uids["train"] & uids["test"]) and not (uids["train"] & uids["val"])
              and not (uids["val"] & uids["test"]))

        ratio = sizes["test"] / n_total
        check(f"{strategy}: test partition is close to {config.TEST_RATIO:.0%}",
              abs(ratio - config.TEST_RATIO) < 0.05, f"got {ratio:.1%}")

        for part, df in parts.items():
            check(f"{strategy}/{part}: contains both classes", df["is_bot"].nunique() == 2,
                  f"classes {sorted(df['is_bot'].unique())}")

    print("\n[4] Temporal split is strictly chronological")
    parts = load("temporal")
    tr, va, te = (parts[p]["session_start"] for p in ("train", "val", "test"))
    check("every training session starts before every validation session",
          tr.max() <= va.min(), f"train max {tr.max()} > val min {va.min()}")
    check("every validation session starts before every test session",
          va.max() <= te.min(), f"val max {va.max()} > test min {te.min()}")
    check("no test session precedes the training window",
          te.min() >= tr.max())

    print("\n[5] Entity split keeps every IP address in a single partition")
    parts = load("entity")
    ips = {p: set(df["ip"]) for p, df in parts.items()}
    check("train and test share no IP address", not (ips["train"] & ips["test"]),
          f"{len(ips['train'] & ips['test'])} shared")
    check("train and validation share no IP address", not (ips["train"] & ips["val"]),
          f"{len(ips['train'] & ips['val'])} shared")
    check("validation and test share no IP address", not (ips["val"] & ips["test"]),
          f"{len(ips['val'] & ips['test'])} shared")

    print("\n[6] Random split preserves the class distribution")
    parts = load("random")
    base = sessions["traffic_class"].value_counts(normalize=True)
    for part, df in parts.items():
        got = df["traffic_class"].value_counts(normalize=True)
        drift = max(abs(got.get(c, 0) - base.get(c, 0)) for c in base.index)
        check(f"random/{part}: class proportions within 1 point of the whole corpus",
              drift < 0.01, f"max drift {drift:.4f}")

    print("\n" + "=" * 72)
    print(f"{checks} checks run | {len(failures)} failed")
    print("=" * 72)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")

    status = {
        "passed": not failures,
        "checks": checks,
        "failed": len(failures),
        "failures": failures,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (HERE / "output" / "validation_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

