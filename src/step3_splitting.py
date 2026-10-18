"""
Step 3: Four splitting strategies - random, temporal, IP-entity, User-Agent-entity.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    SESSION_FEATURES,
    SPLITS_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    ensure_dirs,
    feature_columns,
)


def random_split(df):
    print("\n--- Strategy 1: Random Split ---")
    # Stratify on the three-class target so the rare credential-stuffing class
    # is represented in every partition.
    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=df["traffic_class"]
    )
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=RANDOM_SEED,
        stratify=train_val["traffic_class"],
    )
    _print_sizes(train, val, test)
    return train, val, test


def temporal_split(df):
    print("\n--- Strategy 2: Temporal Split ---")
    df_sorted = df.sort_values("session_start").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    _print_sizes(train, val, test)
    print(f"    Train period: {train['session_start'].min()} -> {train['session_start'].max()}")
    print(f"    Test period:  {test['session_start'].min()} -> {test['session_start'].max()}")
    return train, val, test


def entity_split(df):
    print("\n--- Strategy 3: Entity-Based Split (by IP) ---")
    unique_ips = df["ip"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    unique_ips = unique_ips.copy()
    rng.shuffle(unique_ips)
    n_ips = len(unique_ips)
    train_end = int(n_ips * TRAIN_RATIO)
    val_end = int(n_ips * (TRAIN_RATIO + VAL_RATIO))
    train_ips = set(unique_ips[:train_end])
    val_ips = set(unique_ips[train_end:val_end])
    test_ips = set(unique_ips[val_end:])
    train = df[df["ip"].isin(train_ips)]
    val = df[df["ip"].isin(val_ips)]
    test = df[df["ip"].isin(test_ips)]
    _print_sizes(train, val, test)
    assert set(train["ip"]) & set(test["ip"]) == set()
    assert set(train["ip"]) & set(val["ip"]) == set()
    assert set(val["ip"]) & set(test["ip"]) == set()
    print("  [OK] No IP leakage between partitions")
    return train, val, test


def ua_entity_split(df):
    print("\n--- Strategy 4: Entity-Based Split (by User-Agent) ---")
    unique_uas = np.array(df["useragent"].fillna("").unique())
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_uas)
    n_ua = len(unique_uas)
    train_end = int(n_ua * TRAIN_RATIO)
    val_end = int(n_ua * (TRAIN_RATIO + VAL_RATIO))
    ua = df["useragent"].fillna("")
    train = df[ua.isin(set(unique_uas[:train_end]))]
    val = df[ua.isin(set(unique_uas[train_end:val_end]))]
    test = df[ua.isin(set(unique_uas[val_end:]))]
    _print_sizes(train, val, test)
    assert set(train["useragent"].fillna("")) & set(test["useragent"].fillna("")) == set()
    assert set(train["useragent"].fillna("")) & set(val["useragent"].fillna("")) == set()
    assert set(val["useragent"].fillna("")) & set(test["useragent"].fillna("")) == set()
    print("  [OK] No User-Agent leakage between partitions")
    return train, val, test


def _print_sizes(train, val, test):
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        cs = int((part["traffic_class"] == 2).sum())
        print(f"  {name:<6}: {len(part):,} ({part['is_bot'].mean()*100:.1f}% bot, {cs} stuffing)")


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 3: Data Splitting")
    print("=" * 70)
    df = pd.read_parquet(SESSION_FEATURES)
    df["session_start"] = pd.to_datetime(df["session_start"], utc=True)
    print(f"Loaded {len(df):,} sessions | features: {len(feature_columns(df))}")

    strategies = {"random": random_split, "temporal": temporal_split, "entity": entity_split,
                  "ua_entity": ua_entity_split}
    summary = {}
    for name, fn in strategies.items():
        train, val, test = fn(df)
        for part_name, part_df in [("train", train), ("val", val), ("test", test)]:
            part_df.to_parquet(SPLITS_DIR / f"{name}_{part_name}.parquet", index=False)
        summary[name] = {
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "train_bot_ratio": float(train["is_bot"].mean()),
            "val_bot_ratio": float(val["is_bot"].mean()),
            "test_bot_ratio": float(test["is_bot"].mean()),
            "train_stuffing": int((train["traffic_class"] == 2).sum()),
            "val_stuffing": int((val["traffic_class"] == 2).sum()),
            "test_stuffing": int((test["traffic_class"] == 2).sum()),
            "train_unique_ips": int(train["ip"].nunique()),
            "test_unique_ips": int(test["ip"].nunique()),
        }

    (SPLITS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Splits saved in: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
