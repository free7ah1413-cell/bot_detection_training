"""
Step 2: Feature Engineering & Sessionization (vectorized, memory-aware).

Sessions are (IP + User-Agent) tuples with a 30-minute idle timeout. The stage
produces both targets used in the thesis:
  * is_bot        - binary bot / human label (primary task)
  * traffic_class - Human / Bot / Credential Stuffing (three-class task)

Wide string columns are factorised into integer codes before grouping so the
4.1 M-row log fits comfortably in memory.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    CLASS_BOT,
    CLASS_HUMAN,
    CLASS_STUFFING,
    CLASS_NAMES,
    LABELED_PARQUET,
    OUTPUT_DIR,
    SESSION_FEATURES,
    SESSION_TIMEOUT_SEC,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# Authentication surface of the target application (VuFind library portal).
# Step 1 matched a generic "auth" keyword, which also captured the
# /themes/auth-new/ asset directory and /Author/ pages; the authentication
# surface is therefore re-derived here from explicit endpoint paths.
# ---------------------------------------------------------------------------
# Multiplier used to fold (entity, session number) into one integer session key.
SESSION_NO_LIMIT = 100_000

AUTH_AREA_RE = re.compile(r"/(?:MyResearch|Cart|Checkouts|Holds|Fines)/", re.I)
LOGIN_EP_RE = re.compile(
    r"/MyResearch/(?:UserLogin|Home|Login|Account)|wp-login\.php|/admin/login|xmlrpc\.php"
    r"|/login|/signin|/sign-in|/session/new|/auth/login|/user/login|/wp-admin",
    re.I,
)
LOGIN_SUBMIT_RE = re.compile(r"UserLogin|wp-login|/admin/login|/user/login", re.I)
CMS_PROBE_RE = re.compile(
    r"wp-login\.php|/wp-admin|xmlrpc\.php|/admin/login|/administrator|/phpmyadmin|/getcfg\.php",
    re.I,
)
STATIC_RE = re.compile(r"\.(?:js|css|jpg|jpeg|png|gif|ico|woff|ttf|svg)(?:\?|$)", re.I)

# Credential-stuffing labelling thresholds (documented in Section 4.4)
CS_MIN_LOGIN_ATTEMPTS = 3
CS_MIN_RPM = 6.0
CS_MIN_LOGIN_FAILURES = 3

READ_COLUMNS = [
    "ip", "useragent", "timestamp", "method", "resource",
    "response_code", "bytes", "referrer", "is_bot",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def add_request_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-request descriptors. String columns are dropped once consumed."""
    resource = df["resource"].fillna("-").astype(str)
    method = df["method"].fillna("").astype(str).str.upper()
    code = df["response_code"].fillna(0).astype(np.int32)

    split = resource.str.split("?", n=1)
    path = split.str[0]
    query = split.str[1]
    has_q = query.notna()

    df["uri_length"] = resource.str.len().astype(np.int32)
    df["uri_depth"] = path.str.count("/").astype(np.int16)
    df["has_query"] = has_q.astype(np.int8)
    df["query_param_count"] = np.where(has_q, query.fillna("").str.count("&") + 1, 0).astype(np.int16)

    df["is_2xx"] = code.between(200, 299).astype(np.int8)
    df["is_3xx"] = code.between(300, 399).astype(np.int8)
    df["is_4xx"] = code.between(400, 499).astype(np.int8)
    df["is_5xx"] = code.between(500, 599).astype(np.int8)

    hour = df["timestamp"].dt.hour.astype(np.int8)
    dow = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    df["is_get"] = (method == "GET").astype(np.int8)
    df["is_post"] = (method == "POST").astype(np.int8)

    df["is_ajax"] = resource.str.contains("/AJAX/", case=False, na=False).astype(np.int8)
    df["is_search"] = resource.str.contains("/Search/", case=False, na=False).astype(np.int8)
    df["is_record"] = resource.str.contains("/Record/", case=False, na=False).astype(np.int8)
    df["is_static"] = resource.str.contains(STATIC_RE).astype(np.int8)
    df["is_cover"] = resource.str.contains("/Cover/", case=False, na=False).astype(np.int8)

    # Authentication surface
    df["is_auth_area"] = resource.str.contains(AUTH_AREA_RE).astype(np.int8)
    is_login_ep = resource.str.contains(LOGIN_EP_RE)
    df["is_login_attempt"] = (
        is_login_ep & ((method == "POST") | resource.str.contains(LOGIN_SUBMIT_RE))
    ).astype(np.int8)
    df["is_cms_probe"] = resource.str.contains(CMS_PROBE_RE).astype(np.int8)
    df["is_login_fail"] = (
        (df["is_login_attempt"] == 1) & (code.between(400, 499) | (code == 302))
    ).astype(np.int8)

    referrer = df["referrer"].fillna("-").astype(str)
    has_ref = ~referrer.isin(["-", ""])
    df["has_referrer"] = has_ref.astype(np.int8)
    # Integer codes keep nunique() cheap without holding the full URL strings
    df["referrer_id"] = np.where(has_ref, pd.factorize(referrer)[0], -1).astype(np.int32)
    df["resource_id"] = pd.factorize(resource)[0].astype(np.int32)
    df["response_code"] = code

    df.drop(columns=["resource", "referrer", "method"], inplace=True)
    return df


def add_useragent_features(df: pd.DataFrame) -> pd.DataFrame:
    """User-agent descriptors; entropy is computed once per distinct string."""
    ua = df["useragent"].astype(str)
    entropy_map = {u: _shannon_entropy(u) for u in ua.unique()}
    df["ua_length"] = ua.str.len().astype(np.int16)
    df["ua_entropy"] = ua.map(entropy_map).astype(np.float32)
    df["ua_token_count"] = (ua.str.count(r"[ /;()]") + 1).astype(np.int16)
    return df


def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Group requests into (IP, User-Agent) sessions with an idle timeout."""
    ip_codes, ip_values = pd.factorize(df["ip"].astype(str))
    ua_codes, ua_values = pd.factorize(df["useragent"].astype(str))
    df["ip_id"] = ip_codes.astype(np.int32)
    df["ua_id"] = ua_codes.astype(np.int32)
    df["entity_id"] = (df["ip_id"].astype(np.int64) * len(ua_values) + df["ua_id"]).astype(np.int64)

    df.sort_values(["entity_id", "timestamp"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    gap = df.groupby("entity_id", sort=False)["timestamp"].diff().dt.total_seconds()
    new_session = (gap.isna() | (gap > SESSION_TIMEOUT_SEC)).astype(np.int32)
    df["session_no"] = new_session.groupby(df["entity_id"], sort=False).cumsum().astype(np.int32)
    df["interval"] = np.where(new_session == 1, np.nan, gap).astype(np.float32)
    # The composite key is only injective while no entity exceeds the multiplier,
    # so assert it rather than let a busier corpus merge two sessions silently.
    max_session_no = int(df["session_no"].max())
    if max_session_no >= SESSION_NO_LIMIT:
        raise ValueError(
            f"an entity produced {max_session_no} sessions, which exceeds the "
            f"session-key multiplier of {SESSION_NO_LIMIT}; raise SESSION_NO_LIMIT"
        )
    df["session_uid"] = pd.factorize(
        df["entity_id"].astype(np.int64) * SESSION_NO_LIMIT + df["session_no"]
    )[0].astype(np.int32)
    return df, ip_values, ua_values


def aggregate_sessions(df: pd.DataFrame, ip_values, ua_values) -> pd.DataFrame:
    g = df.groupby("session_uid", sort=False)

    uri_counts = df.groupby(["session_uid", "resource_id"], sort=False).size().rename("cnt")
    uri_repeat = uri_counts.groupby(level=0, sort=False).agg(
        max_uri_repeat="max", mean_uri_repeat="mean"
    )

    agg = g.agg(
        ip_id=("ip_id", "first"),
        ua_id=("ua_id", "first"),
        session_start=("timestamp", "min"),
        session_end=("timestamp", "max"),
        is_bot=("is_bot", "first"),
        n_requests=("resource_id", "size"),
        total_bytes=("bytes", "sum"),
        mean_bytes=("bytes", "mean"),
        std_bytes=("bytes", "std"),
        unique_uris=("resource_id", "nunique"),
        mean_uri_length=("uri_length", "mean"),
        mean_uri_depth=("uri_depth", "mean"),
        query_ratio=("has_query", "mean"),
        mean_query_params=("query_param_count", "mean"),
        n_2xx=("is_2xx", "sum"),
        n_3xx=("is_3xx", "sum"),
        n_4xx=("is_4xx", "sum"),
        n_5xx=("is_5xx", "sum"),
        unique_status_codes=("response_code", "nunique"),
        get_ratio=("is_get", "mean"),
        post_ratio=("is_post", "mean"),
        ajax_ratio=("is_ajax", "mean"),
        search_ratio=("is_search", "mean"),
        record_ratio=("is_record", "mean"),
        static_ratio=("is_static", "mean"),
        cover_ratio=("is_cover", "mean"),
        auth_area_count=("is_auth_area", "sum"),
        n_login_attempts=("is_login_attempt", "sum"),
        n_login_failures=("is_login_fail", "sum"),
        n_cms_probes=("is_cms_probe", "sum"),
        referrer_ratio=("has_referrer", "mean"),
        unique_referrers=("referrer_id", "nunique"),
        ua_length=("ua_length", "first"),
        ua_entropy=("ua_entropy", "first"),
        ua_token_count=("ua_token_count", "first"),
        hour_mean=("hour", "mean"),
        hour_std=("hour", "std"),
        hour_sin_mean=("hour_sin", "mean"),
        hour_cos_mean=("hour_cos", "mean"),
        dow_sin_mean=("dow_sin", "mean"),
        dow_cos_mean=("dow_cos", "mean"),
        mean_interval=("interval", "mean"),
        median_interval=("interval", "median"),
        std_interval=("interval", "std"),
        min_interval=("interval", "min"),
        max_interval=("interval", "max"),
    )
    agg = agg.join(uri_repeat).reset_index().rename(columns={"session_uid": "session_id"})

    # A referrer_id of -1 marks "no referrer" and must not count as a distinct value
    no_ref = (agg["referrer_ratio"] == 0)
    agg.loc[no_ref, "unique_referrers"] = 0

    agg["ip"] = pd.Index(ip_values)[agg["ip_id"].values]
    agg["useragent"] = pd.Index(ua_values)[agg["ua_id"].values]
    agg["is_bot"] = agg["is_bot"].astype(int)

    agg["session_duration_sec"] = (agg["session_end"] - agg["session_start"]).dt.total_seconds()
    agg["requests_per_minute"] = np.where(
        agg["session_duration_sec"] > 0,
        agg["n_requests"] / (agg["session_duration_sec"] / 60.0),
        agg["n_requests"],
    )
    agg["uri_diversity"] = agg["unique_uris"] / agg["n_requests"]
    agg["error_rate_4xx"] = agg["n_4xx"] / agg["n_requests"]
    agg["error_rate_5xx"] = agg["n_5xx"] / agg["n_requests"]
    agg["auth_area_ratio"] = agg["auth_area_count"] / agg["n_requests"]
    agg["cms_probe_ratio"] = agg["n_cms_probes"] / agg["n_requests"]
    agg["login_failure_rate"] = np.where(
        agg["n_login_attempts"] > 0, agg["n_login_failures"] / agg["n_login_attempts"], 0.0
    )
    agg["cv_interval"] = np.where(
        agg["mean_interval"].fillna(0) > 0,
        agg["std_interval"].fillna(0) / agg["mean_interval"],
        0.0,
    )

    for col in [
        "std_bytes", "hour_std", "mean_interval", "median_interval",
        "std_interval", "min_interval", "max_interval", "cv_interval",
        "max_uri_repeat", "mean_uri_repeat",
    ]:
        agg[col] = agg[col].fillna(0)

    agg = assign_traffic_class(agg)

    ordered = [
        "session_id", "ip", "useragent", "session_start", "session_end",
        "is_bot", "traffic_class",
        "n_requests", "session_duration_sec", "requests_per_minute", "total_bytes",
        "mean_interval", "median_interval", "std_interval", "min_interval",
        "max_interval", "cv_interval",
        "unique_uris", "uri_diversity", "mean_uri_length", "mean_uri_depth",
        "query_ratio", "mean_query_params", "max_uri_repeat", "mean_uri_repeat",
        "n_2xx", "n_3xx", "n_4xx", "n_5xx", "error_rate_4xx", "error_rate_5xx",
        "unique_status_codes", "mean_bytes", "std_bytes",
        "get_ratio", "post_ratio", "ajax_ratio", "search_ratio", "record_ratio",
        "static_ratio", "cover_ratio",
        "auth_area_count", "auth_area_ratio",
        "n_login_attempts", "n_login_failures", "login_failure_rate",
        "n_cms_probes", "cms_probe_ratio",
        "referrer_ratio", "unique_referrers",
        "ua_length", "ua_entropy", "ua_token_count",
        "hour_mean", "hour_std", "hour_sin_mean", "hour_cos_mean",
        "dow_sin_mean", "dow_cos_mean",
    ]
    return agg[ordered]


def assign_traffic_class(agg: pd.DataFrame) -> pd.DataFrame:
    """Three-class target: Human (0), Bot (1), Credential Stuffing (2).

    A session is labelled credential stuffing when it shows authentication-targeted
    automation: probing of CMS login endpoints that do not exist on the host
    application, repeated login attempts issued at machine speed, or a burst of
    failed login attempts. The rule depends only on authentication behaviour, not on
    the User-Agent string, so it stays independent of the bot heuristic.
    """
    stuffing = (
        (agg["n_cms_probes"] >= 1)
        | ((agg["n_login_attempts"] >= CS_MIN_LOGIN_ATTEMPTS)
           & (agg["requests_per_minute"] >= CS_MIN_RPM))
        | (agg["n_login_failures"] >= CS_MIN_LOGIN_FAILURES)
    )
    agg["traffic_class"] = np.where(
        stuffing, CLASS_STUFFING, np.where(agg["is_bot"] == 1, CLASS_BOT, CLASS_HUMAN)
    ).astype(int)
    return agg


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 2: Feature Engineering & Sessionization")
    print("=" * 70)
    print(f"Loading: {LABELED_PARQUET}")
    df = pd.read_parquet(LABELED_PARQUET, columns=READ_COLUMNS)
    print(f"Loaded {len(df):,} records")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print("Extracting request-level features...")
    df = add_request_features(df)
    print("Extracting user-agent features...")
    df = add_useragent_features(df)
    print("Building sessions...")
    df, ip_values, ua_values = assign_sessions(df)
    df.drop(columns=["ip", "useragent"], inplace=True)
    print(f"  Total sessions: {df['session_uid'].nunique():,}")

    print("Aggregating session features...")
    sf = aggregate_sessions(df, ip_values, ua_values)
    del df

    print(f"\nTotal sessions: {len(sf):,}")
    print("Binary target (is_bot):")
    print(f"  Human: {(sf['is_bot'] == 0).sum():,} ({(sf['is_bot'] == 0).mean()*100:.2f}%)")
    print(f"  Bot:   {(sf['is_bot'] == 1).sum():,} ({(sf['is_bot'] == 1).mean()*100:.2f}%)")
    print("Three-class target (traffic_class):")
    for cls, name in enumerate(CLASS_NAMES):
        n = int((sf["traffic_class"] == cls).sum())
        print(f"  {name:<20}: {n:,} ({n / len(sf) * 100:.2f}%)")

    sf.to_parquet(SESSION_FEATURES, index=False)
    print(f"\nSaved: {SESSION_FEATURES} ({SESSION_FEATURES.stat().st_size / 1024**2:.1f} MB)")

    feature_cols = [
        c for c in sf.columns
        if c not in {"session_id", "ip", "useragent", "session_start",
                     "session_end", "is_bot", "traffic_class"}
    ]
    sf[feature_cols].describe().round(4).to_csv(OUTPUT_DIR / "feature_summary.csv")
    print(f"Saved: feature_summary.csv ({len(feature_cols)} features)")

    label_stats = {
        "n_sessions": int(len(sf)),
        "n_features": len(feature_cols),
        "binary": {
            "human": int((sf["is_bot"] == 0).sum()),
            "bot": int((sf["is_bot"] == 1).sum()),
        },
        "three_class": {
            "human": int((sf["traffic_class"] == 0).sum()),
            "bot": int((sf["traffic_class"] == 1).sum()),
            "credential_stuffing": int((sf["traffic_class"] == 2).sum()),
        },
        "labelling_rule": {
            "min_login_attempts": CS_MIN_LOGIN_ATTEMPTS,
            "min_requests_per_minute": CS_MIN_RPM,
            "min_login_failures": CS_MIN_LOGIN_FAILURES,
        },
        "totals": {
            "login_attempts": int(sf["n_login_attempts"].sum()),
            "login_failures": int(sf["n_login_failures"].sum()),
            "cms_probes": int(sf["n_cms_probes"].sum()),
            "auth_area_requests": int(sf["auth_area_count"].sum()),
        },
        "session_stats": {
            "mean_requests": float(sf["n_requests"].mean()),
            "median_requests": float(sf["n_requests"].median()),
            "max_requests": int(sf["n_requests"].max()),
            "mean_duration_sec": float(sf["session_duration_sec"].mean()),
        },
    }
    (OUTPUT_DIR / "label_summary.json").write_text(json.dumps(label_stats, indent=2), encoding="utf-8")
    print("Saved: label_summary.json")


if __name__ == "__main__":
    main()
