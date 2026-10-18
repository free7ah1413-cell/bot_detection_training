"""
Step 1: Load public_v2.json and label Bot/Human traffic.
Streams the large JSON with ijson and writes labeled_data.parquet.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import ijson
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import RAW_JSON, OUTPUT_DIR, LABELED_PARQUET, DATASET_SUMMARY, ensure_dirs

BOT_PATTERNS = [
    r"bot\b", r"spider", r"crawl", r"slurp", r"scraper",
    r"fetcher", r"wget", r"curl", r"python-requests", r"python-urllib",
    r"httpclient", r"java/", r"apache-httpclient", r"okhttp",
    r"go-http-client", r"mechanize", r"phantom", r"headless",
    r"selenium", r"bubing", r"icc-crawler", r"nutch",
    r"archive\.org", r"ia_archiver", r"yandex", r"bingbot",
    r"googlebot", r"baiduspider", r"duckduckbot", r"yahoo",
    r"semrush", r"ahrefs", r"mj12bot", r"dotbot", r"rogerbot",
    r"linkfluence", r"screaming frog", r"sistrix", r"seokicks",
    r"petalbot", r"bytespider", r"applebot", r"cfnetwork",
    r"facebookexternalhit", r"twitterbot", r"linkedinbot",
    r"whatsapp", r"telegrambot", r"discordbot",
    r"pingdom", r"uptimerobot", r"statuscake",
    r"zgrab", r"masscan", r"nmap", r"nikto", r"sqlmap",
    r"^-$", r"^$",
]
BOT_REGEX = re.compile("|".join(BOT_PATTERNS), re.IGNORECASE)
# Authentication surface of the host application. The pattern is anchored on real
# endpoint paths so that unrelated assets (e.g. /themes/auth-new/, /Author/) are
# not counted as authentication traffic.
AUTH_PATTERNS = re.compile(
    r"(/MyResearch/|/Cart/|/Checkouts/|/Holds/|/Fines/|wp-login\.php|/wp-admin"
    r"|xmlrpc\.php|/administrator|/phpmyadmin|/admin/login|/login|/signin|/sign-in"
    r"|/session/new|/auth/login|/user/login|/password|/register|/signup)",
    re.IGNORECASE,
)


def is_bot(useragent: str) -> bool:
    if not useragent or useragent.strip() in ["-", ""]:
        return True
    return bool(BOT_REGEX.search(useragent))


def detect_auth_uri(resource: str) -> bool:
    if not resource:
        return False
    return bool(AUTH_PATTERNS.search(resource))


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 1: Loading and Labeling Web Log Data")
    print("=" * 70)
    print(f"Input: {RAW_JSON}")
    if not RAW_JSON.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_JSON}")
    print(f"File size: {RAW_JSON.stat().st_size / (1024**3):.2f} GB\n")

    records = []
    batch_size = 500_000
    batch_num = 0
    total_count = 0
    batch_files = []

    with open(RAW_JSON, "rb") as f:
        parser = ijson.items(f, "", use_float=True)
        for full_obj in parser:
            for record_id, entry in tqdm(full_obj.items(), desc="Processing records"):
                try:
                    ua = entry.get("useragent", "-") or "-"
                    resource = entry.get("resource", "") or ""
                    method = entry.get("method", "") or ""
                    response_code = entry.get("response", "") or ""
                    bytes_sent = entry.get("bytes", "0") or "0"
                    ip = entry.get("ip", "") or ""
                    referrer = entry.get("referrer", "-") or "-"
                    timestamp_str = entry.get("timestamp", "") or ""

                    ts = None
                    if timestamp_str:
                        try:
                            ts = pd.Timestamp(timestamp_str)
                        except Exception:
                            ts = None

                    try:
                        bytes_val = int(bytes_sent)
                    except (ValueError, TypeError):
                        bytes_val = 0

                    try:
                        resp_code = int(response_code)
                    except (ValueError, TypeError):
                        resp_code = 0

                    records.append({
                        "record_id": record_id,
                        "ip": ip,
                        "timestamp": ts,
                        "method": method,
                        "resource": resource,
                        "response_code": resp_code,
                        "bytes": bytes_val,
                        "referrer": referrer,
                        "useragent": ua,
                        "is_bot": is_bot(ua),
                        "is_auth_endpoint": detect_auth_uri(resource),
                    })
                    total_count += 1

                    if len(records) >= batch_size:
                        batch_num += 1
                        batch_file = OUTPUT_DIR / f"raw_batch_{batch_num}.parquet"
                        pd.DataFrame(records).to_parquet(batch_file, index=False)
                        batch_files.append(batch_file)
                        print(f"  Saved batch {batch_num}: {len(records):,} records")
                        records = []
                except Exception:
                    continue

    if records:
        batch_num += 1
        batch_file = OUTPUT_DIR / f"raw_batch_{batch_num}.parquet"
        pd.DataFrame(records).to_parquet(batch_file, index=False)
        batch_files.append(batch_file)
        print(f"  Saved batch {batch_num}: {len(records):,} records")

    print(f"\nTotal records processed: {total_count:,}")
    print("Merging batches...")
    df = pd.concat([pd.read_parquet(bf) for bf in batch_files], ignore_index=True)
    for bf in batch_files:
        bf.unlink(missing_ok=True)

    print(f"Total records: {len(df):,}")
    print(f"  Human: {(~df['is_bot']).sum():,} ({(~df['is_bot']).mean()*100:.1f}%)")
    print(f"  Bot:   {df['is_bot'].sum():,} ({df['is_bot'].mean()*100:.1f}%)")

    before = len(df)
    df = df.dropna(subset=["ip", "timestamp", "resource"])
    print(f"Dropped {before - len(df):,} rows with missing critical fields")
    df = df.sort_values("timestamp").reset_index(drop=True)

    auth_count = int(df["is_auth_endpoint"].sum())
    df.to_parquet(LABELED_PARQUET, index=False)
    print(f"Saved: {LABELED_PARQUET} ({LABELED_PARQUET.stat().st_size / 1024**2:.1f} MB)")

    summary = {
        "total_records": int(len(df)),
        "date_range_start": str(df["timestamp"].min()),
        "date_range_end": str(df["timestamp"].max()),
        "unique_ips": int(df["ip"].nunique()),
        "unique_user_agents": int(df["useragent"].nunique()),
        "unique_resources": int(df["resource"].nunique()),
        "bot_count": int(df["is_bot"].sum()),
        "human_count": int((~df["is_bot"]).sum()),
        "bot_percentage": float(df["is_bot"].mean() * 100),
        "auth_endpoint_requests": auth_count,
        "response_code_dist": {str(k): int(v) for k, v in df["response_code"].value_counts().head(10).items()},
        "method_dist": {str(k): int(v) for k, v in df["method"].value_counts().items()},
    }
    DATASET_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {DATASET_SUMMARY}")


if __name__ == "__main__":
    main()
