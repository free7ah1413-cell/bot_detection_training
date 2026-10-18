"""Portable project configuration; every path resolves from this folder."""
import os
from pathlib import Path

# Project root = the folder received by the trainer.
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT

# Put the downloaded 3.2 GB file at data/public_v2.json. An advanced user can
# point to an existing copy elsewhere with the PUBLIC_V2_JSON environment variable.
DATA_DIR = PROJECT_ROOT / "data"
_configured_dataset = os.environ.get("PUBLIC_V2_JSON")
RAW_JSON = Path(_configured_dataset).expanduser() if _configured_dataset else DATA_DIR / "public_v2.json"
if not RAW_JSON.is_absolute():
    RAW_JSON = (PROJECT_ROOT / RAW_JSON).resolve()

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS_DIR = OUTPUT_DIR / "splits"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

LABELED_PARQUET = OUTPUT_DIR / "labeled_data.parquet"
SESSION_FEATURES = OUTPUT_DIR / "session_features.parquet"
DATASET_SUMMARY = OUTPUT_DIR / "dataset_summary.json"

# Modeling
RANDOM_SEED = 42
SESSION_TIMEOUT_SEC = 30 * 60  # 30 minutes
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SPLIT_STRATEGIES = ["random", "temporal", "entity", "ua_entity"]

# Class encoding for the three-class task
CLASS_HUMAN, CLASS_BOT, CLASS_STUFFING = 0, 1, 2
CLASS_NAMES = ["Human", "Bot", "Credential Stuffing"]

# Identifier / metadata / target columns never used as model inputs
EXCLUDE_COLS = [
    "session_id", "ip", "useragent", "session_start", "session_end",
    "is_bot", "traffic_class",
]

# Features used to derive the credential-stuffing label. They are withheld from
# the three-class task so the classifier cannot trivially invert the labelling rule.
LABEL_DEFINING_COLS = [
    "n_login_attempts", "n_login_failures", "login_failure_rate",
    "n_cms_probes", "cms_probe_ratio",
]

# Descriptors computed from the User-Agent string. The bot label is itself derived
# from that string, so feeding these back to the classifier would be circular: the
# model would recover the labelling rule instead of learning behaviour. They are
# excluded from every reported model and retained only for the circularity
# demonstration in the ablation study (Section 5.3.4).
UA_DERIVED_COLS = ["ua_length", "ua_entropy", "ua_token_count"]

# Feature groups used for the ablation study (Section 5.4)
REQUEST_LEVEL_FEATURES = [
    "mean_uri_length", "mean_uri_depth", "query_ratio", "mean_query_params",
    "n_2xx", "n_3xx", "n_4xx", "n_5xx", "error_rate_4xx", "error_rate_5xx",
    "mean_bytes", "std_bytes", "total_bytes",
    "get_ratio", "post_ratio", "ajax_ratio", "search_ratio", "record_ratio",
    "static_ratio", "cover_ratio",
    "hour_mean", "hour_std", "hour_sin_mean", "hour_cos_mean",
    "dow_sin_mean", "dow_cos_mean",
]

SESSION_LEVEL_FEATURES = [
    "n_requests", "session_duration_sec", "requests_per_minute",
    "mean_interval", "median_interval", "std_interval", "min_interval",
    "max_interval", "cv_interval",
    "unique_uris", "uri_diversity", "max_uri_repeat", "mean_uri_repeat",
    "unique_status_codes", "referrer_ratio", "unique_referrers",
    "auth_area_count", "auth_area_ratio",
]


def ensure_dirs():
    for d in (OUTPUT_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def feature_columns(df, exclude_label_defining=False, include_ua_derived=False):
    """Model input columns.

    exclude_label_defining -- drop the credential-stuffing rule inputs (3-class task)
    include_ua_derived     -- re-admit the circular User-Agent descriptors; used only
                              by the ablation study that quantifies the inflation they
                              cause.
    """
    blocked = set(EXCLUDE_COLS)
    if not include_ua_derived:
        blocked |= set(UA_DERIVED_COLS)
    if exclude_label_defining:
        blocked |= set(LABEL_DEFINING_COLS)
    return [c for c in df.columns if c not in blocked]
