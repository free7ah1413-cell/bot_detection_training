"""Fast preflight checks before the long training run."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "public_v2.json"
EXPECTED_SIZE = 3_218_672_736
MIN_FREE_BYTES = 8 * 1024**3


def main() -> int:
    print("=" * 68)
    print("TRAINING PACKAGE - READINESS CHECK")
    print("=" * 68)
    print(f"Python: {sys.version.split()[0]}")

    errors: list[str] = []
    warnings: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required.")

    if not DATASET.is_file():
        errors.append(f"Dataset is missing: {DATASET}")
    else:
        size = DATASET.stat().st_size
        print(f"Dataset: {DATASET}")
        print(f"Dataset size: {size:,} bytes ({size / 1024**3:.2f} GiB)")
        if size != EXPECTED_SIZE:
            warnings.append(
                f"Dataset size differs from the expected {EXPECTED_SIZE:,} bytes. "
                "Confirm that public_v2.json finished downloading."
            )
        try:
            with DATASET.open("rb") as handle:
                first = handle.read(4096).lstrip()[:1]
            if first != b"{":
                errors.append("Dataset does not look like the expected top-level JSON object.")
        except OSError as exc:
            errors.append(f"Dataset cannot be read: {exc}")

    free = shutil.disk_usage(ROOT).free
    print(f"Free disk space: {free / 1024**3:.1f} GiB")
    if free < MIN_FREE_BYTES:
        warnings.append("Less than 8 GiB is free; 10 GiB or more is recommended.")

    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}")

    if errors:
        print("\nReadiness check FAILED. Fix the errors above before training.")
        return 1
    print("\nReadiness check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
