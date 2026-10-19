"""
Master runner for the Bot / Credential-Stuffing detection ML pipeline.

Usage (from repo root or ml_project):
  python ml_project/run_pipeline.py          # all steps
  python ml_project/run_pipeline.py 2 3 4 5  # skip step 1 if labeled data exists
  python ml_project/run_pipeline.py 4 5      # retrain + figures only
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
SRC = PROJECT_ROOT / "src"

# Prefer project venv if present
VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

STEPS = {
    1: ("step1_load_and_label.py", "Load & Label Data"),
    2: ("step2_features.py", "Feature Engineering & Sessionization"),
    3: ("step3_splitting.py", "Data Splitting (Random / Temporal / Entity)"),
    4: ("step4_train_evaluate.py", "Train & Evaluate (4 models x 3 splits)"),
    6: ("step6_multiclass.py", "Three-Class Detection (Human/Bot/Stuffing)"),
    7: ("step7_ablation.py", "Feature-Set Ablation"),
    8: ("step8_statistical_validation.py", "Statistical & Robustness Validation"),
    9: ("step9_stuffing_analysis.py", "Stuffing CIs & Rule-Adjacent Ablation"),
    5: ("step5_shap_analysis.py", "SHAP + Thesis Figures"),
}

# Execution order (figures are produced last so they can read every result file)
DEFAULT_ORDER = [1, 2, 3, 4, 6, 7, 8, 9, 5]


def run_step(n: int) -> bool:
    script, desc = STEPS[n]
    path = SRC / script
    print(f"\n{'#' * 70}\n# STEP {n}: {desc}\n# {path.name}\n{'#' * 70}\n")
    t0 = time.time()
    result = subprocess.run([PYTHON, str(path)], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"\n{'[OK]' if ok else '[FAIL]'} Step {n} {'done' if ok else 'FAILED'} in {elapsed/60:.1f} min")
    return ok


def main():
    print("=" * 70)
    print("  BOT & CREDENTIAL STUFFING DETECTION - ML PIPELINE")
    print("  Leakage-Aware Evaluation (Master Thesis)")
    print("=" * 70)
    print(f"Python: {PYTHON}")
    print(f"Project: {PROJECT_ROOT}")

    steps = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(DEFAULT_ORDER)
    # Skip step 1 automatically if labeled parquet already exists and user asked for all
    labeled = PROJECT_ROOT / "output" / "labeled_data.parquet"
    if steps == DEFAULT_ORDER and labeled.exists():
        print(f"\nLabeled data found -> skipping Step 1 ({labeled})")
        steps = [s for s in DEFAULT_ORDER if s != 1]

    print(f"Steps to run: {steps}\n")
    t0 = time.time()
    results = {}
    for n in steps:
        if n not in STEPS:
            print(f"Unknown step: {n}")
            continue
        ok = run_step(n)
        results[n] = ok
        if not ok:
            print(f"Stopping after failed step {n}.")
            break

    print(f"\n{'=' * 70}\nPIPELINE SUMMARY\n{'=' * 70}")
    for n, ok in results.items():
        print(f"  Step {n}: {STEPS[n][1]} -- {'OK' if ok else 'FAIL'}")
    print(f"Total: {(time.time() - t0)/60:.1f} min")
    pipeline_ok = all(results.values())

    out = PROJECT_ROOT / "output"
    if out.exists():
        print(f"\nOutputs under {out}:")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                rel = p.relative_to(out)
                label = f"{size/1024**2:.1f} MB" if size > 1024**2 else f"{size/1024:.0f} KB"
                print(f"  {rel} ({label})")

    if not pipeline_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
