"""Build a self-contained English HTML summary from the training artefacts."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
REPORT = OUTPUT / "results_report.html"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (FileNotFoundError, OSError, csv.Error):
        return []


def number(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def integer(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "-"


def score(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("-inf")


def metric_table(rows: list[dict[str, str]], columns: list[tuple[str, str, str]]) -> str:
    if not rows:
        return '<p class="empty">These results have not been generated yet.</p>'
    heads = "".join(f"<th>{escape(label)}</th>" for _, label, _ in columns)
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            raw = row.get(key, "")
            if kind == "score":
                shown = number(raw)
            elif kind == "int":
                shown = integer(raw)
            elif kind == "seconds":
                shown = f"{number(raw, 1)} s"
            else:
                shown = str(raw or "-")
            cells.append(f"<td>{escape(shown)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def card(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="card">'
        f'<div class="card-label">{escape(label)}</div>'
        f'<div class="card-value">{escape(value)}</div>'
        f'<div class="card-note">{escape(note)}</div>'
        '</div>'
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = read_json(OUTPUT / "dataset_summary.json")
    labels = read_json(OUTPUT / "label_summary.json")
    validation = read_json(OUTPUT / "validation_status.json")
    binary = read_csv(RESULTS / "all_results.csv")
    multiclass = read_csv(RESULTS / "multiclass_results.csv")

    binary = [r for r in binary if r.get("model") and score(r, "f1_score") != float("-inf")]
    multiclass = [r for r in multiclass if r.get("model") and score(r, "macro_f1") != float("-inf")]
    binary.sort(key=lambda row: score(row, "f1_score"), reverse=True)
    multiclass.sort(key=lambda row: score(row, "macro_f1"), reverse=True)

    complete = bool(binary and multiclass and validation.get("passed") is True)
    if complete:
        status_class = "ok"
        status_text = f"Training completed and passed {validation.get('checks', 0)} integrity checks."
    elif binary or multiclass:
        status_class = "warn"
        status_text = "Partial results are available, but training or validation is not complete."
    else:
        status_class = "pending"
        status_text = "Training is not complete. Run START_TRAINING.bat first."

    binary_best = binary[0] if binary else {}
    multi_best = multiclass[0] if multiclass else {}
    three = labels.get("three_class", {})

    cards = "".join([
        card("Log Records", integer(dataset.get("total_records"))),
        card("Sessions", integer(labels.get("n_sessions"))),
        card("Bot Sessions", integer(three.get("bot"))),
        card("Credential-Stuffing Sessions", integer(three.get("credential_stuffing"))),
        card(
            "Best Binary F1",
            number(binary_best.get("f1_score")),
            f"{binary_best.get('model', '-')} / {binary_best.get('split_strategy', '-')}",
        ),
        card(
            "Best Multiclass Macro-F1",
            number(multi_best.get("macro_f1")),
            f"{multi_best.get('model', '-')} / {multi_best.get('split_strategy', '-')}",
        ),
    ])

    binary_columns = [
        ("model", "Model", "text"),
        ("split_strategy", "Split", "text"),
        ("precision", "Precision", "score"),
        ("recall", "Recall", "score"),
        ("f1_score", "F1", "score"),
        ("roc_auc", "ROC-AUC", "score"),
        ("pr_auc", "PR-AUC", "score"),
        ("fpr", "FPR", "score"),
        ("train_time_sec", "Training Time", "seconds"),
    ]
    multi_columns = [
        ("model", "Model", "text"),
        ("split_strategy", "Split", "text"),
        ("macro_f1", "Macro-F1", "score"),
        ("weighted_f1", "Weighted-F1", "score"),
        ("macro_roc_auc_ovr", "ROC-AUC OVR", "score"),
        ("stuffing_precision", "Stuffing Precision", "score"),
        ("stuffing_recall", "Stuffing Recall", "score"),
        ("stuffing_f1", "Stuffing F1", "score"),
        ("stuffing_support", "Stuffing Support", "int"),
    ]

    preferred_figures = [
        ("fig_architecture.png", "Detection Pipeline Architecture"),
        ("fig_dataset_overview.png", "Dataset Overview"),
        ("fig_http_methods.png", "HTTP Method Distribution"),
        ("fig_session_distributions.png", "Session Feature Distributions"),
        ("fig_three_class_overview.png", "Three-Class Dataset Overview"),
        ("fig_comparative_performance.png", "Model Performance Comparison"),
        ("fig_leakage_impact_heatmap.png", "Split-Strategy Leakage Impact"),
        ("fig_leakage_delta.png", "F1 Leakage Delta"),
        ("fig_confusion_matrices.png", "Binary Confusion Matrices"),
        ("fig_multiclass_confusion.png", "Three-Class Confusion Matrices"),
        ("fig_roc_curves.png", "ROC Curves"),
        ("fig_feature_importance.png", "Feature Importance"),
        ("fig_shap_summary.png", "SHAP Summary"),
        ("fig_shap_bar.png", "Mean Absolute SHAP Importance"),
        ("fig_ablation.png", "Feature-Set Ablation"),
        ("fig_circularity.png", "Circularity Effect"),
        ("fig_bootstrap_ci.png", "Bootstrap Confidence Intervals"),
    ]
    figure_html = []
    for filename, caption in preferred_figures:
        if (FIGURES / filename).is_file():
            figure_html.append(
                '<figure>'
                f'<img src="figures/{escape(filename)}" alt="{escape(caption)}">'
                f'<figcaption>{escape(caption)}</figcaption>'
                '</figure>'
            )
    if not figure_html:
        figure_html.append('<p class="empty">Figures will appear here after training is complete.</p>')

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    html = f"""<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bot Detection Training Results</title>
  <style>
    :root {{ --ink:#172033; --muted:#667085; --line:#e4e7ec; --brand:#175cd3; --bg:#f7f9fc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Arial,sans-serif; line-height:1.6; }}
    main {{ width:min(1500px,94%); margin:32px auto 64px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(25px,4vw,42px); }}
    h2 {{ margin-top:36px; border-bottom:2px solid var(--line); padding-bottom:8px; }}
    .subtitle,.generated {{ color:var(--muted); }}
    .status {{ margin:22px 0; padding:15px 18px; border-radius:12px; font-weight:bold; }}
    .status.ok {{ background:#ecfdf3; color:#027a48; border:1px solid #abefc6; }}
    .status.warn {{ background:#fffaeb; color:#b54708; border:1px solid #fedf89; }}
    .status.pending {{ background:#eff8ff; color:#175cd3; border:1px solid #b2ddff; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }}
    .card {{ background:white; border:1px solid var(--line); border-radius:14px; padding:16px; box-shadow:0 1px 3px #10182812; }}
    .card-label {{ color:var(--muted); font-size:14px; }}
    .card-value {{ color:var(--brand); font-size:27px; font-weight:bold; text-align:left; }}
    .card-note {{ color:var(--muted); font-size:12px; text-align:left; min-height:18px; }}
    .table-wrap {{ overflow:auto; background:white; border:1px solid var(--line); border-radius:12px; }}
    table {{ width:100%; border-collapse:collapse; white-space:nowrap; text-align:center; }}
    th {{ background:#eef4ff; color:#1849a9; position:sticky; top:0; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); }}
    tbody tr:hover {{ background:#f9fafb; }}
    .figures {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
    figure {{ margin:0; background:white; border:1px solid var(--line); border-radius:14px; padding:12px; }}
    figure img {{ display:block; width:100%; height:auto; border-radius:8px; }}
    figcaption {{ text-align:center; color:var(--muted); margin-top:8px; }}
    .empty {{ background:white; border:1px dashed #98a2b3; border-radius:12px; padding:20px; color:var(--muted); }}
    code {{ display:inline-block; }}
  </style>
</head>
<body>
<main>
  <h1>Bot Detection Training Results</h1>
  <div class="subtitle">Bot &amp; Credential-Stuffing Detection - Leakage-Aware Evaluation</div>
  <div class="generated">Report generated: {escape(generated)}</div>
  <div class="status {status_class}">{escape(status_text)}</div>
  <section class="cards">{cards}</section>
  <h2>Binary Classification Results: Human / Bot</h2>
  {metric_table(binary, binary_columns)}
  <h2>Three-Class Classification Results</h2>
  {metric_table(multiclass, multi_columns)}
  <h2>Training Figures</h2>
  <section class="figures">{"".join(figure_html)}</section>
  <h2>Output Locations</h2>
  <p>Models: <code>output/models</code> - Tables: <code>output/results</code> - Figures: <code>output/figures</code></p>
</main>
</body>
</html>
"""
    REPORT.write_text(html, encoding="utf-8")

    print("=" * 68)
    print("RESULTS SUMMARY")
    print("=" * 68)
    print(status_text)
    if binary_best:
        print(
            "Best binary: "
            f"{binary_best.get('model')} / {binary_best.get('split_strategy')} / "
            f"F1={number(binary_best.get('f1_score'))}"
        )
    if multi_best:
        print(
            "Best multiclass: "
            f"{multi_best.get('model')} / {multi_best.get('split_strategy')} / "
            f"Macro-F1={number(multi_best.get('macro_f1'))}"
        )
    print(f"HTML report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
