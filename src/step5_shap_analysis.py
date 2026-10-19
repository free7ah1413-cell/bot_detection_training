"""
Step 5: SHAP analysis + thesis figures (PNG).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import auc, roc_curve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (
    OUTPUT_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    DATASET_SUMMARY,
    SESSION_FEATURES,
    SPLIT_STRATEGIES,
    ensure_dirs,
    feature_columns,
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X


def select_positive_class_shap(shap_values, n_samples, n_features):
    """Return a 2-D SHAP matrix for the positive class of a binary model."""
    if isinstance(shap_values, list):
        # SHAP < 0.45 returned one (samples, features) array per class.
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        # Current SHAP returns (samples, features, outputs) for classifiers.
        if values.shape[:2] == (n_samples, n_features):
            values = values[:, :, 1]
        # Retain compatibility with output-first arrays from other explainers.
        elif values.shape[1:] == (n_samples, n_features):
            values = values[1]

    expected_shape = (n_samples, n_features)
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {expected_shape} "
            "after selecting the positive class."
        )
    return values


def plot_dataset_overview():
    print("  dataset overview...")
    summary = json.loads(DATASET_SUMMARY.read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.pie(
        [summary["human_count"], summary["bot_count"]],
        explode=(0, 0.05), labels=["Human", "Bot"], colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%", shadow=True, startangle=90,
    )
    ax1.set_title("Traffic Distribution: Human vs Bot")
    resp = summary.get("response_code_dist", {})
    if resp:
        codes, counts = list(resp.keys()), list(resp.values())
        ax2.bar(range(len(codes)), counts, color="#3498db", edgecolor="white")
        ax2.set_xticks(range(len(codes)))
        ax2.set_xticklabels(codes, rotation=45)
        ax2.set_xlabel("HTTP Response Code")
        ax2.set_ylabel("Count")
        ax2.set_title("Response Code Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_dataset_overview.png", bbox_inches="tight")
    plt.close()

    methods = summary.get("method_dist", {})
    if methods:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(methods.keys()), list(methods.values()), color="#3498db")
        ax.set_xlabel("Count")
        ax.set_title("HTTP Method Distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_http_methods.png", bbox_inches="tight")
        plt.close()


def plot_session_distributions():
    print("  session distributions...")
    sf = pd.read_parquet(SESSION_FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    specs = [
        ("n_requests", "Requests per Session", True),
        ("session_duration_sec", "Session Duration (sec)", True),
        ("requests_per_minute", "Requests per Minute", True),
        ("unique_uris", "Unique URIs per Session", True),
        ("mean_interval", "Mean Inter-Request Interval", True),
        ("cv_interval", "CV of Interval", False),
    ]
    for idx, (feat, title, use_log) in enumerate(specs):
        ax = axes[idx // 3][idx % 3]
        cap = sf[feat].quantile(0.99)
        h = sf[sf["is_bot"] == 0][feat].clip(upper=cap)
        b = sf[sf["is_bot"] == 1][feat].clip(upper=cap)
        ax.hist(h, bins=50, alpha=0.6, label="Human", color="#2ecc71", density=True)
        ax.hist(b, bins=50, alpha=0.6, label="Bot", color="#e74c3c", density=True)
        ax.set_title(title)
        ax.legend()
        if use_log:
            ax.set_yscale("log")
    plt.suptitle("Session Feature Distributions: Human vs Bot", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_session_distributions.png", bbox_inches="tight")
    plt.close()


def plot_comparative_performance():
    print("  comparative performance...")
    results = pd.read_csv(RESULTS_DIR / "all_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (metric, title) in enumerate([
        ("f1_score", "F1-Score"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")
    ]):
        ax = axes[idx]
        if metric not in results.columns:
            continue
        pivot = results.pivot_table(values=metric, index="model", columns="split_strategy")
        x = np.arange(len(pivot.index))
        n_split = len(SPLIT_STRATEGIES)
        width = 0.8 / n_split
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        for i, strategy in enumerate(SPLIT_STRATEGIES):
            if strategy not in pivot.columns:
                continue
            vals = pivot[strategy].values
            bars = ax.bar(x + i * width, vals, width, label=strategy.capitalize(), color=colors[i])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + width * (n_split - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Model and Split")
        ax.legend()
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_comparative_performance.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_f1 = results.pivot_table(values="f1_score", index="model", columns="split_strategy")
    if "random" in pivot_f1.columns:
        diff = pivot_f1.copy()
        for col in diff.columns:
            diff[col] = pivot_f1["random"] - pivot_f1[col]
        sns.heatmap(diff, annot=True, fmt=".4f", cmap="RdYlGn_r", center=0, ax=ax, linewidths=1)
        ax.set_title("F1 Inflation: Random − Other Splits")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig_leakage_impact_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrices():
    print("  confusion matrices...")
    results = pd.read_csv(RESULTS_DIR / "all_results.csv")
    models = list(results["model"].dropna().unique())
    n_split = len(SPLIT_STRATEGIES)
    fig, axes = plt.subplots(len(models), n_split, figsize=(5 * n_split, 4.5 * len(models)))
    if len(models) == 1:
        axes = np.array([axes])
    for row, model_name in enumerate(models):
        for col, strategy in enumerate(SPLIT_STRATEGIES):
            ax = axes[row][col]
            row_data = results[(results["model"] == model_name) & (results["split_strategy"] == strategy)]
            if len(row_data) and "tp" in row_data.columns:
                tp, fp = int(row_data["tp"].iloc[0]), int(row_data["fp"].iloc[0])
                fn, tn = int(row_data["fn"].iloc[0]), int(row_data["tn"].iloc[0])
                cm = np.array([[tn, fp], [fn, tp]])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            ax.set_title(f"{model_name}\n({strategy})", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_confusion_matrices.png", bbox_inches="tight")
    plt.close()


def plot_feature_importance():
    print("  feature importance...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for idx, model_name in enumerate(["random_forest", "xgboost"]):
        ax = axes[idx]
        path = RESULTS_DIR / f"feature_importance_{model_name}_entity.csv"
        if not path.exists():
            path = RESULTS_DIR / f"feature_importance_{model_name}_random.csv"
        if not path.exists():
            continue
        imp = pd.read_csv(path).head(15)
        ax.barh(range(len(imp)), imp["importance"].values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp))))
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15: {model_name.replace('_', ' ').title()}")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_feature_importance.png", bbox_inches="tight")
    plt.close()


def plot_roc_curves():
    print("  ROC curves...")
    fig, axes = plt.subplots(1, len(SPLIT_STRATEGIES), figsize=(6 * len(SPLIT_STRATEGIES), 5))
    model_files = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "Isolation Forest": "isolation_forest",
    }
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    for col, strategy in enumerate(SPLIT_STRATEGIES):
        ax = axes[col]
        test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
        cols = feature_columns(test)
        X_test = clean_matrix(test[cols].values)
        y_test = test["is_bot"].values.astype(int)
        scaler_path = MODELS_DIR / f"scaler_{strategy}.joblib"
        if not scaler_path.exists():
            continue
        X_test_s = joblib.load(scaler_path).transform(X_test)
        for (name, slug), color in zip(model_files.items(), colors):
            path = MODELS_DIR / f"{slug}_{strategy}.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)
            if name == "Isolation Forest":
                scores = -model.decision_function(X_test_s)
                smin, smax = scores.min(), scores.max()
                y_prob = (scores - smin) / (smax - smin) if smax > smin else np.full_like(scores, 0.5)
            else:
                y_prob = model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC: {strategy.capitalize()} Split")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_roc_curves.png", bbox_inches="tight")
    plt.close()


def plot_leakage_delta():
    print("  leakage delta...")
    results = pd.read_csv(RESULTS_DIR / "all_results.csv")
    models = list(results["model"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    temporal_d, entity_d = [], []
    for model_name in models:
        m = results[results["model"] == model_name]
        r = m[m["split_strategy"] == "random"]["f1_score"].values
        t = m[m["split_strategy"] == "temporal"]["f1_score"].values
        e = m[m["split_strategy"] == "entity"]["f1_score"].values
        temporal_d.append((r[0] - t[0]) if len(r) and len(t) else 0)
        entity_d.append((r[0] - e[0]) if len(r) and len(e) else 0)
    bars1 = ax.bar(x - width / 2, temporal_d, width, label="Random − Temporal", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, entity_d, width, label="Random − Entity", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("F1 Inflation (Δ)")
    ax.set_title("Data Leakage Impact on F1-Score")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:+.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_leakage_delta.png", bbox_inches="tight")
    plt.close()


def run_shap_analysis():
    print("  SHAP analysis...")
    # Prefer Random Forest for SHAP: newer XGBoost base_score format can break TreeExplainer
    strategy = "entity"
    candidates = ["random_forest", "xgboost"]
    model_name = None
    model_path = None
    for name in candidates:
        path = MODELS_DIR / f"{name}_{strategy}.joblib"
        if path.exists():
            model_name, model_path = name, path
            break
    if model_path is None:
        print("    No tree model found; skipping SHAP.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / f"scaler_{strategy}.joblib")
    test = pd.read_parquet(SPLITS_DIR / f"{strategy}_test.parquet")
    cols = feature_columns(test)
    X_test = scaler.transform(clean_matrix(test[cols].values))
    n_shap = min(1000, len(X_test))
    idx = np.random.RandomState(42).choice(len(X_test), n_shap, replace=False)
    X_shap = X_test[idx]
    # Show real feature values in the plot colours, while explaining the scaled
    # matrix that was actually supplied to the trained model.
    X_display = pd.DataFrame(
        clean_matrix(test.iloc[idx][cols].values),
        columns=cols,
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
    except Exception as e:
        print(f"    TreeExplainer failed ({e}); using permutation Explainer fallback...")
        explainer = shap.Explainer(model.predict, X_shap[:100])
        explanation = explainer(X_shap)
        shap_values = explanation.values

    # A 3-D classifier output is otherwise mistaken by summary_plot for SHAP
    # interaction values, producing a very tall, duplicated two-column figure.
    shap_values = select_positive_class_shap(shap_values, n_shap, len(cols))

    shap.summary_plot(
        shap_values,
        X_display,
        feature_names=cols,
        show=False,
        max_display=20,
        plot_size=(12, 9),
        alpha=0.65,
    )
    ax = plt.gca()
    ax.set_title(
        f"SHAP Summary ({model_name.replace('_', ' ').title()}, Entity Split)",
        pad=16,
        fontweight="bold",
    )
    ax.set_xlabel("SHAP value (impact on bot prediction)")
    ax.grid(axis="x", alpha=0.2)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.31, right=0.96, top=0.91, bottom=0.10)
    fig.savefig(FIGURES_DIR / "fig_shap_summary.png", bbox_inches="tight", facecolor="white")
    plt.close()

    shap.summary_plot(
        shap_values,
        X_display,
        feature_names=cols,
        plot_type="bar",
        show=False,
        max_display=20,
        plot_size=(11, 9),
        color="#2878B5",
    )
    ax = plt.gca()
    ax.set_title(
        f"Mean |SHAP| ({model_name.replace('_', ' ').title()}, Entity Split)",
        pad=16,
        fontweight="bold",
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.2)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.34, right=0.96, top=0.91, bottom=0.10)
    fig.savefig(FIGURES_DIR / "fig_shap_bar.png", bbox_inches="tight", facecolor="white")
    plt.close()

    vals = np.abs(shap_values)
    shap_imp = pd.DataFrame({
        "feature": cols,
        "mean_abs_shap": vals.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    print("    Top SHAP features:")
    for _, row in shap_imp.head(10).iterrows():
        print(f"      {row['feature']}: {row['mean_abs_shap']:.4f}")


def plot_pipeline_architecture():
    """Block diagram of the five-layer detection framework (Figure 4.1)."""
    print("  pipeline architecture...")
    layers = [
        ("Layer 1 - Log Ingestion & Parsing",
         "Streaming JSON reader | field extraction | type coercion | UTC normalisation",
         "#34495e"),
        ("Layer 2 - Ground-Truth Labelling",
         "User-Agent bot heuristic | authentication-surface mapping | 3-class assignment",
         "#2980b9"),
        ("Layer 3 - Sessionization & Feature Engineering",
         "(IP, User-Agent) sessions, 30-min timeout | 49 behavioural features",
         "#16a085"),
        ("Layer 4 - Leakage-Aware Partitioning",
         "Random (baseline) | Temporal | Entity-based (by IP) | train-only scaling",
         "#e67e22"),
        ("Layer 5 - Learning, Evaluation & Explanation",
         "LR | RF | XGBoost | Isolation Forest -> metrics, statistics, SHAP",
         "#c0392b"),
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) * 2 + 1)
    ax.axis("off")

    for i, (title, detail, color) in enumerate(layers):
        y = (len(layers) - i - 1) * 2 + 1
        ax.add_patch(plt.Rectangle((0.6, y), 8.8, 1.4, facecolor=color, alpha=0.90,
                                   edgecolor="black", linewidth=1.2, zorder=2))
        ax.text(5.0, y + 0.95, title, ha="center", va="center",
                color="white", fontsize=12.5, fontweight="bold", zorder=3)
        ax.text(5.0, y + 0.40, detail, ha="center", va="center",
                color="white", fontsize=9.5, zorder=3)
        if i < len(layers) - 1:
            ax.annotate("", xy=(5.0, y - 0.55), xytext=(5.0, y - 0.05),
                        arrowprops=dict(arrowstyle="-|>", linewidth=2, color="#2c3e50"))

    ax.text(5.0, len(layers) * 2 + 0.5, "Web Server Access Logs (public_v2.json)",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.4, "Detection Results, Leakage Analysis & Feature Attributions",
            ha="center", va="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_architecture.png", bbox_inches="tight")
    plt.close()


def plot_three_class_overview():
    """Class balance and per-class detection quality for the 3-class task."""
    mc_path = RESULTS_DIR / "multiclass_per_class.csv"
    if not SESSION_FEATURES.exists() or not mc_path.exists():
        return
    print("  three-class overview...")
    sf = pd.read_parquet(SESSION_FEATURES, columns=["traffic_class"])
    counts = sf["traffic_class"].value_counts().reindex([0, 1, 2], fill_value=0)
    names = ["Human", "Bot", "Credential\nStuffing"]
    colors = ["#2ecc71", "#e67e22", "#c0392b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    bars = ax1.bar(names, counts.values, color=colors, edgecolor="black", linewidth=0.7)
    ax1.set_yscale("log")
    ax1.set_ylabel("Sessions (log scale)")
    ax1.set_title("Three-Class Session Distribution")
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.15,
                 f"{int(v):,}\n({v / counts.sum() * 100:.2f}%)",
                 ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(top=counts.max() * 4)

    per_class = pd.read_csv(mc_path)
    sub = per_class[(per_class["split_strategy"] == "entity")]
    models = sub["model"].unique()
    x = np.arange(len(models))
    width = 0.25
    for i, (metric, label) in enumerate(
        [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1")]
    ):
        vals = [
            sub[(sub["model"] == m) & (sub["class"] == "Credential Stuffing")][metric].iloc[0]
            for m in models
        ]
        bars = ax2.bar(x + i * width, vals, width, label=label)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Score")
    ax2.set_title("Credential-Stuffing Class Detection (Entity Split)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_three_class_overview.png", bbox_inches="tight")
    plt.close()


def plot_multiclass_confusion():
    path = RESULTS_DIR / "multiclass_confusion.json"
    if not path.exists():
        return
    print("  multiclass confusion matrices...")
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [k for k in data if k.endswith("|entity")]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    labels = ["Human", "Bot", "Stuffing"]
    for ax, key in zip(axes, keys):
        cm = np.array(data[key])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(key.split("|")[0] + " (entity split)", fontsize=11)
    plt.suptitle("Three-Class Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_multiclass_confusion.png", bbox_inches="tight")
    plt.close()


def plot_ablation():
    path = RESULTS_DIR / "ablation_results.csv"
    if not path.exists():
        return
    print("  feature-set ablation...")
    df = pd.read_csv(path)
    order = ["Request-level", "Session-level", "Combined", "Combined + UA (circular)"]
    order = [o for o in order if o in df["feature_set"].unique()]

    fig, axes = plt.subplots(1, len(SPLIT_STRATEGIES), figsize=(6.4 * len(SPLIT_STRATEGIES), 5.5), sharey=True)
    colors = ["#95a5a6", "#3498db", "#16a085", "#c0392b"]
    for ax, strategy in zip(axes, SPLIT_STRATEGIES):
        sub = df[df["split_strategy"] == strategy]
        models = [m for m in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
                  if m in sub["model"].unique()]
        x = np.arange(len(models))
        width = 0.8 / max(len(order), 1)
        for i, fs in enumerate(order):
            vals = []
            for m in models:
                row = sub[(sub["model"] == m) & (sub["feature_set"] == fs)]["f1_score"]
                vals.append(float(row.iloc[0]) if len(row) else 0.0)
            bars = ax.bar(x + i * width, vals, width, label=fs, color=colors[i % len(colors)])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.set_xticks(x + width * (len(order) - 1) / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(f"{strategy.capitalize()} split")
        ax.set_ylim(0, 1.22)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1-Score")
    axes[-1].legend(loc="lower right", fontsize=9)
    plt.suptitle("Feature-Set Ablation: contribution of each feature family", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_ablation.png", bbox_inches="tight")
    plt.close()


def plot_bootstrap_ci():
    path = RESULTS_DIR / "bootstrap_ci.csv"
    if not path.exists():
        return
    print("  bootstrap confidence intervals...")
    df = pd.read_csv(path)
    models = [m for m in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Isolation Forest"]
              if m in df["model"].unique()]
    colors = {"random": "#3498db", "temporal": "#e74c3c", "entity": "#2ecc71", "ua_entity": "#f39c12"}

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = {"random": -0.3, "temporal": -0.1, "entity": 0.1, "ua_entity": 0.3}
    for strategy, off in offsets.items():
        sub = df[df["split_strategy"] == strategy].set_index("model")
        xs, ys, errs = [], [], []
        for i, m in enumerate(models):
            if m not in sub.index:
                continue
            row = sub.loc[m]
            xs.append(i + off)
            ys.append(row["f1_mean"])
            errs.append([row["f1_mean"] - row["f1_ci_low"], row["f1_ci_high"] - row["f1_mean"]])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=np.array(errs).T, fmt="o", capsize=5, capthick=1.6,
                    markersize=7, linewidth=1.6, color=colors[strategy],
                    label=strategy.capitalize())
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("F1-Score (bootstrap mean, 95% CI)")
    ax.set_title("Bootstrap 95% Confidence Intervals on the Test Partition (1000 resamples)")
    ax.legend(title="Split strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_bootstrap_ci.png", bbox_inches="tight")
    plt.close()


def plot_circularity():
    path = RESULTS_DIR / "circularity_inflation.csv"
    if not path.exists():
        return
    print("  circularity inflation...")
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(11, 6))
    models = df["model"].unique()
    x = np.arange(len(models))
    n_split = len(SPLIT_STRATEGIES)
    width = 0.8 / n_split
    for i, strategy in enumerate(SPLIT_STRATEGIES):
        vals = []
        for m in models:
            row = df[(df["model"] == m) & (df["split_strategy"] == strategy)]["inflation"]
            vals.append(float(row.iloc[0]) if len(row) else 0.0)
        bars = ax.bar(x + i * width, vals, width, label=strategy.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0015, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (n_split - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("F1 inflation from label-correlated features")
    ax.set_title("Circularity Effect: adding User-Agent descriptors that encode the label")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylim(top=df["inflation"].max() * 1.18)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_circularity.png", bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    print("=" * 70)
    print("STEP 5: SHAP Analysis & Visualization")
    print("=" * 70)

    plot_pipeline_architecture()
    if DATASET_SUMMARY.exists():
        plot_dataset_overview()
    if SESSION_FEATURES.exists():
        plot_session_distributions()
        plot_three_class_overview()
    if (RESULTS_DIR / "all_results.csv").exists():
        plot_comparative_performance()
        plot_confusion_matrices()
        plot_leakage_delta()
    plot_multiclass_confusion()
    plot_ablation()
    plot_circularity()
    plot_bootstrap_ci()
    plot_feature_importance()
    plot_roc_curves()
    run_shap_analysis()

    print("\nGENERATED FIGURES:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    print(f"\n[OK] Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
