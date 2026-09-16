"""Create the compact three-model artifact pack for the Medium article."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "outputs" / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
ARTIFACT_DIR = PAPER_DIR / "article_artifacts"
MODEL_ORDER = ["Martingale", "Quantile", "LSTM_HuPin"]
DISPLAY_NAMES = {
    "Martingale": "Martingale",
    "Quantile": "Median Quantile",
    "LSTM_HuPin": "HuPin LSTM",
}
COLORS = {
    "Martingale": "#6B7280",
    "Quantile": "#3B82F6",
    "LSTM_HuPin": "#D97706",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_tables(micro: pd.DataFrame, metrics: pd.DataFrame) -> None:
    summary = micro.reset_index().copy()
    summary["Display model"] = summary["Model"].map(DISPLAY_NAMES)
    summary.to_csv(PAPER_DIR / "article_model_summary.csv", index=False)

    central = summary[["Display model", "RMSE", "R²", "N"]].copy()
    central.columns = ["Model", "RMSE", "test R²", "N"]
    bmr_table = summary[
        [
            "Display model", "BearMissRate", "BearMissRate_tol",
            "BearMissRate_z", "bear-miss", "bear-miss_tol",
            "bearish-observations", "z-miss", "z-bears",
        ]
    ].copy()
    bmr_table.columns = [
        "Model", "BMR", "BMR_tol", "BMR_Z", "BMR misses",
        "BMR_tol misses", "bearish N", "BMR_Z misses", "tail N",
    ]
    ticker_table = metrics.copy()
    ticker_table["Model"] = ticker_table["Model"].map(DISPLAY_NAMES)

    central.to_csv(ARTIFACT_DIR / "central_metrics.csv", index=False)
    bmr_table.to_csv(ARTIFACT_DIR / "bmr_metrics.csv", index=False)
    ticker_table.to_csv(ARTIFACT_DIR / "all_metrics_by_ticker.csv", index=False)

    rendered = pd.DataFrame(
        {
            "Model": summary["Display model"],
            "RMSE": summary["RMSE"].map(lambda value: f"{value:.6f}"),
            "test R²": summary["R²"].map(lambda value: f"{value:.4f}"),
            "BMR": summary["BearMissRate"].map(lambda value: f"{100 * value:.1f}%"),
            "BMR_tol": summary["BearMissRate_tol"].map(lambda value: f"{100 * value:.1f}%"),
            "BMR_Z": summary["BearMissRate_z"].map(lambda value: f"{100 * value:.1f}%"),
        }
    )
    note = (
        "# Copy-ready article results\n\n"
        + _markdown_table(rendered)
        + "\n\n"
        + "Common test panel: 1,575 security-days (315 dates for each of five "
        "UK-listed energy securities), 2024-04-22 to 2025-07-18. For the article, BMR_tol uses a "
        "pre-specified one-percentage-point materiality threshold (epsilon = "
        "0.01), replacing the MSc model-dependent validation-RMSE rule. "
        "BMR_Z uses one ticker-specific, train-only Z = 2 threshold shared "
        "by all models. The BMR family is interpreted for a long position.\n"
    )
    (ARTIFACT_DIR / "results_table.md").write_text(note, encoding="utf-8")


def _central_figure(micro: pd.DataFrame) -> None:
    labels = [DISPLAY_NAMES[model] for model in MODEL_ORDER]
    colors = [COLORS[model] for model in MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))

    rmse_values = micro["RMSE"]
    axes[0].bar(labels, rmse_values, color=colors)
    axes[0].set_title("Average forecast error")
    axes[0].set_ylabel("Test RMSE")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylim(0.0, float(rmse_values.max()) * 1.15)
    for index, value in enumerate(rmse_values):
        axes[0].text(index, value + 0.00035, f"{value:.6f}", ha="center", fontsize=10)

    r2_values = micro["R²"]
    axes[1].bar(labels, r2_values, color=colors)
    axes[1].axhline(0, color="#374151", linestyle="--", linewidth=1.3)
    axes[1].set_title("Out-of-sample explanatory power")
    axes[1].set_ylabel("Test R²")
    axes[1].tick_params(axis="x", rotation=18)
    for index, value in enumerate(r2_values):
        axes[1].text(index, value - 0.00018, f"{value:.4f}", ha="center", va="top", fontsize=10)

    fig.suptitle("Similar average forecast scores across the three models", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "central_metrics.png", dpi=200, bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "central_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bmr_figure(micro: pd.DataFrame) -> None:
    labels = [DISPLAY_NAMES[model] for model in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.24
    series = [
        ("BMR", "BearMissRate", "#2563EB"),
        (r"BMR$_{tol}$", "BearMissRate_tol", "#D97706"),
        (r"BMR$_Z$", "BearMissRate_z", "#7C3AED"),
    ]
    fig, axis = plt.subplots(figsize=(12.5, 6.2))
    for offset, (label, column, color) in enumerate(series):
        values = 100 * micro[column].to_numpy()
        bars = axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 110)
    axis.set_ylabel("Miss rate conditional on realised downside (%)")
    fig.suptitle(
        "Downside-conditioned diagnostics reveal a different error pattern",
        fontsize=18,
        y=0.98,
    )
    axis.legend(ncols=1, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axis.text(
        0.01,
        -0.17,
        "BMR: any optimistic loss-day miss. BMR_tol: miss >= 1pp. "
        "BMR_Z: missed train-defined severe-loss state. Lower is better.",
        transform=axis.transAxes,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.20, top=0.82)
    fig.savefig(FIGURE_DIR / "bmr_family.png", dpi=200, bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "bmr_family.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ticker_r2_figure(metrics: pd.DataFrame) -> None:
    per_ticker = metrics[metrics["Ticker"] != "All"].copy()
    per_ticker["Ticker"] = per_ticker["Ticker"].astype(int)
    heatmap = per_ticker.pivot(index="Model", columns="Ticker", values="R²")
    heatmap = heatmap.loc[MODEL_ORDER]
    heatmap.columns = ["BSIF", "HTG", "SSE", "TLW", "UKW"]
    heatmap.index = [DISPLAY_NAMES[model] for model in heatmap.index]
    limit = max(0.05, float(np.nanmax(np.abs(heatmap.to_numpy()))))
    fig, axis = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.5,
        cbar_kws={"label": "Test R²"},
        ax=axis,
    )
    axis.set_title("The pooled central result is not uniform across securities")
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "per_ticker_r2_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(ARTIFACT_DIR / "per_ticker_r2_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(PAPER_DIR / "common_panel_metrics.csv")
    micro = metrics[metrics["Ticker"] == "All"].set_index("Model")
    if set(micro.index) != set(MODEL_ORDER):
        raise ValueError(f"Expected article models {MODEL_ORDER}; found {list(micro.index)}")
    micro = micro.loc[MODEL_ORDER].copy()
    zero_rmse = float(micro.loc["Martingale", "RMSE"])
    micro["RMSE_improvement_vs_zero_pct"] = 100 * (1 - micro["RMSE"] / zero_rmse)

    _write_tables(micro, metrics)
    _central_figure(micro)
    _bmr_figure(micro)
    _ticker_r2_figure(metrics)

    manifest = {
        "purpose": "Medium-ready three-model evidence pack",
        "models": [DISPLAY_NAMES[model] for model in MODEL_ORDER],
        "metrics": ["RMSE", "test R²", "BMR", "BMR_tol", "BMR_Z"],
        "files": {
            "central_metrics.csv": "Pooled conventional metrics",
            "bmr_metrics.csv": "Pooled BMR values and count denominators",
            "all_metrics_by_ticker.csv": "Full pooled and per-ticker audit table",
            "results_table.md": "Copy-ready Markdown results table",
            "central_metrics.png": "Conventional-metric figure",
            "bmr_family.png": "Main BMR-family figure",
            "per_ticker_r2_heatmap.png": "Per-security diagnostic figure",
            "README.md": "Artifact guide and provenance",
        },
    }
    artifact_readme = """# Three-model Medium artifact pack

This folder is generated by `python paper/make_article_figures.py` from the
verified common-panel output. It contains only Martingale, median Quantile,
and HuPin LSTM, evaluated with RMSE, test R², BMR, BMR_tol, and BMR_Z.
The article uses common epsilon = 0.01 for every model and security.

- `results_table.md`: copy-ready article table.
- `central_metrics.csv` and `bmr_metrics.csv`: pooled results.
- `all_metrics_by_ticker.csv`: pooled and per-ticker audit rows.
- `central_metrics.png` and `bmr_family.png`: main article figures.
- `per_ticker_r2_heatmap.png`: optional supporting figure.
- `../verification_summary.json`: automated reproduction checks.
- `../../../paper/DATA_PROVENANCE.md`: input sources and redistribution boundary.
"""
    (ARTIFACT_DIR / "README.md").write_text(artifact_readme, encoding="utf-8")
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote three-model article artifact pack to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
