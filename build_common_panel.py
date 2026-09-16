"""Build the article common panel with the agreed BMR methodology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import robust_bmr_z_threshold
from src.paper_evaluation import (
    build_common_panel,
    canonical_truth,
    common_panel_wide,
    evaluate_common_panel,
    load_default_predictions,
)


DATA_DIR = Path("outputs/paper/data")
PAPER_DIR = Path("outputs/paper")
TICKERS = [0, 1, 2, 3, 4]
MODELS = ["LSTM_HuPin", "Quantile", "Martingale"]

# Agreed article definitions. EPS_ERR is a user-defined materiality threshold
# in daily log-return units; it is not estimated from any model or data split.
EPS_ERR = 0.01
Z_MULTIPLIER = 2.0
Z_FALLBACK_QUANTILE = 0.0228


def common_metric_parameters(train: pd.DataFrame) -> pd.DataFrame:
    """Return model-independent BMR parameters fixed before test evaluation.

    One downside-tail threshold is estimated per ticker from the full corrected
    pre-test training sample and reused for every model. A non-negative
    threshold does not define a downside tail and is therefore recorded as
    ``NaN`` with an explicit reason.
    """

    required = {"ticker_id", "Date", "log_return"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Training data are missing columns: {sorted(missing)}")

    threshold_rows: dict[int, dict[str, object]] = {}
    ordered = train.sort_values(["ticker_id", "Date"])
    for ticker, group in ordered.groupby("ticker_id", sort=True):
        values = pd.to_numeric(group["log_return"], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            threshold = float("nan")
            reason = "training sample is empty"
        else:
            candidate = robust_bmr_z_threshold(
                values,
                mad_multiplier=Z_MULTIPLIER,
                fallback_quantile=Z_FALLBACK_QUANTILE,
            )
            if not np.isfinite(candidate):
                threshold = float("nan")
                reason = "training sample does not define a finite tail threshold"
            elif candidate >= 0.0:
                threshold = float("nan")
                reason = "training sample does not define a negative tail"
            else:
                threshold = float(candidate)
                reason = ""
        threshold_rows[int(ticker)] = {
            "z-threshold": threshold,
            "threshold_reason": reason,
            "z_reference_n": int(values.size),
        }

    if sorted(threshold_rows) != TICKERS:
        raise ValueError(f"Expected tickers {TICKERS}; found {sorted(threshold_rows)}")

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for ticker in TICKERS:
            rows.append(
                {
                    "Model": model,
                    "Ticker": str(ticker),
                    "eps_err": EPS_ERR,
                    **threshold_rows[ticker],
                    "eps_source": "user-defined constant in daily log-return units",
                    "z_source": "full corrected pre-test train; ticker-specific and model-independent; Z=2",
                }
            )
        rows.append(
            {
                "Model": model,
                "Ticker": "All",
                "eps_err": EPS_ERR,
                "z-threshold": np.nan,
                "threshold_reason": "pooled from ticker-specific BMR_Z counts",
                "z_reference_n": int(sum(row["z_reference_n"] for row in threshold_rows.values())),
                "eps_source": "user-defined constant in daily log-return units",
                "z_source": "micro-pooled from the common ticker-specific train-only thresholds",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA_DIR / "df_LSTM_test.csv", parse_dates=["Date"])
    train = pd.read_csv(DATA_DIR / "df_LSTM_train.csv", parse_dates=["Date"])
    truth = canonical_truth(test)
    panel, coverage = build_common_panel(truth, load_default_predictions())
    parameters = common_metric_parameters(train)
    metrics = evaluate_common_panel(panel, parameters)
    wide = common_panel_wide(panel)

    panel.to_csv(PAPER_DIR / "common_panel_long.csv", index=False)
    wide.to_csv(PAPER_DIR / "common_panel_wide.csv", index=False)
    coverage.to_csv(PAPER_DIR / "common_panel_coverage.csv", index=False)
    parameters.to_csv(PAPER_DIR / "metric_parameters.csv", index=False)
    metrics.to_csv(PAPER_DIR / "common_panel_metrics.csv", index=False)
    print(
        f"Common panel: {len(wide)} observations, {panel['model'].nunique()} models "
        f"(including Martingale); epsilon={EPS_ERR:.4f}."
    )
    print(f"Wrote agreed BMR outputs to {PAPER_DIR.resolve()}")


if __name__ == "__main__":
    main()
