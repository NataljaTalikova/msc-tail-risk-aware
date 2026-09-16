"""Fail-fast verification for the corrected article artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_evaluation import DEFAULT_PREDICTION_PATHS, canonical_predictions, canonical_truth  # noqa: E402
from src.metrics import robust_bmr_z_threshold  # noqa: E402


EXPECTED_TUNING_ROWS = {
    "hupin": 48,
    "quantile_joint": 4,
}
EXPECTED_TICKERS = 5
EXPECTED_TEST_ROWS_PER_TICKER = 315
EXPECTED_MODELS = {"Martingale", "Quantile", "LSTM_HuPin"}
PAPER_DIR = ROOT / "outputs" / "paper"


def _assert_equal_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    columns = ["ticker_id", "target_date"]
    left_keys = left[columns].sort_values(columns).reset_index(drop=True)
    right_keys = right[columns].sort_values(columns).reset_index(drop=True)
    if not left_keys.equals(right_keys):
        missing = left_keys.merge(right_keys, on=columns, how="left", indicator=True)
        missing = missing.loc[missing["_merge"] == "left_only", columns]
        raise AssertionError(f"{label} key mismatch; missing examples: {missing.head().to_dict('records')}")


def main() -> None:
    train = pd.read_csv(PAPER_DIR / "data" / "df_LSTM_train.csv", parse_dates=["Date"])
    context = pd.read_csv(PAPER_DIR / "data" / "df_LSTM_context.csv", parse_dates=["Date"])
    test = pd.read_csv(PAPER_DIR / "data" / "df_LSTM_test.csv", parse_dates=["Date"])
    for label, frame in {"train": train, "context": context, "test": test}.items():
        if frame.isna().any().any():
            raise AssertionError(f"{label} contains missing values")
        if frame.duplicated(["ticker_id", "Date"]).any():
            raise AssertionError(f"{label} contains duplicate ticker/date keys")
        if frame["ticker_id"].nunique() != EXPECTED_TICKERS:
            raise AssertionError(f"{label} does not contain five tickers")

    for ticker in sorted(train["ticker_id"].unique()):
        train_dates = train.loc[train["ticker_id"] == ticker, "Date"]
        context_dates = context.loc[context["ticker_id"] == ticker, "Date"]
        test_dates = test.loc[test["ticker_id"] == ticker, "Date"]
        if context_dates.empty:
            raise AssertionError(f"Ticker {ticker} has no outer-purge forecast context")
        if not train_dates.max() < context_dates.min() <= context_dates.max() < test_dates.min():
            raise AssertionError(f"Ticker {ticker} fit/context/test dates are not ordered")

    per_ticker_test = test.groupby("ticker_id").size()
    if not (per_ticker_test == EXPECTED_TEST_ROWS_PER_TICKER).all():
        raise AssertionError(f"Unexpected corrected test counts: {per_ticker_test.to_dict()}")

    truth = canonical_truth(test)
    prediction_summary: dict[str, object] = {}
    for model, candidates in DEFAULT_PREDICTION_PATHS.items():
        existing = next((ROOT / path for path in candidates if (ROOT / path).exists()), None)
        if existing is None:
            raise AssertionError(f"Missing final predictions for {model}: {list(candidates)}")
        raw = pd.read_csv(existing) if existing.suffix == ".csv" else pd.read_parquet(existing)
        predictions = canonical_predictions(raw, model)
        _assert_equal_keys(truth, predictions, model)
        prediction_summary[model] = {
            "rows": int(len(predictions)),
            "first_date": str(predictions["target_date"].min().date()),
            "last_date": str(predictions["target_date"].max().date()),
        }

    tuning_summary: dict[str, object] = {}
    for model, expected in EXPECTED_TUNING_ROWS.items():
        path = PAPER_DIR / model / "tuning_results.csv"
        frame = pd.read_csv(path)
        successful = frame.loc[frame["status"] == "ok", "config_key"].nunique()
        failed = int((frame["status"] == "failed").sum())
        if successful != expected or failed:
            raise AssertionError(
                f"{model} tuning incomplete: successful={successful}/{expected}, failed={failed}"
            )
        tuning_summary[model] = {"successful": int(successful), "failed": failed}

    config_paths = [
        PAPER_DIR / "hupin" / "best_config.json",
        PAPER_DIR / "quantile_joint" / "best_config.json",
    ]
    for path in config_paths:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)

    metrics = pd.read_csv(PAPER_DIR / "common_panel_metrics.csv")
    panel = pd.read_csv(PAPER_DIR / "common_panel_long.csv")
    if set(metrics["Model"]) != EXPECTED_MODELS or set(panel["model"]) != EXPECTED_MODELS:
        raise AssertionError("Article outputs do not contain exactly the three agreed models")
    original_metric_columns = {
        "BearMissRate", "BearMissRate_tol", "BearMissRate_z",
        "z-miss", "z-bears", "z-threshold",
    }
    if not original_metric_columns.issubset(metrics.columns):
        raise AssertionError("Original BMR-family columns are missing")
    forbidden_replacements = {"DUR", "DUR_tol", "tail_miss_rate"}
    if forbidden_replacements & set(metrics.columns):
        raise AssertionError("Renamed replacement metrics remain in the output")
    micro = metrics[metrics["Ticker"] == "All"]
    if micro["N"].nunique() != 1 or int(micro["N"].iloc[0]) != len(truth):
        raise AssertionError("Models do not have equal canonical evaluation counts")

    parameters = pd.read_csv(PAPER_DIR / "metric_parameters.csv")
    expected_models = {*DEFAULT_PREDICTION_PATHS, "Martingale"}
    if set(parameters["Model"]) != expected_models:
        raise AssertionError("Metric parameter table has an unexpected model set")
    if parameters.duplicated(["Model", "Ticker"]).any():
        raise AssertionError("Metric parameter rows are not unique")
    if not np.allclose(parameters["eps_err"], 0.01, rtol=0.0, atol=0.0):
        raise AssertionError("BMR_tol does not use the agreed epsilon=0.01")
    ticker_parameters = parameters[parameters["Ticker"].astype(str) != "All"].copy()
    for ticker, group in ticker_parameters.groupby("Ticker"):
        if group["z-threshold"].nunique(dropna=False) != 1:
            raise AssertionError(f"Ticker {ticker} has model-specific BMR_Z thresholds")
        threshold = float(group["z-threshold"].iloc[0])
        if not np.isfinite(threshold) or threshold >= 0.0:
            raise AssertionError(f"Ticker {ticker} has no valid negative training threshold")
        train_values = (
            train.loc[train["ticker_id"] == int(ticker)]
            .sort_values("Date")["log_return"]
            .to_numpy()
        )
        expected_threshold = robust_bmr_z_threshold(train_values)
        if not np.isclose(threshold, expected_threshold, rtol=0.0, atol=1e-15):
            raise AssertionError(f"Ticker {ticker} threshold is not full-train Z=2")

    count_columns = {
        "bear-miss", "bear-miss_tol", "bearish-observations",
        "z-miss", "z-bears", "undefined_reason",
    }
    if not count_columns.issubset(metrics.columns):
        raise AssertionError("Metric numerator/denominator audit columns are missing")
    for model, group in metrics.groupby("Model"):
        all_row = group[group["Ticker"] == "All"].iloc[0]
        ticker_rows = group[group["Ticker"] != "All"]
        for count in ["bear-miss", "bear-miss_tol", "bearish-observations", "z-miss", "z-bears"]:
            if int(all_row[count]) != int(ticker_rows[count].sum()):
                raise AssertionError(f"{model} ALL {count} is not micro-pooled")
        if not np.isclose(
            float(all_row["BearMissRate_tol"]),
            float(all_row["bear-miss_tol"]) / float(all_row["bearish-observations"]),
        ):
            raise AssertionError(f"{model} BMR_tol does not match its counts")

    # Recalculate all five reported metrics directly from the saved common
    # panel. This is independent of the evaluator's metric helper functions.
    parameter_lookup = parameters.set_index(["Model", "Ticker"])
    for _, row in metrics.iterrows():
        ticker = str(row["Ticker"])
        selected = panel[panel["model"] == row["Model"]]
        if ticker != "All":
            selected = selected[selected["ticker_id"] == int(ticker)]
        actual = selected["y_true"].to_numpy()
        prediction = selected["y_pred"].to_numpy()
        error = actual - prediction
        expected_rmse = float(np.sqrt(np.mean(error ** 2)))
        expected_r2 = float(1.0 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2))
        if not np.isclose(float(row["RMSE"]), expected_rmse, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{row['Model']}/{ticker} RMSE did not reproduce")
        if not np.isclose(float(row["R²"]), expected_r2, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{row['Model']}/{ticker} R² did not reproduce")

        bearish = actual < 0.0
        bearish_n = int(bearish.sum())
        bmr_misses = int((bearish & (prediction > actual)).sum())
        tol_misses = int((bearish & ((prediction - actual) >= 0.01)).sum())
        expected_bmr = bmr_misses / bearish_n if bearish_n else np.nan
        expected_tol = tol_misses / bearish_n if bearish_n else np.nan
        if bmr_misses != int(row["bear-miss"]) or tol_misses != int(row["bear-miss_tol"]):
            raise AssertionError(f"{row['Model']}/{ticker} BMR numerator did not reproduce")
        if bearish_n != int(row["bearish-observations"]):
            raise AssertionError(f"{row['Model']}/{ticker} bearish denominator did not reproduce")
        if not np.isclose(float(row["BearMissRate"]), expected_bmr, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{row['Model']}/{ticker} BMR did not reproduce")
        if not np.isclose(float(row["BearMissRate_tol"]), expected_tol, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{row['Model']}/{ticker} BMR_tol did not reproduce")

        z_misses = 0
        z_bears = 0
        ticker_groups = selected.groupby("ticker_id") if ticker == "All" else [(int(ticker), selected)]
        for ticker_id, ticker_group in ticker_groups:
            threshold = float(parameter_lookup.loc[(row["Model"], str(int(ticker_id))), "z-threshold"])
            ticker_actual = ticker_group["y_true"].to_numpy()
            ticker_prediction = ticker_group["y_pred"].to_numpy()
            tail = ticker_actual <= threshold
            z_bears += int(tail.sum())
            z_misses += int((tail & (ticker_prediction > threshold)).sum())
        expected_z = z_misses / z_bears if z_bears else np.nan
        if z_misses != int(row["z-miss"]) or z_bears != int(row["z-bears"]):
            raise AssertionError(f"{row['Model']}/{ticker} BMR_Z counts did not reproduce")
        if not np.isclose(float(row["BearMissRate_z"]), expected_z, rtol=0.0, atol=1e-15):
            raise AssertionError(f"{row['Model']}/{ticker} BMR_Z did not reproduce")

    summary = {
        "status": "passed",
        "train_rows": int(len(train)),
        "forecast_context_rows": int(len(context)),
        "test_rows": int(len(test)),
        "prediction_outputs": prediction_summary,
        "tuning": tuning_summary,
        "bmr_tolerance": 0.01,
        "bmr_z_thresholds": {
            str(ticker): float(group["z-threshold"].iloc[0])
            for ticker, group in ticker_parameters.groupby("Ticker")
        },
        "common_panel_rows_per_model": int(micro["N"].iloc[0]),
        "metric_recalculation": "all five metrics reproduced directly from common_panel_long.csv",
    }
    output = PAPER_DIR / "verification_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
