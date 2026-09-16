"""Common-panel construction with the project's original BMR metrics.

The article panel aligns the three agreed illustrative models on identical
target dates and applies the article-approved, model-independent BMR
parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import bmr, bmr_tolerant, bmr_z, pooled_bmr_z, r_squared, rmse


KEY_COLUMNS = ["ticker_id", "target_date"]
TICKER_TO_ID = {"BSIF": 0, "HTG": 1, "SSE": 2, "TLW": 3, "UKW": 4}
DEFAULT_PREDICTION_PATHS: dict[str, tuple[Path, ...]] = {
    "LSTM_HuPin": (Path("outputs/paper/hupin/final_test_predictions.csv"), Path("outputs/paper/hupin/final_test_predictions.parquet")),
    "Quantile": (Path("outputs/paper/quantile_joint/final_test_predictions.csv"), Path("outputs/paper/quantile_joint/final_test_predictions.parquet")),
}


def read_first_available(paths: Sequence[str | Path]) -> pd.DataFrame:
    attempted: list[str] = []
    parquet_error: Exception | None = None
    for candidate in paths:
        path = Path(candidate)
        attempted.append(str(path))
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            try:
                return pd.read_parquet(path)
            except ImportError as exc:
                parquet_error = exc
                continue
        raise ValueError(f"Unsupported prediction format: {path}")
    if parquet_error is not None:
        raise ImportError("A Parquet file exists but no Parquet engine is installed.") from parquet_error
    raise FileNotFoundError(f"No prediction file found. Tried: {attempted}")


def canonical_truth(
    dataframe: pd.DataFrame,
    *,
    ticker_column: str = "ticker_id",
    date_column: str = "Date",
    target_column: str = "log_return",
) -> pd.DataFrame:
    required = {ticker_column, date_column, target_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Canonical test data are missing columns: {sorted(missing)}")
    truth = dataframe[[ticker_column, date_column, target_column]].copy()
    truth.columns = ["ticker_id", "target_date", "y_true"]
    truth["ticker_id"] = pd.to_numeric(truth["ticker_id"], errors="raise").astype(int)
    truth["target_date"] = pd.to_datetime(truth["target_date"], errors="raise").dt.tz_localize(None).dt.normalize()
    truth["y_true"] = pd.to_numeric(truth["y_true"], errors="raise")
    truth = truth.dropna(subset=["y_true"])
    duplicates = truth.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = truth.loc[duplicates, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"Canonical truth has duplicate keys, for example: {examples}")
    return truth.sort_values(KEY_COLUMNS).reset_index(drop=True)


def canonical_predictions(dataframe: pd.DataFrame, model: str) -> pd.DataFrame:
    lookup = {column.lower(): column for column in dataframe.columns}
    date_column = lookup.get("target_date") or lookup.get("date")
    prediction_column = lookup.get("y_pred") or lookup.get("predicted") or lookup.get("prediction")
    ticker_column = lookup.get("ticker_id") or lookup.get("ticker")
    if date_column is None or prediction_column is None or ticker_column is None:
        raise ValueError(f"{model} predictions need ticker, date, and prediction columns; found {list(dataframe.columns)}")
    predictions = dataframe[[ticker_column, date_column, prediction_column]].copy()
    predictions.columns = ["ticker_id", "target_date", "y_pred"]
    if not pd.api.types.is_numeric_dtype(predictions["ticker_id"]):
        cleaned = predictions["ticker_id"].astype(str).str.replace(".L", "", regex=False)
        predictions["ticker_id"] = cleaned.map(TICKER_TO_ID)
    predictions["ticker_id"] = pd.to_numeric(predictions["ticker_id"], errors="raise").astype(int)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="raise").dt.tz_localize(None).dt.normalize()
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="raise")
    predictions["model"] = str(model)
    predictions = predictions.dropna(subset=["y_pred"])
    duplicates = predictions.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = predictions.loc[duplicates, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"{model} predictions have duplicate keys: {examples}")
    return predictions.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_common_panel(
    truth: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    zero_model_name: str = "Martingale",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions:
        raise ValueError("At least one trained-model prediction frame is required.")
    common_keys = pd.MultiIndex.from_frame(truth[KEY_COLUMNS])
    normalized: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, object]] = []
    for model, frame in predictions.items():
        normalized_frame = frame.copy() if set(KEY_COLUMNS + ["y_pred"]).issubset(frame.columns) else canonical_predictions(frame, model)
        normalized_frame["model"] = model
        normalized[model] = normalized_frame
        common_keys = common_keys.intersection(pd.MultiIndex.from_frame(normalized_frame[KEY_COLUMNS]))
        coverage_rows.append({
            "model": model,
            "n_predictions_before_intersection": int(len(normalized_frame)),
            "first_target_date": normalized_frame["target_date"].min(),
            "last_target_date": normalized_frame["target_date"].max(),
        })
    if len(common_keys) == 0:
        raise ValueError("The model outputs have no common ticker-date observations.")
    keys = common_keys.to_frame(index=False)
    common_truth = keys.merge(truth, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if common_truth["ticker_id"].nunique() != truth["ticker_id"].nunique():
        raise ValueError("The common panel does not contain every canonical ticker.")
    frames: list[pd.DataFrame] = []
    for model, frame in normalized.items():
        selected = keys.merge(frame[KEY_COLUMNS + ["y_pred"]], on=KEY_COLUMNS, how="left", validate="one_to_one")
        selected["model"] = model
        frames.append(selected)
    zero = keys.copy()
    zero["y_pred"] = 0.0
    zero["model"] = zero_model_name
    frames.append(zero)
    panel = pd.concat(frames, ignore_index=True).merge(common_truth, on=KEY_COLUMNS, how="left", validate="many_to_one")
    panel["resid"] = panel["y_true"] - panel["y_pred"]
    panel = panel.sort_values(["model", *KEY_COLUMNS]).reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage["n_common"] = int(len(common_truth))
    coverage["common_first_target_date"] = common_truth["target_date"].min()
    coverage["common_last_target_date"] = common_truth["target_date"].max()
    return panel, coverage


def evaluate_common_panel(panel: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the agreed BMR family on the canonical common panel.

    The parameter table repeats one fixed ``eps_err`` and one common
    ticker-specific train-only tail threshold for every model. Undefined
    conditional rates remain ``NaN`` and carry a machine-readable reason.
    """

    missing = set(KEY_COLUMNS + ["model", "y_true", "y_pred"]) - set(panel.columns)
    if missing:
        raise ValueError(f"Common panel is missing columns: {sorted(missing)}")
    missing = {"Model", "Ticker", "eps_err", "z-threshold"} - set(parameters.columns)
    if missing:
        raise ValueError(f"Metric parameter table is missing columns: {sorted(missing)}")
    if parameters.duplicated(["Model", "Ticker"]).any():
        raise ValueError("Metric parameters must be unique by Model and Ticker.")

    rows: list[dict[str, object]] = []
    for model, model_group in panel.groupby("model", sort=False):
        model_parameters = parameters[parameters["Model"] == model]
        z_counts: list[tuple[int, int]] = []
        invalid_threshold = False
        for ticker_id, group in model_group.groupby("ticker_id", sort=True):
            parameter = model_parameters[model_parameters["Ticker"].astype(str) == str(int(ticker_id))]
            if len(parameter) != 1:
                raise ValueError(f"Missing metric parameters for {model}, ticker {ticker_id}.")
            eps_err = float(parameter["eps_err"].iloc[0])
            threshold = float(parameter["z-threshold"].iloc[0])
            reasons: list[str] = []
            if int((group["y_true"] < 0).sum()) == 0:
                reasons.append("no bearish observations")
            if not np.isfinite(threshold):
                rate_z, misses_z, bears_z = float("nan"), 0, 0
                invalid_threshold = True
                parameter_reason = str(parameter.get("threshold_reason", pd.Series([""])).iloc[0]).strip()
                reasons.append(parameter_reason or "training tail threshold is undefined")
            else:
                rate_z, misses_z, bears_z = bmr_z(
                    group["y_true"].to_numpy(),
                    group["y_pred"].to_numpy(),
                    threshold,
                )
                if bears_z == 0:
                    reasons.append("no realised test-tail observations")
            z_counts.append((misses_z, bears_z))
            rows.append(
                _metric_row(
                    model,
                    str(int(ticker_id)),
                    group,
                    eps_err,
                    rate_z,
                    misses_z,
                    bears_z,
                    threshold,
                    "; ".join(reasons),
                )
            )

        all_parameter = model_parameters[model_parameters["Ticker"].astype(str).str.casefold() == "all"]
        if len(all_parameter) != 1:
            raise ValueError(f"Missing pooled metric parameters for {model}.")
        reasons = []
        if int((model_group["y_true"] < 0).sum()) == 0:
            reasons.append("no bearish observations")
        if invalid_threshold:
            misses_z = int(sum(misses for misses, _ in z_counts))
            bears_z = int(sum(bears for _, bears in z_counts))
            rate_z = float("nan")
            reasons.append("one or more training tail thresholds are undefined")
        else:
            rate_z, misses_z, bears_z = pooled_bmr_z(z_counts)
            if bears_z == 0:
                reasons.append("no realised test-tail observations")
        rows.append(
            _metric_row(
                model,
                "All",
                model_group,
                float(all_parameter["eps_err"].iloc[0]),
                rate_z,
                misses_z,
                bears_z,
                float("nan"),
                "; ".join(reasons),
            )
        )

    columns = [
        "Model", "Ticker", "RMSE", "R²", "BearMissRate",
        "BearMissRate_tol", "BearMissRate_z", "bear-miss", "bear-miss_tol",
        "bearish-observations", "z-miss", "z-bears", "z-threshold",
        "eps_err", "undefined_reason", "N",
    ]
    return pd.DataFrame(rows)[columns]


def _metric_row(
    model: str,
    ticker: str,
    group: pd.DataFrame,
    eps_err: float,
    rate_z: float,
    misses_z: int,
    bears_z: int,
    threshold: float,
    undefined_reason: str,
) -> dict[str, object]:
    actual = group["y_true"].to_numpy()
    prediction = group["y_pred"].to_numpy()
    bearish = actual < 0
    bear_misses = bearish & (prediction > actual)
    tolerant_misses = bearish & ((prediction - actual) >= eps_err)
    return {
        "Model": model,
        "Ticker": ticker,
        "RMSE": rmse(actual, prediction),
        "R²": r_squared(actual, prediction),
        "BearMissRate": bmr(actual, prediction),
        "BearMissRate_tol": bmr_tolerant(actual, prediction, eps_err),
        "BearMissRate_z": rate_z,
        "bear-miss": int(bear_misses.sum()),
        "bear-miss_tol": int(tolerant_misses.sum()),
        "bearish-observations": int(bearish.sum()),
        "z-miss": int(misses_z),
        "z-bears": int(bears_z),
        "z-threshold": threshold,
        "eps_err": float(eps_err),
        "undefined_reason": undefined_reason,
        "N": int(len(group)),
    }


def common_panel_wide(panel: pd.DataFrame) -> pd.DataFrame:
    truth = panel[KEY_COLUMNS + ["y_true"]].drop_duplicates(KEY_COLUMNS)
    predictions = panel.pivot(index=KEY_COLUMNS, columns="model", values="y_pred").reset_index()
    predictions.columns.name = None
    return truth.merge(predictions, on=KEY_COLUMNS, validate="one_to_one")


def load_default_predictions(prediction_paths: Mapping[str, Sequence[str | Path]] = DEFAULT_PREDICTION_PATHS) -> dict[str, pd.DataFrame]:
    return {model: read_first_available(paths) for model, paths in prediction_paths.items()}


__all__ = ["DEFAULT_PREDICTION_PATHS", "KEY_COLUMNS", "TICKER_TO_ID", "build_common_panel", "canonical_predictions", "canonical_truth", "common_panel_wide", "evaluate_common_panel", "load_default_predictions", "read_first_available"]
