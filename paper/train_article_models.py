"""Retune and refit the two estimated models retained for the article.

The script is intentionally separate from the archived dissertation notebook.
It uses a real purged train/validation split, scaler fits confined to the
training prefix, reduced grids, and target-date forecasts that include the
entire corrected test period.  Tuning logs are append-only and successful
configurations are skipped on reruns.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import robust_bmr_z_threshold  # noqa: E402
from src.forecast_context import concatenate_forecast_context  # noqa: E402
from src.models import (  # noqa: E402
    HuberPinballLoss,
    build_article_lstm,
    build_submitted_quantile_regression,
)
from src.splits import purged_train_validation_indices  # noqa: E402


DATA_DIR = ROOT / "outputs" / "paper" / "data"
TRAIN_CSV = DATA_DIR / "df_LSTM_train.csv"
CONTEXT_CSV = DATA_DIR / "df_LSTM_context.csv"
TEST_CSV = DATA_DIR / "df_LSTM_test.csv"

ID_COL = "ticker_id"
TARGET_COL = "log_return"
TICKERS = [0, 1, 2, 3, 4]
TICKER_NAMES = {0: "BSIF", 1: "HTG", 2: "SSE", 3: "TLW", 4: "UKW"}

NUMERIC_CANDIDATES = [
    "EMA_30",
    "RSI",
    "MACD",
    "YZ_vol",
    "BrGBP_logchg",
    "MarketCap_GBP",
    "OVX",
    "GUK10Y",
    "FTSE_lr",
]
FLAG_CANDIDATES = [
    "brexit_transition",
    "covid_awareness",
    "post_pandemic",
    "war_ukraine",
    "renewables",
]

VAL_FRACTION = 0.10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SEED = 42
HU_PIN_PRECISION_FLOOR = 0.30
GRIDS = {
    "hupin": {
        "lookback": [60, 90],
        "hidden_dim": [32, 64],
        "n_layers": [2],
        "dropout": [0.2],
        "huber_beta": [0.05, 0.075],
        "tau": [0.01, 0.025, 0.05],
        "lambda_pin": [0.5, 1.0],
    },
    "quantile_joint": {
        "lookback": [60, 90],
        "alpha": [1e-4, 1e-3],
        "tau": [0.5],
    },
}


@dataclass(frozen=True)
class TickerArrays:
    y: np.ndarray
    x: np.ndarray
    dates: np.ndarray


def _set_reproducibility() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _grid(grid: dict[str, list[object]], smoke: bool) -> list[dict[str, object]]:
    keys = list(grid)
    values = [[grid[key][0]] if smoke else grid[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def _config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _add_ticker_ohe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    columns = [f"tic_{ticker}" for ticker in TICKERS]
    for ticker in TICKERS:
        result[f"tic_{ticker}"] = (result[ID_COL].astype(int) == ticker).astype("int8")
    return result, columns


def _validate_context_frames(
    train: pd.DataFrame, context: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Require distinct, ordered fit/context/score segments for every ticker."""

    for label, frame in {"train": train, "context": context, "test": test}.items():
        if sorted(frame[ID_COL].unique()) != TICKERS:
            raise ValueError(f"{label} does not contain exactly the expected tickers")
        if frame.duplicated([ID_COL, "Date"]).any():
            raise ValueError(f"{label} contains duplicate ticker/date keys")
    for ticker in TICKERS:
        training = train.loc[train[ID_COL] == ticker, "Date"]
        excluded = context.loc[context[ID_COL] == ticker, "Date"]
        testing = test.loc[test[ID_COL] == ticker, "Date"]
        if not training.max() < excluded.min() <= excluded.max() < testing.min():
            raise ValueError(f"Ticker {ticker} has overlapping or unordered forecast context")


def _load_frames(
    joint: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    if not all(path.exists() for path in [TRAIN_CSV, CONTEXT_CSV, TEST_CSV]):
        raise FileNotFoundError(
            "Corrected article data are missing. Execute EDA.ipynb first so "
            f"that {TRAIN_CSV}, {CONTEXT_CSV}, and {TEST_CSV} exist."
        )
    train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"]).sort_values([ID_COL, "Date"])
    context = pd.read_csv(CONTEXT_CSV, parse_dates=["Date"]).sort_values([ID_COL, "Date"])
    test = pd.read_csv(TEST_CSV, parse_dates=["Date"]).sort_values([ID_COL, "Date"])
    _validate_context_frames(train, context, test)
    ohe_columns: list[str] = []
    if joint:
        train, ohe_columns = _add_ticker_ohe(train)
        context, _ = _add_ticker_ohe(context)
        test, _ = _add_ticker_ohe(test)

    numeric = [column for column in NUMERIC_CANDIDATES if column in train.columns]
    flags = [column for column in FLAG_CANDIDATES if column in train.columns]
    covariates = numeric + flags + ohe_columns
    required = [ID_COL, "Date", TARGET_COL, *covariates]
    if any(frame[required].isna().any().any() for frame in [train, context, test]):
        raise ValueError("Corrected train/context/test data contain model-input NaNs.")
    numeric_indices = [covariates.index(column) for column in numeric]
    return train, context, test, covariates, numeric_indices


def _arrays(frame: pd.DataFrame, covariates: list[str]) -> dict[int, TickerArrays]:
    result: dict[int, TickerArrays] = {}
    for ticker, group in frame.groupby(ID_COL, sort=True):
        ordered = group.sort_values("Date")
        result[int(ticker)] = TickerArrays(
            y=ordered[TARGET_COL].to_numpy(dtype=np.float32),
            x=ordered[covariates].to_numpy(dtype=np.float32),
            dates=ordered["Date"].to_numpy(),
        )
    if sorted(result) != TICKERS:
        raise ValueError(f"Expected tickers {TICKERS}; found {sorted(result)}")
    return result


def _fit_scaler(
    raw: dict[int, TickerArrays],
    end_by_ticker: dict[int, int],
    numeric_indices: list[int],
    tickers: Iterable[int],
) -> StandardScaler:
    scaler = StandardScaler()
    ticker_list = list(tickers)
    if numeric_indices:
        blocks = [
            raw[ticker].x[: end_by_ticker[ticker], numeric_indices]
            for ticker in ticker_list
        ]
        scaler.fit(np.vstack(blocks))
    return scaler


def _scaled(
    raw: dict[int, TickerArrays],
    scaler: StandardScaler,
    numeric_indices: list[int],
) -> dict[int, TickerArrays]:
    result: dict[int, TickerArrays] = {}
    for ticker, item in raw.items():
        values = item.x.copy()
        if numeric_indices:
            values[:, numeric_indices] = scaler.transform(values[:, numeric_indices])
        result[ticker] = TickerArrays(item.y, values, item.dates)
    return result


def _series(values: np.ndarray):
    from darts import TimeSeries

    return TimeSeries.from_values(values.astype(np.float32, copy=False))


def _split_maps(
    raw: dict[int, TickerArrays], lookback: int, tickers: Iterable[int]
) -> tuple[dict[int, int], dict[int, int]]:
    train_ends: dict[int, int] = {}
    validation_starts: dict[int, int] = {}
    for ticker in tickers:
        train_end, validation_start = purged_train_validation_indices(
            len(raw[ticker].y), lookback, VAL_FRACTION
        )
        train_ends[ticker] = train_end
        validation_starts[ticker] = validation_start
    return train_ends, validation_starts


def _fit_tuning_lstm(
    scaled: dict[int, TickerArrays],
    tickers: list[int],
    train_ends: dict[int, int],
    validation_starts: dict[int, int],
    config: dict[str, object],
    loss_fn: nn.Module,
    max_epochs: int,
    patience: int,
    progress: bool,
):
    lookback = int(config["lookback"])
    model = build_article_lstm(
        lookback=lookback,
        hidden_dim=int(config["hidden_dim"]),
        n_rnn_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        batch_size=BATCH_SIZE,
        n_epochs=max_epochs,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        loss_fn=loss_fn,
        early_patience=patience,
        progress_bar=progress,
    )
    training_series = []
    training_covariates = []
    validation_series = []
    validation_covariates = []
    for ticker in tickers:
        item = scaled[ticker]
        train_end = train_ends[ticker]
        validation_start = validation_starts[ticker]
        validation_context_start = validation_start - lookback
        training_series.append(_series(item.y[:train_end]))
        training_covariates.append(_series(item.x[:train_end]))
        validation_series.append(_series(item.y[validation_context_start:]))
        validation_covariates.append(_series(item.x[validation_context_start:]))
    model.fit(
        series=training_series,
        past_covariates=training_covariates,
        val_series=validation_series,
        val_past_covariates=validation_covariates,
        verbose=progress,
    )
    return model


def _historical_predictions(
    model,
    scaled: dict[int, TickerArrays],
    tickers: Iterable[int],
    starts: dict[int, int],
) -> pd.DataFrame:
    ticker_list = list(tickers)
    targets = {ticker: _series(scaled[ticker].y) for ticker in ticker_list}
    covariates = {ticker: _series(scaled[ticker].x) for ticker in ticker_list}
    forecasts: dict[int, object] = {}

    common_start = len({starts[ticker] for ticker in ticker_list}) == 1
    if len(ticker_list) > 1 and common_start:
        start = starts[ticker_list[0]]
        batched = model.historical_forecasts(
            series=[targets[ticker] for ticker in ticker_list],
            past_covariates=[covariates[ticker] for ticker in ticker_list],
            start=targets[ticker_list[0]].time_index[start],
            forecast_horizon=1,
            stride=1,
            retrain=False,
            last_points_only=True,
            verbose=False,
        )
        forecasts = dict(zip(ticker_list, batched))
    else:
        for ticker in ticker_list:
            forecasts[ticker] = model.historical_forecasts(
                series=targets[ticker],
                past_covariates=covariates[ticker],
                start=targets[ticker].time_index[starts[ticker]],
                forecast_horizon=1,
                stride=1,
                retrain=False,
                last_points_only=True,
                verbose=False,
            )

    rows: list[pd.DataFrame] = []
    for ticker in ticker_list:
        item = scaled[ticker]
        target_series = targets[ticker]
        forecast = forecasts[ticker].slice_intersect(target_series)
        indices = np.asarray(forecast.time_index, dtype=int)
        prediction = forecast.values(copy=False).reshape(-1)
        rows.append(
            pd.DataFrame(
                {
                    ID_COL: int(ticker),
                    "Date": pd.to_datetime(item.dates[indices]),
                    "y_true": item.y[indices].astype(float),
                    "y_pred": prediction.astype(float),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True).sort_values([ID_COL, "Date"])
    result["resid"] = result["y_true"] - result["y_pred"]
    return result.reset_index(drop=True)


def _common_outer_training_thresholds(
    raw: dict[int, TickerArrays], tickers: Iterable[int]
) -> dict[int, float]:
    """Return one full-pre-test-training threshold per ticker for selection.

    The same event definition is used by every configuration and by final
    common-panel evaluation. It is deliberately independent of model lookback.
    """

    thresholds: dict[int, float] = {}
    for ticker in tickers:
        threshold = robust_bmr_z_threshold(raw[ticker].y)
        thresholds[ticker] = (
            float(threshold)
            if np.isfinite(threshold) and threshold < 0.0
            else float("nan")
        )
    return thresholds


def _validation_summary(
    predictions: pd.DataFrame, thresholds: dict[int, float]
) -> dict[str, float | int]:
    errors = predictions["y_true"].to_numpy() - predictions["y_pred"].to_numpy()
    rmse = float(np.sqrt(np.mean(errors**2)))
    tp = fp = fn = 0
    for ticker, group in predictions.groupby(ID_COL):
        threshold = thresholds[int(ticker)]
        actual_tail = group["y_true"].to_numpy() <= threshold
        predicted_tail = group["y_pred"].to_numpy() <= threshold
        tp += int(np.sum(actual_tail & predicted_tail))
        fp += int(np.sum(~actual_tail & predicted_tail))
        fn += int(np.sum(actual_tail & ~predicted_tail))
    precision = float(tp / (tp + fp)) if tp + fp else float("nan")
    recall = float(tp / (tp + fn)) if tp + fn else float("nan")
    f2 = (
        float(5 * precision * recall / (4 * precision + recall))
        if np.isfinite(precision)
        and np.isfinite(recall)
        and (4 * precision + recall) > 0
        else float("nan")
    )
    return {
        "val_rmse": rmse,
        "tail_precision": precision,
        "tail_recall": recall,
        "tail_f2": f2,
        "tail_tp": tp,
        "tail_fp": fp,
        "tail_fn": fn,
        "val_n": int(len(predictions)),
    }


def _trained_epochs(model, maximum: int) -> int:
    trainer = getattr(model, "trainer", None)
    current = getattr(trainer, "current_epoch", None)
    if current is None:
        return int(maximum)
    return max(1, min(int(maximum), int(current) + 1))


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "status" not in frame or "config_key" not in frame:
        return set()
    return set(frame.loc[frame["status"] == "ok", "config_key"].astype(str))


def _write_best_config(
    log_path: Path, output_path: Path, model_name: str, ticker: int | None = None
) -> dict[str, object]:
    frame = pd.read_csv(log_path)
    frame = frame[frame["status"] == "ok"].copy()
    if ticker is not None:
        frame = frame[frame[ID_COL] == ticker]
    if frame.empty:
        raise RuntimeError(f"No successful tuning rows for {model_name} ticker={ticker}")

    if model_name == "hupin":
        qualified = frame[
            frame["tail_precision"].ge(HU_PIN_PRECISION_FLOOR)
            & frame["tail_f2"].notna()
        ]
        candidates = qualified if not qualified.empty else frame
        candidates = candidates.sort_values(
            ["tail_f2", "tail_recall", "val_rmse"],
            ascending=[False, False, True],
            na_position="last",
        )
        objective = "max tail F2 subject to validation precision >= 0.30; RMSE tie-break"
    else:
        candidates = frame.sort_values("val_rmse", ascending=True)
        objective = "minimum pooled validation RMSE"

    best = candidates.iloc[0]
    config = json.loads(best["config_key"])
    payload: dict[str, object] = {
        "model": model_name,
        "best_config": config,
        "refit_epochs": int(best.get("trained_epochs", 1)),
        "selection_objective": objective,
        "validation": {
            "rmse": float(best["val_rmse"]),
            "tail_precision": float(best["tail_precision"]),
            "tail_recall": float(best["tail_recall"]),
            "tail_f2": float(best["tail_f2"]),
            "n": int(best["val_n"]),
        },
    }
    if ticker is not None:
        payload["ticker_id"] = int(ticker)
        payload["ticker"] = TICKER_NAMES[int(ticker)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2), encoding="utf-8"
    )
    return payload


def _json_safe(value):
    """Replace non-finite floats so metadata remains strict JSON."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def tune_hupin(
    output_root: Path,
    max_epochs: int,
    patience: int,
    smoke: bool,
    progress: bool,
) -> None:
    model_name = "hupin"
    train, _, _, covariates, numeric_indices = _load_frames(joint=True)
    raw = _arrays(train, covariates)
    configs = _grid(GRIDS[model_name], smoke)
    thresholds = _common_outer_training_thresholds(raw, TICKERS)
    directory = output_root / model_name
    log_path = directory / "tuning_results.csv"
    completed = _completed_keys(log_path)

    for number, config in enumerate(configs, start=1):
        key = _config_key(config)
        if key in completed:
            print(f"[{model_name} {number}/{len(configs)}] resume-skip {key}", flush=True)
            continue
        print(f"[{model_name} {number}/{len(configs)}] joint {config}", flush=True)
        started = time.perf_counter()
        base_row: dict[str, object] = {
            "model": model_name,
            ID_COL: "ALL",
            "config_key": key,
            **config,
        }
        try:
            lookback = int(config["lookback"])
            train_ends, validation_starts = _split_maps(raw, lookback, TICKERS)
            scaler = _fit_scaler(raw, train_ends, numeric_indices, TICKERS)
            scaled = _scaled(raw, scaler, numeric_indices)
            loss_fn: nn.Module = HuberPinballLoss(
                beta=float(config["huber_beta"]),
                tau=float(config["tau"]),
                lam=float(config["lambda_pin"]),
            )
            model = _fit_tuning_lstm(
                scaled,
                TICKERS,
                train_ends,
                validation_starts,
                config,
                loss_fn,
                max_epochs,
                patience,
                progress,
            )
            epochs_trained = _trained_epochs(model, max_epochs)
            predictions = _historical_predictions(
                model, scaled, TICKERS, validation_starts
            )
            metrics = _validation_summary(predictions, thresholds)
            row = {
                **base_row,
                "status": "ok",
                **metrics,
                "trained_epochs": epochs_trained,
                "seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
            print(
                f"  RMSE={metrics['val_rmse']:.6f} F2={metrics['tail_f2']:.3f} "
                f"epochs={row['trained_epochs']} seconds={row['seconds']}",
                flush=True,
            )
        except Exception as exc:  # keep the remaining resumable grid running
            row = {
                **base_row,
                "status": "failed",
                "val_rmse": np.nan,
                "tail_precision": np.nan,
                "tail_recall": np.nan,
                "tail_f2": np.nan,
                "tail_tp": 0,
                "tail_fp": 0,
                "tail_fn": 0,
                "val_n": 0,
                "trained_epochs": 0,
                "seconds": round(time.perf_counter() - started, 3),
                "error": repr(exc),
            }
            print(f"  FAILED: {exc!r}", flush=True)
        _append_row(log_path, row)

    _write_best_config(log_path, directory / "best_config.json", model_name)


def tune_quantile(output_root: Path, smoke: bool) -> None:
    model_name = "quantile_joint"
    train, _, _, covariates, numeric_indices = _load_frames(joint=True)
    raw = _arrays(train, covariates)
    configs = _grid(GRIDS[model_name], smoke)
    thresholds = _common_outer_training_thresholds(raw, TICKERS)
    directory = output_root / model_name
    log_path = directory / "tuning_results.csv"
    completed = _completed_keys(log_path)

    for number, config in enumerate(configs, start=1):
        key = _config_key(config)
        if key in completed:
            print(f"[{model_name} {number}/{len(configs)}] resume-skip {key}", flush=True)
            continue
        print(f"[{model_name} {number}/{len(configs)}] {config}", flush=True)
        started = time.perf_counter()
        base_row: dict[str, object] = {
            "model": model_name,
            ID_COL: "ALL",
            "config_key": key,
            **config,
        }
        try:
            lookback = int(config["lookback"])
            train_ends, validation_starts = _split_maps(raw, lookback, TICKERS)
            scaler = _fit_scaler(raw, train_ends, numeric_indices, TICKERS)
            scaled = _scaled(raw, scaler, numeric_indices)
            model = build_submitted_quantile_regression(
                lookback=lookback,
                alpha=float(config["alpha"]),
                tau=float(config["tau"]),
            )
            model.fit(
                series=[_series(scaled[ticker].y[: train_ends[ticker]]) for ticker in TICKERS],
                past_covariates=[_series(scaled[ticker].x[: train_ends[ticker]]) for ticker in TICKERS],
            )
            predictions = _historical_predictions(
                model, scaled, TICKERS, validation_starts
            )
            metrics = _validation_summary(predictions, thresholds)
            row = {
                **base_row,
                "status": "ok",
                **metrics,
                "trained_epochs": 1,
                "seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
            print(f"  RMSE={metrics['val_rmse']:.6f} seconds={row['seconds']}", flush=True)
        except Exception as exc:
            row = {
                **base_row,
                "status": "failed",
                "val_rmse": np.nan,
                "tail_precision": np.nan,
                "tail_recall": np.nan,
                "tail_f2": np.nan,
                "tail_tp": 0,
                "tail_fp": 0,
                "tail_fn": 0,
                "val_n": 0,
                "trained_epochs": 1,
                "seconds": round(time.perf_counter() - started, 3),
                "error": repr(exc),
            }
            print(f"  FAILED: {exc!r}", flush=True)
        _append_row(log_path, row)
    _write_best_config(log_path, directory / "best_config.json", model_name)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Run the tuning stage first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _final_test_predictions(
    model,
    train_scaled: dict[int, TickerArrays],
    context_scaled: dict[int, TickerArrays],
    test_scaled: dict[int, TickerArrays],
    tickers: Iterable[int],
) -> pd.DataFrame:
    combined: dict[int, TickerArrays] = {}
    starts: dict[int, int] = {}
    for ticker in tickers:
        training = train_scaled[ticker]
        context = context_scaled[ticker]
        testing = test_scaled[ticker]
        y, x, dates, score_start = concatenate_forecast_context(
            training, context, testing
        )
        combined[ticker] = TickerArrays(
            y=y,
            x=x,
            dates=dates,
        )
        starts[ticker] = score_start
    return _historical_predictions(model, combined, tickers, starts)


def _save_predictions(frame: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "final_test_predictions.csv", index=False)
    frame.to_parquet(directory / "final_test_predictions.parquet", index=False)


def fit_hupin(output_root: Path, progress: bool) -> None:
    model_name = "hupin"
    train, context, test, covariates, numeric_indices = _load_frames(joint=True)
    train_raw = _arrays(train, covariates)
    context_raw = _arrays(context, covariates)
    test_raw = _arrays(test, covariates)
    directory = output_root / model_name

    payload = _load_json(directory / "best_config.json")
    config = payload["best_config"]
    epochs = int(payload["refit_epochs"])
    full_ends = {ticker: len(train_raw[ticker].y) for ticker in TICKERS}
    scaler = _fit_scaler(train_raw, full_ends, numeric_indices, TICKERS)
    train_scaled = _scaled(train_raw, scaler, numeric_indices)
    context_scaled = _scaled(context_raw, scaler, numeric_indices)
    test_scaled = _scaled(test_raw, scaler, numeric_indices)
    loss_fn = HuberPinballLoss(
        beta=float(config["huber_beta"]),
        tau=float(config["tau"]),
        lam=float(config["lambda_pin"]),
    )
    model = build_article_lstm(
        lookback=int(config["lookback"]),
        hidden_dim=int(config["hidden_dim"]),
        n_rnn_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        batch_size=BATCH_SIZE,
        n_epochs=epochs,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        loss_fn=loss_fn,
        early_patience=None,
        progress_bar=progress,
    )
    model.fit(
        series=[_series(train_scaled[ticker].y) for ticker in TICKERS],
        past_covariates=[_series(train_scaled[ticker].x) for ticker in TICKERS],
        verbose=progress,
    )
    model_path = directory / "models" / f"{model_name}.darts"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    predictions = _final_test_predictions(
        model, train_scaled, context_scaled, test_scaled, TICKERS
    )
    _save_predictions(predictions, directory)


def fit_quantile(output_root: Path) -> None:
    model_name = "quantile_joint"
    train, context, test, covariates, numeric_indices = _load_frames(joint=True)
    train_raw = _arrays(train, covariates)
    context_raw = _arrays(context, covariates)
    test_raw = _arrays(test, covariates)
    directory = output_root / model_name
    payload = _load_json(directory / "best_config.json")
    config = payload["best_config"]
    full_ends = {ticker: len(train_raw[ticker].y) for ticker in TICKERS}
    scaler = _fit_scaler(train_raw, full_ends, numeric_indices, TICKERS)
    train_scaled = _scaled(train_raw, scaler, numeric_indices)
    context_scaled = _scaled(context_raw, scaler, numeric_indices)
    test_scaled = _scaled(test_raw, scaler, numeric_indices)
    model = build_submitted_quantile_regression(
        lookback=int(config["lookback"]),
        alpha=float(config["alpha"]),
        tau=float(config["tau"]),
    )
    model.fit(
        series=[_series(train_scaled[ticker].y) for ticker in TICKERS],
        past_covariates=[_series(train_scaled[ticker].x) for ticker in TICKERS],
    )
    model_path = directory / "models" / "quantile_joint.darts"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    predictions = _final_test_predictions(
        model, train_scaled, context_scaled, test_scaled, TICKERS
    )
    _save_predictions(predictions, directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "hupin", "quantile_joint"],
        default="all",
    )
    parser.add_argument("--stage", choices=["tune", "fit", "all"], default="all")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one configuration per model with two epochs under outputs/paper/smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_reproducibility()
    output_root = ROOT / "outputs" / "paper" / ("smoke" if args.smoke else "")
    max_epochs = min(args.max_epochs, 2) if args.smoke else args.max_epochs
    patience = 1 if args.smoke else args.patience
    selected = ["hupin", "quantile_joint"] if args.model == "all" else [args.model]
    for model_name in selected:
        if args.stage in {"tune", "all"}:
            if model_name == "quantile_joint":
                tune_quantile(output_root, args.smoke)
            else:
                tune_hupin(
                    output_root,
                    max_epochs,
                    patience,
                    args.smoke,
                    args.progress,
                )
        if args.stage in {"fit", "all"}:
            if model_name == "quantile_joint":
                fit_quantile(output_root)
            else:
                fit_hupin(output_root, args.progress)
    print("Article model workflow completed.", flush=True)


if __name__ == "__main__":
    main()
