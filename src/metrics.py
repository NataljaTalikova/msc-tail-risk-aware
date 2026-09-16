"""BMR-family metric definitions used by the article workflow.

The strict BMR boundaries and directional BMR_tol error reproduce the
submitted code. The article-approved evaluator supplies a pre-specified common
``eps_err`` and common ticker-specific train-only BMR_Z thresholds. Conditional
rates with zero denominators return ``NaN`` rather than a misleading zero.

Archived dissertation behaviour
--------------------------------
The submitted per-ticker Quantile Regression loop overwrites each ticker's
BMR_Z threshold while iterating over candidate lookbacks.  The saved metrics
therefore use the last evaluated lookback rather than the selected model's
threshold. That behaviour remains visible in the archived notebook but is not
used by the article evaluator.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import r2_score


MAD_SCALE = 1.4826
MAD_MULTIPLIER = 2.0
TAIL_QUANTILE = 0.0228
MIN_TAIL_EVENTS = 5


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the submitted root-mean-squared error calculation.

    The existing LSTM and linear-model helpers return ``NaN`` for an empty
    input, so that behaviour is retained here.
    """

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def r_squared(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return R² as used by the submitted sklearn-based model sections.

    Submitted sections guard single-observation groups before calling
    ``sklearn.metrics.r2_score``.  The same guard is applied here.
    """

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    if actual.size <= 1:
        return float("nan")
    return float(r2_score(actual, prediction))


def bmr(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the submitted standard Bear Miss Rate.

    Bearish class: ``actual < 0``.
    Miss: ``prediction > actual``.
    Equality is not a miss.  An empty bearish class returns ``NaN`` because
    the conditional rate has a zero denominator.
    """

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    bearish = actual < 0
    denominator = int(bearish.sum())
    if denominator == 0:
        return float("nan")
    misses = bearish & (prediction > actual)
    return float(misses.sum() / denominator)


def bmr_tolerant(
    actual: np.ndarray,
    prediction: np.ndarray,
    eps_err: float,
) -> float:
    """Return the submitted directional tolerant Bear Miss Rate.

    Bearish class: ``actual < 0``.
    Miss: ``(prediction - actual) >= eps_err``.
    The boundary is inclusive and no absolute error is used.  An empty
    bearish class returns ``NaN`` because the conditional rate has a zero
    denominator.
    """

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    bearish = actual < 0
    denominator = int(bearish.sum())
    if denominator == 0:
        return float("nan")
    misses = bearish & ((prediction - actual) >= eps_err)
    return float(misses.sum() / denominator)


def robust_bmr_z_threshold(
    y_train: np.ndarray,
    mad_multiplier: float = MAD_MULTIPLIER,
    fallback_quantile: float = TAIL_QUANTILE,
    min_tail_events: int = MIN_TAIL_EVENTS,
) -> float:
    """Construct the submitted robust BMR_Z threshold from supplied returns.

    The caller remains responsible for supplying the reference sample. The
    archived Martingale notebook applies an additional negative cap at its
    call site; the article evaluator deliberately does not.

    ``mad`` is the raw, unscaled median absolute deviation. The candidate is
    used only when ``1.4826 * mad`` is finite and strictly positive and it
    contains at least ``min_tail_events`` observations. Otherwise the function
    returns the supplied training quantile. ``np.quantile`` uses its default
    linear interpolation convention here.
    """

    y_train = np.asarray(y_train)
    if y_train.size == 0:
        return float("nan")

    median = float(np.median(y_train))
    mad = float(np.median(np.abs(y_train - median)))
    sigma_robust = MAD_SCALE * mad

    if np.isfinite(sigma_robust) and sigma_robust > 0.0:
        candidate = median - mad_multiplier * sigma_robust
        if int((y_train <= candidate).sum()) >= min_tail_events:
            return float(candidate)

    return float(np.quantile(y_train, fallback_quantile))


def bmr_z(
    actual: np.ndarray,
    prediction: np.ndarray,
    tau: float,
) -> tuple[float, int, int]:
    """Return submitted BMR_Z as ``(rate, misses, tail_observations)``.

    Actual tail: ``actual <= tau``.
    Miss: ``prediction > tau``.
    Equality of the prediction with ``tau`` is not a miss.  An empty actual
    tail returns ``(NaN, 0, 0)``.
    """

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    tail = actual <= tau
    tail_observations = int(tail.sum())
    if tail_observations == 0:
        return float("nan"), 0, 0
    misses = tail & (prediction > tau)
    miss_count = int(misses.sum())
    return float(miss_count / tail_observations), miss_count, tail_observations


def pooled_bmr_z(
    counts: Iterable[tuple[int, int]],
) -> tuple[float, int, int]:
    """Micro-aggregate submitted per-ticker BMR_Z miss/tail counts.

    ``counts`` contains ``(z_misses, z_tail_observations)`` pairs.  The rate
    is calculated from summed counts, not from the mean of ticker rates.
    When the pooled tail class is empty, the rate is ``NaN``.
    """

    miss_total = 0
    tail_total = 0
    for misses, tail_observations in counts:
        miss_total += int(misses)
        tail_total += int(tail_observations)

    rate = float(miss_total / tail_total) if tail_total > 0 else float("nan")
    return rate, miss_total, tail_total


__all__ = [
    "bmr",
    "bmr_tolerant",
    "bmr_z",
    "pooled_bmr_z",
    "r_squared",
    "rmse",
    "robust_bmr_z_threshold",
]
