"""Outer and inner time-split helpers used by the article workflow."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def fixed_dissertation_split(
    dataframe: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    purge_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return the submitted fixed train/test frames and purge boundary.

    Training rows satisfy ``index < cutoff_date - purge_days`` and test rows
    satisfy ``index >= cutoff_date``.  Rows between those boundaries are
    omitted.  Input order is preserved; the notebook sorts before this call.
    """

    cutoff = pd.to_datetime(cutoff_date)
    purge_date = cutoff - timedelta(days=purge_days)
    train = dataframe[dataframe.index < purge_date].copy()
    test = dataframe[dataframe.index >= cutoff].copy()
    return train, test, purge_date


def purged_train_validation_indices(
    n: int,
    lookback: int,
    val_fraction: float,
    *,
    minimum_gap: int = 5,
) -> tuple[int, int]:
    """Return ``(train_end, validation_start)`` for the article workflow.

    The validation suffix keeps approximately ``val_fraction`` of the sample.
    A distinct gap of ``max(lookback, minimum_gap)`` observations is excluded
    between the training prefix and validation suffix.  Indices follow normal
    Python slicing: training is ``[:train_end]`` and validation begins at
    ``[validation_start:]``.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if minimum_gap < 0:
        raise ValueError("minimum_gap cannot be negative")

    validation_size = max(1, int(np.floor(val_fraction * n)))
    validation_start = n - validation_size
    gap = max(int(lookback), int(minimum_gap))
    train_end = validation_start - gap
    if train_end <= lookback:
        raise ValueError(
            "Not enough observations for the requested lookback, purge, and "
            "validation fraction."
        )
    return int(train_end), int(validation_start)


__all__ = [
    "fixed_dissertation_split",
    "purged_train_validation_indices",
]
