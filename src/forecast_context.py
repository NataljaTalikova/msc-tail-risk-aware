"""Pure helpers for assembling fit, context, and scored forecast segments."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ArraySegment(Protocol):
    y: np.ndarray
    x: np.ndarray
    dates: np.ndarray


def concatenate_forecast_context(
    training: ArraySegment,
    context: ArraySegment,
    testing: ArraySegment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Join the segments and return the first index that should be scored.

    Context rows sit between the fitted training data and the test data. They
    supply observed lag history, but the returned score start excludes them.
    """

    if not len(training.y) or not len(context.y) or not len(testing.y):
        raise ValueError("training, context, and testing segments must be non-empty")
    if not (
        np.max(training.dates) < np.min(context.dates)
        and np.max(context.dates) < np.min(testing.dates)
    ):
        raise ValueError("forecast segments must be strictly ordered and disjoint")

    y = np.concatenate([training.y, context.y, testing.y])
    x = np.vstack([training.x, context.x, testing.x])
    dates = np.concatenate([training.dates, context.dates, testing.dates])
    score_start = len(training.y) + len(context.y)
    return y, x, dates, score_start
