import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features import yang_zhang_volatility
from src.forecast_context import concatenate_forecast_context


class YangZhangFormulaTests(unittest.TestCase):
    def test_uses_open_to_close_variance_and_full_rogers_satchell_term(self):
        frame = pd.DataFrame(
            {
                "Open": [100.0, 103.0, 102.0, 106.0],
                "High": [104.0, 107.0, 108.0, 110.0],
                "Low": [98.0, 100.0, 101.0, 104.0],
                "Close": [102.0, 101.0, 105.0, 107.0],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        window = 3

        high_open = np.log(frame["High"] / frame["Open"])
        low_open = np.log(frame["Low"] / frame["Open"])
        close_open = np.log(frame["Close"] / frame["Open"])
        open_previous_close = np.log(frame["Open"] / frame["Close"].shift(1))
        rogers_satchell = (
            high_open * (high_open - close_open)
            + low_open * (low_open - close_open)
        )
        weight = 0.34 / (1.34 + (window + 1) / (window - 1))
        expected = np.sqrt(
            open_previous_close.rolling(window).var()
            + weight * close_open.rolling(window).var()
            + (1 - weight) * rogers_satchell.rolling(window).mean()
        )

        pd.testing.assert_series_equal(
            yang_zhang_volatility(frame, window), expected, check_names=False
        )


class ForecastContextTests(unittest.TestCase):
    @staticmethod
    def _segment(values, dates):
        array = np.asarray(values, dtype=float)
        return SimpleNamespace(
            y=array,
            x=array.reshape(-1, 1),
            dates=np.asarray(pd.to_datetime(dates)),
        )

    def test_context_is_in_history_but_excluded_from_scored_test(self):
        training = self._segment([1, 2], ["2024-04-11", "2024-04-12"])
        context = self._segment(
            [3, 4, 5], ["2024-04-15", "2024-04-16", "2024-04-19"]
        )
        testing = self._segment([6, 7], ["2024-04-22", "2024-04-23"])

        y, x, dates, score_start = concatenate_forecast_context(
            training, context, testing
        )

        self.assertEqual(score_start, 5)
        np.testing.assert_array_equal(y[:score_start], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(x[score_start:].ravel(), [6, 7])
        np.testing.assert_array_equal(dates[score_start:], testing.dates)

    def test_overlapping_context_is_rejected(self):
        training = self._segment([1], ["2024-04-12"])
        context = self._segment([2], ["2024-04-12"])
        testing = self._segment([3], ["2024-04-22"])
        with self.assertRaisesRegex(ValueError, "ordered and disjoint"):
            concatenate_forecast_context(training, context, testing)


if __name__ == "__main__":
    unittest.main()
