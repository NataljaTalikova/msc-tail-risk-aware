import unittest

import numpy as np
import pandas as pd

from build_common_panel import common_metric_parameters
from src.features import (
    add_yang_zhang_volatility_causal,
    align_daily_features_past_only,
)
from src.metrics import bmr, bmr_tolerant, bmr_z
from src.paper_evaluation import (
    build_common_panel,
    canonical_predictions,
    canonical_truth,
    evaluate_common_panel,
)
from src.splits import purged_train_validation_indices


class PaperFeatureTests(unittest.TestCase):
    def test_past_only_alignment_does_not_backfill_leading_gap(self):
        source = pd.DataFrame(
            {"value": [2.0]}, index=pd.to_datetime(["2024-01-02"])
        )
        target = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        result = align_daily_features_past_only(source, target)
        self.assertTrue(np.isnan(result.loc["2024-01-01", "value"]))
        self.assertEqual(result.loc["2024-01-03", "value"], 2.0)

    def test_causal_yang_zhang_drops_warmup(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        close = np.linspace(100.0, 107.0, len(dates))
        frame = pd.DataFrame(
            {
                "Open": close - 0.2,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=dates,
        )
        result = add_yang_zhang_volatility_causal(frame, window=3)
        self.assertGreater(result.index.min(), dates.min())
        self.assertFalse(result["YZ_vol"].isna().any())

    def test_purged_split_excludes_a_distinct_gap(self):
        train_end, validation_start = purged_train_validation_indices(
            1000, lookback=60, val_fraction=0.10
        )
        self.assertEqual(validation_start, 900)
        self.assertEqual(train_end, 840)
        self.assertEqual(validation_start - train_end, 60)


class CommonPanelTests(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame(
            {
                "ticker_id": [0, 0, 0, 1, 1, 1],
                "Date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ),
                "log_return": [-0.03, 0.01, -0.01, -0.02, 0.02, -0.04],
            }
        )

    def test_common_panel_uses_intersection_and_adds_zero(self):
        truth = canonical_truth(self.test_data)
        model_a = pd.DataFrame(
            {
                "ticker_id": [0, 0, 1, 1],
                "Date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
                ),
                "y_pred": [0.0, -0.01, 0.0, -0.03],
            }
        )
        model_b = model_a.copy()
        model_b["y_pred"] += 0.001
        panel, coverage = build_common_panel(
            truth,
            {
                "A": canonical_predictions(model_a, "A"),
                "B": canonical_predictions(model_b, "B"),
            },
        )
        self.assertEqual(panel["model"].nunique(), 3)
        self.assertEqual(panel["target_date"].nunique(), 2)
        self.assertEqual(len(panel), 12)
        self.assertTrue((panel.loc[panel["model"] == "Martingale", "y_pred"] == 0).all())
        self.assertTrue((coverage["n_common"] == 4).all())

    def test_bmr_strict_boundary_does_not_count_equality(self):
        self.assertEqual(bmr(np.array([-0.01]), np.array([-0.01])), 0.0)

    def test_bmr_z_strict_boundary_does_not_count_cutoff_equality(self):
        rate, misses, tail_observations = bmr_z(
            np.array([-0.03]), np.array([-0.02]), tau=-0.02
        )
        self.assertEqual(rate, 0.0)
        self.assertEqual(misses, 0)
        self.assertEqual(tail_observations, 1)

    def test_common_parameters_reject_nonnegative_training_tail(self):
        rows = []
        for ticker in range(5):
            for date in pd.date_range("2020-01-01", periods=10, freq="D"):
                rows.append(
                    {
                        "ticker_id": ticker,
                        "Date": date,
                        "log_return": 0.01,
                    }
                )
        parameters = common_metric_parameters(pd.DataFrame(rows))
        ticker_rows = parameters[parameters["Ticker"] != "All"]
        self.assertTrue(ticker_rows["z-threshold"].isna().all())
        self.assertTrue(
            ticker_rows["threshold_reason"].eq(
                "training sample does not define a negative tail"
            ).all()
        )
        self.assertTrue(parameters["eps_err"].eq(0.01).all())

    def test_metrics_preserve_original_names_and_model_parameters(self):
        truth = canonical_truth(self.test_data)
        predictions = pd.DataFrame(
            {
                "ticker_id": self.test_data["ticker_id"],
                "Date": self.test_data["Date"],
                "y_pred": [-0.05, 0.0, -0.02, 0.0, 0.0, -0.05],
            }
        )
        panel, _ = build_common_panel(
            truth, {"A": canonical_predictions(predictions, "A")}
        )
        parameters = pd.DataFrame(
            [
                {"Model": model, "Ticker": ticker, "eps_err": 0.01, "z-threshold": -0.025}
                for model in ["A", "Martingale"]
                for ticker in ["0", "1"]
            ]
            + [
                {"Model": model, "Ticker": "All", "eps_err": 0.01, "z-threshold": np.nan}
                for model in ["A", "Martingale"]
            ]
        )
        metrics = evaluate_common_panel(panel, parameters)
        model_rows = metrics[metrics["Model"] == "A"]
        self.assertEqual(set(model_rows["Ticker"]), {"0", "1", "All"})
        self.assertTrue(
            {"BearMissRate", "BearMissRate_tol", "BearMissRate_z"}.issubset(
                metrics.columns
            )
        )
        self.assertFalse({"DUR", "DUR_tol", "tail_miss_rate"} & set(metrics.columns))
        self.assertEqual(
            int(model_rows.loc[model_rows["Ticker"] == "All", "N"].iloc[0]), 6
        )
        all_row = model_rows.loc[model_rows["Ticker"] == "All"].iloc[0]
        ticker_rows = model_rows.loc[model_rows["Ticker"] != "All"]
        self.assertEqual(
            int(all_row["bear-miss_tol"]), int(ticker_rows["bear-miss_tol"].sum())
        )

    def test_empty_bearish_class_returns_nan_with_reason(self):
        actual = np.array([0.01, 0.02])
        prediction = np.array([0.00, 0.01])
        self.assertTrue(np.isnan(bmr(actual, prediction)))
        self.assertTrue(np.isnan(bmr_tolerant(actual, prediction, 0.01)))

        truth = canonical_truth(
            pd.DataFrame(
                {
                    "ticker_id": [0, 0],
                    "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "log_return": actual,
                }
            )
        )
        raw = pd.DataFrame(
            {
                "ticker_id": [0, 0],
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "y_pred": prediction,
            }
        )
        panel, _ = build_common_panel(
            truth, {"A": canonical_predictions(raw, "A")}
        )
        parameters = pd.DataFrame(
            [
                {"Model": model, "Ticker": ticker, "eps_err": 0.01, "z-threshold": threshold}
                for model in ["A", "Martingale"]
                for ticker, threshold in [("0", -0.05), ("All", np.nan)]
            ]
        )
        metrics = evaluate_common_panel(panel, parameters)
        self.assertTrue(metrics["BearMissRate"].isna().all())
        self.assertTrue(metrics["BearMissRate_tol"].isna().all())
        self.assertTrue(metrics["undefined_reason"].str.contains("no bearish").all())

    def test_invalid_training_tail_threshold_is_nan_with_reason(self):
        truth = canonical_truth(self.test_data[self.test_data["ticker_id"] == 0])
        raw = pd.DataFrame(
            {
                "ticker_id": [0, 0, 0],
                "Date": truth["target_date"],
                "y_pred": [0.0, 0.0, 0.0],
            }
        )
        panel, _ = build_common_panel(
            truth, {"A": canonical_predictions(raw, "A")}
        )
        parameters = pd.DataFrame(
            [
                {
                    "Model": model,
                    "Ticker": "0",
                    "eps_err": 0.01,
                    "z-threshold": np.nan,
                    "threshold_reason": "training sample does not define a negative tail",
                }
                for model in ["A", "Martingale"]
            ]
            + [
                {"Model": model, "Ticker": "All", "eps_err": 0.01, "z-threshold": np.nan}
                for model in ["A", "Martingale"]
            ]
        )
        metrics = evaluate_common_panel(panel, parameters)
        self.assertTrue(metrics["BearMissRate_z"].isna().all())
        self.assertTrue(
            metrics["undefined_reason"].str.contains("threshold|negative tail").all()
        )


if __name__ == "__main__":
    unittest.main()
