import math
import unittest

from forecast_intervals import Design, Forecast, backtest, conformal_radius, simulate, summarize


class OrderStatisticTests(unittest.TestCase):
    def test_uses_finite_sample_rank_including_ties(self):
        self.assertEqual(conformal_radius([1, 2, 2, 3], 0.6), 2)
        self.assertEqual(conformal_radius(range(9), 0.9), 8)

    def test_on_exact_decimal_boundary(self):
        self.assertEqual(conformal_radius(range(99), 0.07), 6)

    def test_small_samples_have_unbounded_radius(self):
        self.assertEqual(conformal_radius([1.0, 2.0], 0.95), math.inf)

    def test_rejects_invalid_calibration_inputs(self):
        for sample, coverage in (([], 0.5), ([1], 0), ([1], 1), ([-0.1], 0.8),
                                 ([math.nan], 0.9), ([math.inf], 0.9)):
            with self.subTest(sample=sample, coverage=coverage), self.assertRaises(ValueError):
                conformal_radius(sample, coverage)


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = Design(period=12, train_end=72, test_start=96, shift=140, length=180)
        cls.series = simulate(7, cls.design)
        cls.rows = backtest(cls.series, cls.design, 7, window=24)

    def test_generator_reproducible_across_runs(self):
        self.assertEqual(self.series, simulate(7, self.design))
        self.assertNotEqual(self.series, simulate(8, self.design))

    def test_current_target_and_future_cannot_change_forecast_or_bounds(self):
        cut = 121
        changed = self.series[:cut] + [1000.0] * (self.design.length - cut)
        rerun = backtest(changed, self.design, 7, window=24)
        def prior_forecasts(rows):
            return [(r.time, r.model, r.calibration, r.forecast, r.low, r.high)
                    for r in rows if r.time <= cut]
        self.assertEqual(prior_forecasts(self.rows), prior_forecasts(rerun))
        self.assertNotEqual(self.rows, rerun)

    def test_balanced_origins_and_first_intervals(self):
        self.assertEqual(len(self.rows), 4 * (self.design.length - self.design.test_start))
        for model in {"seasonal_ar", "seasonal_naive"}:
            first = [r for r in self.rows if r.time == self.design.test_start and r.model == model]
            self.assertEqual({r.calibration for r in first}, {"fixed", "rolling"})
            self.assertEqual(len({(r.forecast, r.low, r.high) for r in first}), 1)

    def test_fixed_interval_stays_constant(self):
        fixed = [r for r in self.rows if r.model == "seasonal_ar" and r.calibration == "fixed"]
        for r in fixed:
            self.assertAlmostEqual(r.high - r.low, fixed[0].high - fixed[0].low)

    def test_rolling_window_contains_last_resolved_absolute_errors(self):
        t, n = 136, 24
        scores = [abs(self.series[i] - self.series[i - self.design.period]) for i in range(t-n, t)]
        row = next(r for r in self.rows if r.time == t and r.model == "seasonal_naive"
                   and r.calibration == "rolling")
        self.assertAlmostEqual(row.high - row.forecast, conformal_radius(scores, 0.9))

    def test_rejects_invalid_inputs(self):
        for sample, window in ((self.series, 1), (self.series, 25), (self.series[:-1], 24),
                               ([math.nan] + self.series[1:], 24)):
            with self.subTest(window=window), self.assertRaises(ValueError):
                backtest(sample, self.design, 7, window=window)

    def test_coverage_width_mae_and_regime_boundary(self):
        rows = [Forecast(7, 139, "a", "fixed", 5, 4, 6, 6),
                Forecast(7, 140, "a", "fixed", 5, 4, 6, 8)]
        summary = {r["phase"]: r for r in summarize(rows, self.design)}
        self.assertEqual(summary["pre_shift"]["coverage"], 1)
        self.assertEqual(summary["post_shift"]["coverage"], 0)
        self.assertEqual(summary["post_shift"]["mean_width"], 2)
        self.assertEqual(summary["post_shift"]["mae"], 3)


if __name__ == "__main__":
    unittest.main()
