"""One-step prediction intervals for a seasonal series with a volatility shift."""

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean

import numpy as np


@dataclass(frozen=True)
class Design:
    period: int = 24
    train_end: int = 384
    test_start: int = 512
    shift: int = 768
    length: int = 1024

    def __post_init__(self):
        if not 2 < self.period < self.train_end < self.test_start < self.shift < self.length:
            raise ValueError("Require 2 < period < train_end < test_start < shift < length.")


def simulate(seed, design):
    """Seasonal signal plus autocorrelated noise with a known scale change."""
    rng = np.random.default_rng(seed)
    t = np.arange(design.length)
    angle = 2 * np.pi * t / design.period
    mean = 40 + 0.008 * t + 5 * np.sin(angle) + 1.5 * np.cos(2 * angle)
    innovations = rng.normal(size=design.length) * np.where(t < design.shift, 1.0, 3.0)
    disturbance = 0.0
    values = []
    for signal, innovation in zip(mean, innovations):
        disturbance = 0.55 * disturbance + innovation
        values.append(float(signal + disturbance))
    return values


def features(t, previous, period):
    angle = 2 * math.pi * t / period
    return (1.0, t / period, previous, math.sin(angle), math.cos(angle),
            math.sin(2 * angle), math.cos(2 * angle))


def conformal_radius(scores, coverage):
    """Finite-sample order statistic, with an infinite upper sentinel."""
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one.")
    ordered = sorted(scores)
    if not ordered or any(not math.isfinite(x) or x < 0 for x in ordered):
        raise ValueError("Supply at least one finite, nonnegative residual.")
    rank = math.ceil((len(ordered) + 1) * Fraction(str(coverage)))
    return math.inf if rank > len(ordered) else ordered[rank - 1]


@dataclass(frozen=True)
class Forecast:
    seed: int
    time: int
    model: str
    calibration: str
    forecast: float
    low: float
    high: float
    observed: float


def backtest(series, design, seed, coverage=0.9, window=128):
    """Fit once, then predict each next observation before scoring the outcome."""
    if len(series) != design.length or not all(math.isfinite(x) for x in series):
        raise ValueError("Expected a complete finite series of the specified length.")
    if not 1 <= window <= design.test_start - design.train_end:
        raise ValueError("Window must fit inside the initial calibration block.")
    if math.isinf(conformal_radius([0.0] * window, coverage)):
        raise ValueError("Window too small for finite intervals at this coverage.")

    train = series[:design.train_end]
    matrix = [features(t, train[t - 1], design.period) for t in range(1, len(train))]
    coefficients = np.linalg.lstsq(matrix, train[1:], rcond=None)[0]
    outputs = []
    for model in ("seasonal_ar", "seasonal_naive"):
        last_scores = deque(maxlen=window)
        fixed_radius = None
        for t in range(design.train_end, design.length):
            if model == "seasonal_ar":
                estimate = float(np.dot(coefficients, features(t, series[t - 1], design.period)))
            else:
                estimate = series[t - design.period]
            if t >= design.test_start:
                rolling_radius = conformal_radius(last_scores, coverage)
                if t == design.test_start:
                    fixed_radius = rolling_radius
                for calibration, radius in (("fixed", fixed_radius), ("rolling", rolling_radius)):
                    outputs.append(Forecast(seed, t, model, calibration, estimate,
                                            estimate - radius, estimate + radius, series[t]))
            last_scores.append(abs(series[t] - estimate))
    return outputs


def summarize(rows, design):
    """Pool observations at each model/calibration/regime for matched seeds."""
    groups = defaultdict(list)
    for row in rows:
        phase = "pre_shift" if row.time < design.shift else "post_shift"
        groups[row.model, row.calibration, phase].append(row)
    results = []
    for (model, calibration, phase), sample in sorted(groups.items()):
        results.append(dict(
            model=model, calibration=calibration, phase=phase, n=len(sample),
            coverage=fmean(row.low <= row.observed <= row.high for row in sample),
            mean_width=fmean(row.high - row.low for row in sample),
            mae=fmean(abs(row.observed - row.forecast) for row in sample),
        ))
    return results


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("run"))
    parser.add_argument("--save-forecasts", action="store_true",
                        help="Write every one-step forecast alongside the pooled summary.")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct nonnegative seeds.")
    design = Design()
    rows = []
    seed_summaries = []
    try:
        for seed in args.seeds:
            forecasts = backtest(simulate(seed, design), design, seed, args.coverage, args.window)
            rows.extend(forecasts)
            seed_summaries.extend({"seed": seed, **entry} for entry in summarize(forecasts, design))
    except ValueError as error:
        parser.error(str(error))
    summary = summarize(rows, design)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"design": asdict(design), "seeds": args.seeds, "target_coverage": args.coverage,
              "calibration_window": args.window, "numpy_version": np.__version__}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "per_seed.csv", seed_summaries)
    if args.save_forecasts:
        write_csv(args.out / "forecasts.csv", [asdict(row) for row in rows])
    print(f"{'point model':<15} {'interval':<9} {'regime':<11} {'n':>5} {'covered':>8} {'width':>8} {'MAE':>8}")
    for item in summary:
        print("{model:<15} {calibration:<9} {phase:<11} {n:>5} {coverage:>8.1%} "
              "{mean_width:>8.2f} {mae:>8.2f}".format(**item))


if __name__ == "__main__":
    main()
