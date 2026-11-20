# Forecast Interval Lab

How quickly do forecast intervals recover when a seasonal series becomes more volatile? This experiment compares two ways of calibrating a one-step prediction interval. One keeps its initial error threshold. The other updates its threshold after every new observation. Both wrap the same point forecasts.

## Results

Across five simulated series, the rolling interval around the autoregressive forecast covered **82.5%** of outcomes after the shift. The fixed interval covered **42.3%**. This increase came with wider intervals, and the rolling method still fell short of the 90% target.

| Point forecast | Calibration | Before shift: coverage / mean width | After shift: coverage / mean width |
| --- | --- | ---: | ---: |
| Seasonal autoregression | Fixed | 90.9% / 3.42 | 42.3% / 3.42 |
| Seasonal autoregression | Rolling | 90.0% / 3.34 | 82.5% / 8.63 |
| Seasonal naive | Fixed | 89.8% / 5.59 | 42.4% / 5.59 |
| Seasonal naive | Rolling | 89.6% / 5.57 | 81.4% / 14.57 |

Each column summarizes 1,280 one-step outcomes per method (five seeds × 256 origins). The autoregression's mean absolute error rose from 0.78 to 2.37 after the shift. The seasonal naive baseline rose from 1.36 to 4.00. All widths and errors are in the units of the simulated series. The committed [pooled results](results/summary.csv), [per-seed results](results/per_seed.csv) and [run configuration](results/config.json) give the unrounded values and experimental settings.

## Reproduce the experiment

The reference run uses Python 3.12 and NumPy 2.3.5. From this directory, run:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python forecast_intervals.py --out run --save-forecasts
```

The optional `run/forecasts.csv` records the seed, time, model, calibration method, point prediction, interval endpoints and observed value. For a sensitivity analysis using a shorter calibration window, run `python forecast_intervals.py --window 32 --out run/window32`. Each calibration method starts with the same 32 residuals in that run. Use `--coverage` and `--seeds` to change the target and repetitions.

## Data and evaluation

Each synthetic series contains 1,024 observations with a linear trend, a 24-step seasonal cycle and a second harmonic. A disturbance follows an AR(1) process with coefficient 0.55. At time 768 the Gaussian innovation standard deviation increases from one to three. The evaluation uses the known shift to group results.

1. **Train, times 0–383.** Fit a least-squares forecaster with trend, one observed lag, and sine/cosine pairs at the first two harmonics. Freeze the coefficients. The seasonal naive comparator predicts the observed value from 24 steps ago.
2. **Calibrate, times 384–511.** Collect each model's absolute one-step forecast errors on the subsequent block. Initialize both calibration methods with the last 128 errors.
3. **Evaluate, times 512–1023.** At each origin, produce the point prediction and both intervals. Once the observation arrives, record coverage and replace the oldest rolling error with the new absolute error. The fixed threshold remains unchanged. The series is fully observed between successive predictions.

For a requested coverage `p` and `n` resolved errors, the radius is the error at one-based rank `ceil((n + 1) * p)` in ascending order. An unachievable rank gives an infinite radius. The CLI rejects settings that would give only unbounded intervals. Empirical coverage counts outcomes inside the closed interval, including the endpoints. A regression test perturbs the current target and every future observation and checks that every forecast and interval up to that origin is unchanged.

## Interpretation

The rolling sample initially includes errors from the quieter regime. Its interval widens as larger errors enter the window. The finite-sample split conformal construction has a coverage guarantee under exchangeability. Serial dependence and the deliberate change in noise scale make coverage an empirical quantity in this experiment. The results describe one synthetic process with a known seasonal period.

For the finite-sample construction see [Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*](https://arxiv.org/abs/2107.07511). For an alternative approach that changes the target miscoverage rate online, see [Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*](https://papers.nips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
