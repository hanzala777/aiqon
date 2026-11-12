# Port Congestion & Delay Risk

Predicts, **48 hours before a vessel's scheduled arrival**, whether the port call will lose more than 12 hours against plan. For every call the system returns:

1. a **calibrated probability** of congestion,
2. a **confidence score** (how far to trust that probability: model agreement, data completeness, similarity to past situations),
3. the **factors driving the prediction** in plain language, with a **recommended action** and an expected anchorage-wait range.

> Aiqon take-home. Technical report: [`reports/report.pdf`](reports/report.pdf) (source: [`reports/report.md`](reports/report.md)).

```
HIGH congestion risk (>99%, confidence 63%). Main drivers: heavy traffic: 22 vessels due in the 48h before
ETA, 6 waiting at anchorage, 6/6 berths occupied, 7-day utilisation 92%; storm risk: forecast wind 35 kn
(cranes stop above 32 kn); ... Expected anchorage wait ~38h (80% range 9-48h).
Recommended actions:
  * Request a later berth window or slow-steam to arrive after the peak ...
  * Add a weather buffer to the customer ETA ...
Caveats: estimate is unstable: models refitted on resampled history disagree; situation is unlike the
training data (extrapolation)
```

The risk is near-certain, but confidence is only moderate: a 22-vessel storm peak is rarer than anything in the training year, so the *size* of the risk is an extrapolation.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_all.py               # simulate -> train -> evaluate -> robustness experiments (~3 min)
streamlit run app.py            # interactive demo
pytest -q                       # 27 tests: leakage, validation, monotonic risk, confidence, CLI, app
```

The generated data, the trained model and the reports are committed, so `streamlit run app.py` works without running the pipeline first. Everything is seeded (`config.yaml: seed: 42`), so `run_all.py` reproduces every number in the report.

Single prediction from the command line (any live input you leave out is treated as a feed outage; impossible values such as a negative queue, utilisation above 1 or more occupied berths than the port has are rejected with an error):

```bash
python -m src.predict --port MID_B --eta "2025-10-20 06:00" --teu 4200 --queue 6 --occupied 6 \
    --ahead 22 --window 20 --util7d 0.92 --wind 35          # add --json for machine-readable output
```

## Results

**Primary estimate: rolling-origin evaluation.** The whole system is refitted 5 times on the past and predicts the next 10% of the timeline each time, giving **250 out-of-time calls** (30% congested). That is four times more evaluation data than a single hold-out. The production model's final hold-out (most recent 75 calls) is shown for reference.

| Model | Rolling-origin ROC-AUC [95% CI] (250 calls) | PR-AUC | Brier | Hold-out AUC (75 calls) |
|---|---|---|---|---|
| **Logistic regression (production)** | **0.85** [0.79–0.90] | **0.76** | **0.13** | 0.87 |
| LightGBM (challenger) | 0.81 [0.74–0.86] | 0.72 | 0.15 | 0.82 |
| Rule of thumb (demand/berth > 3.5 or storm) | 0.81 [0.75–0.86] | 0.62 | – | 0.83 |

- **Cheaper decisions than the rule of thumb.** At the expected-cost threshold (0.17), the model catches 88% of congested calls. Its total cost is 236 against 350 for the rule (never acting costs 750); the 95% interval for the difference is [−226, −14].
- **The confidence score is informative.** High-confidence calls have well-calibrated probabilities (miscalibration, measured as excess Brier, is −0.015 in the top tercile vs +0.021 in the bottom; Brier 0.08 vs 0.18). Confidence is not a restatement of the risk: its rank correlation with |p − 0.5| is 0.26, and 30–70% risk calls range from 42 to 85 confidence.
- **Stable across simulated worlds.** Across 5 other simulated years the rolling-origin AUC is **0.81 ± 0.05** (worst 0.72), and seed 42's 0.85 lies within one standard deviation. Details are in the report, §5.

## How it works

| Step | File | What it does |
|---|---|---|
| Simulate | [`src/simulate.py`](src/simulate.py) | Hour-by-hour multi-berth FCFS queue for 5 fictional ports over one year (≈10.7k calls): seasonal and surge arrivals, ETA noise, storms halting cranes, labour action, night shifts. 500 calls are sampled with what a planner could observe at ETA − 48 h, plus the realised outcome. |
| Ground truth | `simulate.py` | **Excess port time** = anchorage wait + berth overrun beyond planned service time. **Congested = excess > 12 h** (sensitivity checked at 6 h and 24 h). |
| Features | [`src/features.py`](src/features.py) | Input validation, then leakage-safe features: traffic pressure, capacity, weather *forecast*, labour alert, port history, calendar. Imputation is fitted on training data only. |
| Model | [`src/model.py`](src/model.py) | Sign-constrained, Platt-calibrated logistic regression (production), LightGBM challenger, 20-model bootstrap ensemble with multiple imputation of missing feeds, k-NN familiarity check, conformalised quantile intervals for wait time, expected-cost decision threshold. |
| Explain | [`src/explain.py`](src/explain.py) | Exact per-feature log-odds contributions (equal to SHAP for a linear model), summed into operational factors and turned into sentences and actions. |
| Evaluate | [`src/evaluate.py`](src/evaluate.py), [`src/experiments.py`](src/experiments.py) | Pooled rolling-origin evaluation of the full system (primary), 75-call hold-out, bootstrap CIs, calibration, per-port and storm breakdowns, confidence validation, cost curve, label-threshold and data-quality stress tests, seed variance. |
| Serve | [`src/predict.py`](src/predict.py), [`app.py`](app.py) | CLI / JSON output and a Streamlit demo (single prediction, port comparison, test-set explorer, model performance). |

Configuration and every modelling assumption live in [`config.yaml`](config.yaml).

## Key design decisions

- **Synthetic data from a queueing mechanism**, not `label = f(features) + noise`. Congestion emerges from berth occupancy, so the model has to learn from noisy, partial observations, much as it would on real data. Turning raw AIS pings into labelled port calls would have used the whole three-day budget.
- **No leakage.** Every feature is available at ETA − 48 h. A test checks that scrambling all outcome columns leaves the feature matrix unchanged.
- **Simple model by evidence.** Logistic regression beat LightGBM on every rolling-CV fold. With about 350 training rows the trees overfit.
- **Domain knowledge as constraints.** More traffic, wind, labour action, delay history or vessel size can never lower the predicted risk: those coefficients are constrained to be ≥ 0. Unconstrained, correlated inputs picked up wrong signs ("more berths occupied → lower risk"; vessel size acting as a proxy for port size) at no gain in accuracy.
- **Confidence is not the probability.** "50% risk, 95% confidence" means a genuine coin flip. "50% risk, 40% confidence" means refresh the data first. Model agreement is measured on the log-odds scale (so it does not simply shrink near 50%), and a missing feed is replaced by many plausible historical values, so losing data always widens the spread.
- **Decisions, not just scores.** Act when risk ≥ c_fp / (c_fp + c_fn) = 0.17, the expected-cost optimum for calibrated probabilities with a missed delay costing 5× a false alarm (configurable). It does not depend on the 75-call validation slice, where the empirical optimum swung between 0.06 and 0.39.

## Limitations (full list in the report)

- **All data is simulated.** This shows the method works on a plausible world, not that it reaches this accuracy on a real port.
- **Simplified port operations:** first-come-first-served only, no tides, pilotage or yard congestion, and carriers never divert away from congested ports.
- **Sign constraints are a prior.** If one of the assumed directions were wrong for a real port, the model could not learn it (trained on shuffled labels it still scores AUC 0.64 from the prior alone).
- **Small sample.** 500 records (the brief's upper bound). Rolling-origin evaluation uses 250 of them out of time, but the final 75-call hold-out alone has wide intervals (AUC 0.76–0.96).
- **Year-end shift in wait times.** Conformal calibration brings the P10–P90 wait interval to 80% coverage overall, but only 62% in the last (December) fold.
- **Storms at quiet ports are under-predicted.** Storm examples are rare in training.

## Repository layout

```
config.yaml            assumptions: ports, disruptions, label, horizon, costs
run_all.py             one-command reproduction
app.py                 Streamlit demo
src/                   simulate, features, model, train, explain, predict, evaluate, experiments
tests/                 pytest suite
scripts/               make_report_pdf.py (report.md -> report.pdf)
data/                  port_calls.csv (500 labelled calls) + simulation_meta.json
models/                model.joblib
reports/               report.md/.pdf, metrics.json, experiments.json, figures/
```
