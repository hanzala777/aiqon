# Port Congestion & Delay Risk

Predicts, **48 hours before a vessel's scheduled arrival**, whether the port call will lose more than 12 hours against plan. For every call the system returns:

1. a **calibrated probability** of congestion,
2. a **confidence score** (how far to trust that probability: model agreement, data completeness, similarity to past situations),
3. the **factors driving the prediction** in plain language, with a **recommended action** and an expected anchorage-wait range.

> Aiqon AI Internship take-home. Technical report: [`reports/report.pdf`](reports/report.pdf) (source: [`reports/report.md`](reports/report.md)).

```
HIGH congestion risk (>99%, confidence 90%). Main drivers: heavy traffic: 22 vessels due in the 48h before
ETA, 6 waiting at anchorage, 6/6 berths occupied (7-day utilisation 92%); storm risk: forecast wind 35 kn
(cranes stop above 32 kn). Expected anchorage wait ~38h (80% range 9-48h).
Recommended actions:
  * Request a later berth window or slow-steam to arrive after the peak ...
  * Add a weather buffer to the customer ETA ...
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_all.py               # simulate -> train -> evaluate -> robustness experiments (~1 min)
streamlit run app.py            # interactive demo
pytest -q                       # 8 tests: leakage guard, ranges, reproducibility, explanation sanity
```

The generated data, the trained model and the reports are committed, so `streamlit run app.py` works without running the pipeline first. Everything is seeded (`config.yaml: seed: 42`), so `run_all.py` reproduces every number in the report.

Single prediction from the command line (any live input you leave out is treated as a feed outage):

```bash
python -m src.predict --port MID_B --eta "2025-10-20 06:00" --teu 4200 --queue 6 --occupied 6 \
    --ahead 22 --window 20 --util7d 0.92 --wind 35          # add --json for machine-readable output
```

## Results (test set = most recent 75 calls, chronological split)

| Model | ROC-AUC [95% CI] | PR-AUC | Brier | Rolling-origin CV AUC (5 folds) |
|---|---|---|---|---|
| **Logistic regression (production)** | **0.87** [0.76–0.96] | **0.81** | **0.11** | **0.86 ± 0.05** |
| LightGBM (challenger) | 0.82 [0.70–0.93] | 0.71 | 0.14 | 0.81 ± 0.07 |
| Rule of thumb (demand/berth > 3.5 or storm) | 0.83 [0.72–0.93] | 0.70 | – | 0.81 ± 0.04 |

- **The confidence score is informative.** Accuracy is 76% / 88% / 96% in the low / medium / high confidence terciles.
- **Read these with caution.** Across 5 other simulated years the production model averages AUC **0.79 ± 0.06**, so seed 42 is a favourable draw. The operational-cost comparison with the rule of thumb is inconclusive at this sample size. Details are in the report, §5.

## How it works

| Step | File | What it does |
|---|---|---|
| Simulate | [`src/simulate.py`](src/simulate.py) | Hour-by-hour multi-berth FCFS queue for 5 fictional ports over one year (≈10.7k calls): seasonal and surge arrivals, ETA noise, storms halting cranes, labour action, night shifts. 500 calls are sampled with what a planner could observe at ETA − 48 h, plus the realised outcome. |
| Ground truth | `simulate.py` | **Excess port time** = anchorage wait + berth overrun beyond planned service time. **Congested = excess > 12 h** (sensitivity checked at 6 h and 24 h). |
| Features | [`src/features.py`](src/features.py) | Leakage-safe features: traffic pressure, capacity, weather *forecast*, labour alert, port history, calendar. Imputation is fitted on training data only. |
| Model | [`src/model.py`](src/model.py) | Platt-calibrated logistic regression (production), LightGBM challenger, 20-model bootstrap ensemble, k-NN familiarity check, quantile models for wait time, cost-optimal decision threshold. |
| Explain | [`src/explain.py`](src/explain.py) | Exact per-feature log-odds contributions (equal to SHAP for a linear model), summed into operational factors and turned into sentences and actions. |
| Evaluate | [`src/evaluate.py`](src/evaluate.py), [`src/experiments.py`](src/experiments.py) | Bootstrap CIs, rolling-origin CV, calibration, per-port and storm breakdowns, confidence validation, cost curve, label-threshold and data-quality stress tests, seed variance. |
| Serve | [`src/predict.py`](src/predict.py), [`app.py`](app.py) | CLI / JSON output and a Streamlit demo (single prediction, port comparison, test-set explorer, model performance). |

Configuration and every modelling assumption live in [`config.yaml`](config.yaml).

## Key design decisions

- **Synthetic data from a queueing mechanism**, not `label = f(features) + noise`. Congestion emerges from berth occupancy, so the model has to learn from noisy, partial observations, much as it would on real data. Turning raw AIS pings into labelled port calls would have used the whole three-day budget.
- **No leakage.** Every feature is available at ETA − 48 h. A test checks that scrambling all outcome columns leaves the feature matrix unchanged.
- **Simple model by evidence.** Logistic regression beat LightGBM on every rolling-CV fold. With about 350 training rows the trees overfit.
- **Confidence is not the probability.** "50% risk, 95% confidence" means a genuine coin flip. "50% risk, 40% confidence" means refresh the data first.
- **Decisions, not just scores.** The alert threshold minimises expected cost, with a missed delay costing 5× a false alarm (configurable).

## Limitations (full list in the report)

- **All data is simulated.** This shows the method works on a plausible world, not that it reaches this accuracy on a real port.
- **Simplified port operations:** first-come-first-served only, no tides, pilotage or yard congestion, and carriers never divert away from congested ports.
- **Small sample.** 500 records mean wide confidence intervals (75 test calls).
- **Under-covering wait intervals.** The P10–P90 interval covers only 67% of actual waits, against an 80% target.
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
