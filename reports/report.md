# Predicting Port Congestion and Shipment Delays

**Aiqon AI Internship – Technical Report** · Hanzala Ahmed · November 2026

## 1. Problem framing

**Decision supported.** 48 hours before a vessel's scheduled arrival, a planner (carrier, forwarder or terminal) must decide whether to act: re-time the arrival, divert, or warn the customer. The system therefore answers one question per port call: *will this call lose more than 12 hours against plan?* It returns a probability, a confidence score, the factors behind it, and a suggested action.

**Definition of congestion (ground truth).** *Excess port time* = anchorage waiting time + berth overrun beyond the terminal's planned service time. A call is **congested if excess time > 12 h**. I first labelled calls by "turnaround > 1.5× port median". I dropped it because it flagged large ships simply for being large: a long *planned* stay is not congestion. Sensitivity to the 12 h threshold is reported in §5.

**Prediction horizon and leakage.** Every feature must be known at *ETA − 48 h*. Realised quantities (actual arrival, storm during the stay, wait time) are used only to build labels. A unit test checks this: scrambling every outcome column leaves the feature matrix unchanged.

## 2. Data: a queueing simulation, not a formula

Turning raw AIS position pings into labelled port calls (geofencing anchorages, splitting calls, deriving waits) is a project in its own right. I chose a synthetic dataset instead and made it **mechanistic**: congestion *emerges* from a simulated queue rather than being drawn from `label = f(features) + noise`.

- **5 fictional ports** (3–12 berths) with different vessel sizes, crane productivity and regional weather, simulated hour by hour for one year (**10,723 calls**). **500 calls** are sampled as the modelling dataset (23% congested).
- **Arrivals:** a Poisson process with seasonal (Aug–Nov peak), weekday and *schedule-surge* (bunching) effects. Actual arrival = scheduled ETA + N(0, 6 h).
- **Service:** first-come-first-served berths. Work = 35% of TEU capacity ÷ port productivity, plus 4 h. Productivity drops to 0 in storms (wind > 32 kn, the crane limit), 0.6 during labour action, and 0.8 at night.
- **Imperfect observations:** the model sees a wind *forecast* (± 4 kn error), not the actual weather. Labour alerts have 85% recall and a 2% false-alarm rate. 5% of live-feed values are missing at random (AIS / forecast outages).

*A bug I found and fixed:* my first calibration used ±20% seasonal swings. That pushed the small ports to near-100% utilisation, so their queues stayed unstable for months and 84% of the test period was congested. The task collapsed into "predict the month". I damped the seasonality to ±12% and checked queue stability across several random seeds.

## 3. Features

| Group | Features (all known at ETA − 48 h) | Real-world source |
|---|---|---|
| Traffic / capacity | vessels at anchorage, berths occupied, 7-day utilisation, arrivals scheduled before ETA and within ±24 h, **pressure index** = (queue + occupied + scheduled-ahead) / berths | AIS, terminal systems, pre-arrival notices |
| Weather / labour | forecast max wind, storm-forecast flag, labour-action alert | forecast feed, port authority |
| Port & vessel | berths, port identity, 30-day delay rate, vessel TEU | port statistics |
| Time | hour, weekend, month (sin/cos), peak-season flag | schedule |

Missing values are filled with the *training-set* median. Missing-value indicator columns were tested and removed: missingness here is random, so they added noise (CV AUC 0.859 vs 0.863) and appeared as spurious "drivers". Missing data is surfaced through the confidence score instead. Impossible inputs (negative counts, utilisation above 1, more occupied berths than the port has, unknown ports, unparseable ETAs) are rejected with an error rather than scored.

## 4. Models, confidence and explanations

- **Split:** chronological, with 350 calls for training (Jan–Sep), 75 for validation and 75 for testing (Nov–Dec). This mimics deployment: train on the past, predict the future.
- **Candidates:** (a) a *rule of thumb*: flag if pressure index > 3.5 vessels per berth or a storm is forecast, with the threshold tuned on training data; (b) L2-regularised **logistic regression**; (c) **LightGBM**. Probabilities are Platt-calibrated on the validation slice.
- **Choice: logistic regression with domain sign constraints.** It beat LightGBM on every fold of a 5-fold rolling-origin evaluation (AUC 0.86 vs 0.81). With 350 training rows the trees overfit, and the data shows little non-additive structure: in Fig. 2 (centre), a storm forecast raises congestion at every traffic level, which a linear logit captures. Coefficients of traffic, wind, labour, delay-history and vessel-size features are constrained to be ≥ 0. Unconstrained, correlated inputs took wrong signs ("more berths occupied → lower risk"; vessel size, a proxy for port size, gave "bigger ship → lower risk", although a 9,000-call check shows no within-port size effect), with no accuracy gain. LightGBM stays in the code as a challenger, together with its SHAP interaction analysis.
- **Confidence (0–100)** is kept separate from the probability. It answers "how much should I trust this number?", not "how sure is the event?". It combines **50% model agreement**, **25% data completeness** (share of live feeds actually observed) and **25% familiarity** (k-NN distance to the training data, flagging extrapolation). Agreement is 1 − the spread, *in log-odds*, of 20 bootstrap-refitted models. On the probability scale the spread shrinks mechanically near 0 and 1, which made confidence a restatement of the risk. A missing feed is replaced by 25 plausible values from the port's history (multiple imputation), so losing data widens the spread. A 50% risk with 95% confidence means "genuinely a coin flip". A 50% risk with 40% confidence means "refresh the data first".
- **Explanations:** for a linear model, coefficient × standardised value is the exact SHAP value. Contributions are summed into operational factors (traffic, weather, labour, port, capacity, vessel, season, timing). Each factor becomes a sentence with real numbers, and the top factors map to actions.
- **Decision threshold** = c_fp / (c_fp + c_fn) = 2 / 12 = 0.17, the expected-cost optimum for calibrated probabilities, with a missed delay costing 10 and a false alarm costing 2 (illustrative weights; they should come from the user's real demurrage and penalty figures). Picking the threshold empirically on the 75 validation calls was unstable: under bootstrap resampling it ranged from 0.06 to 0.39.
- **Expected wait:** LightGBM quantile regressors (P10 / P50 / P90), conformally calibrated on the validation slice to target 80% coverage.

Example output: *"HIGH congestion risk (>99%, confidence 63%). Main drivers: heavy traffic: 22 vessels due in the 48 h before ETA, 6 waiting at anchorage, 6/6 berths occupied, 7-day utilisation 92%; storm risk: forecast wind 35 kn (cranes stop above 32 kn). Expected anchorage wait ~38 h (80% range 9–48 h). → Request a later berth window / slow-steam; add a weather buffer to the customer ETA."* Confidence is moderate because a 22-vessel storm peak lies outside the training experience.

## 5. Results

**Primary evaluation: rolling origin.** A single 75-call test set is too small to compare models, costs or ports (one missed delay moves the cost by 10). The whole system is therefore refitted 5 times on the past (fit and calibration slices in the production proportions) and scored on the next 10% of the timeline, giving **250 out-of-time calls** (30% congested). The production model's final hold-out (75 calls, Nov–Dec) is reported alongside.

| Rolling origin (250 calls) | ROC-AUC [95% CI] | PR-AUC | Brier | Precision / Recall @ 0.17 | Hold-out AUC (75) |
|---|---|---|---|---|---|
| **Logistic regression (production)** | **0.85** [0.79–0.90] | **0.76** | **0.13** | 0.48 / 0.88 | 0.87 [0.76–0.96] |
| LightGBM (challenger) | 0.81 [0.74–0.86] | 0.72 | 0.15 | 0.45 / 0.83 | 0.82 |
| Rule of thumb | 0.81 [0.75–0.86] | 0.62 | – | 0.64 / 0.60 | 0.83 |

![Performance](figures/performance.png)

*Figure 1 – Discrimination and calibration on the 250 rolling-origin calls.*

![Explanations](figures/explanations.png)

*Figure 2 – Left: global factor importance (production model). Centre: observed congestion rate by traffic level and storm forecast (all 500 calls). Right: Brier and miscalibration by confidence tercile (rolling origin).*

**What the results show:**

- **Cheaper decisions than the rule of thumb.** At the 0.17 threshold the model catches 88% of congested calls. Its operational cost is 236 against 350 for the rule (never acting: 750); the 95% bootstrap interval for the difference, [−226, −14], excludes zero. On the 75-call hold-out alone the two are indistinguishable (66 vs 64, interval [−36, +40]).
- **Drivers match the simulation.** Global factor importance ranks traffic first (1.9), then port (0.7), weather (0.5) and capacity (0.5), with labour weak (only about 2 short episodes per port). The vessel-size coefficient is held at zero by its constraint.
- **The confidence score carries information.** Across the low, medium and high confidence terciles, Brier falls from 0.18 to 0.14 to 0.08 and miscalibration (excess Brier = Brier − mean p(1 − p)) from +0.021 to +0.003 to −0.015 (Fig. 2, right). It is not a restatement of the risk: rank correlation with |p − 0.5| is 0.26, and 30–70% risk calls range from 42 to 85 confidence. Accuracy at the threshold is *not* monotonic in confidence (64%, 63%, 75%), as expected: accuracy mostly reflects how extreme p is.
- **Per port** (rolling origin), AUC ranges from 0.72 (SML_B, 27 calls) to 0.91; the hub, with 91 calls and 21 congested, scores 0.86. Its hold-out AUC of 0.60 rests on 3 congested calls.
- **Wait-time estimate.** MAE is 4.9 h on the hold-out, compared with 8.1 h for predicting the median. The conformal P10–P90 interval covers 80% of waits in the rolling evaluation, but only 62% in the last (December) fold and 63% on the hold-out: wait times shift at year-end.
- **Label threshold.** Hold-out AUC is 0.83 at 6 h, 0.87 at 12 h and 0.93 at 24 h. Long delays are easier to anticipate than marginal ones.
- **Data quality.** With 15% of feed values missing, hold-out AUC stays at 0.88; with ETA noise doubled to 12 h, it falls to 0.84. Average confidence drops (78 → 75 with both).
- **Error analysis: storms at quiet ports.** All 8 hold-out calls hit by a storm were congested, and all 8 storms had been correctly forecast (about 35 kn). The model caught 6. The 2 misses (risk 2% and 7%) were at ports with *empty anchorages*, where light traffic outweighed the storm in the model. Yet a crane shutdown of 12 h or more alone pushes a berthed ship past the threshold. Storm-forecast calls are rare in training (about 8%), so the storm effect is underestimated. More storm history, or a rule that never lets risk fall below a floor when the forecast exceeds 32 kn, would fix this.

**Honest caveats on these numbers:**

1. **Seed 42 is a typical world.** Re-running the whole pipeline on 5 other simulated years gives a rolling-origin AUC of **0.81 ± 0.05** (worst 0.72); seed 42's 0.85 lies within one standard deviation. On 75-call hold-outs alone the spread is wider (logistic regression 0.80 ± 0.08, LightGBM 0.78 ± 0.09, rule 0.73 ± 0.07), and LightGBM won two of the five (one by 0.001).
2. **The hold-out remains small.** Its AUC interval (0.76–0.96) is twice as wide as the rolling-origin one, so model selection relies on the rolling evaluation.

## 6. Assumptions and limitations

- **Everything is synthetic.** The results show the *method* works on a plausible world whose rules I wrote. They are not evidence of accuracy on a real port. The model will recover structure I put in, and real ports have mechanisms I left out.
- **Simplified port operations:** first-come-first-served only (real terminals have priority windows and berth contracts), no tides or pilotage, no yard or gate congestion, no knock-on effects between ports, and no carriers diverting away from congested ports (which in reality limits queue growth).
- **The sign constraints are a prior.** They encode "more traffic, wind, labour action or delay history never lowers risk". The prior is strong: trained on *shuffled* labels the constrained model still scores AUC 0.64, against 0.49 unconstrained. If a sign were wrong for a real port, the model could not learn it, so the constraints must be revisited when moving to real data.
- **Small data.** 500 records (the brief's upper bound). The rolling-origin evaluation makes the comparisons meaningful, but model choice and thresholds should be revalidated on more data.
- **Year-end shift in wait times.** Conformal calibration fixes coverage on average (80%) but not in December (62%). A calibration window that tracks the season, or adaptive conformal prediction, would be needed in production.

## 7. Next steps toward production

1. **Real data pipeline:** geofence anchorage and berth polygons in AIS (MarineCadastre), derive waiting times from vessel positions, and join weather forecasts and port-call schedules.
2. Retrain per port with **rolling-window retraining** and drift monitoring, and add a *carrier-diversion* feedback loop to the simulator.
3. **Calibrate the costs with users** (demurrage, missed rail connections), and make the conformal wait intervals season-aware.
4. Ship as an API that serves the JSON output (`python -m src.predict --json`) to planning dashboards.

*Code, reproduction instructions and the demo are in the repository (`python run_all.py` reproduces every number in this report in about three minutes).*
