"""Synthetic port-call generator based on an hourly multi-berth FCFS queue simulation.

Why a queue simulation (instead of `label = f(features) + noise`)?
Waiting time *emerges* from berth occupancy, arrival bunching, weather shutdowns and labour
action -- the same mechanism that produces congestion in real ports. The model therefore has to
learn the relationship from noisy, partially observed signals, just as it would on real data.

World model (per port, simulated hour by hour for one year):
  * Scheduled arrivals follow a non-homogeneous Poisson process:
        rate(t) = base * season(month) * day_of_week * surge(t)
    `base` is set so that average berth utilisation is close to the port's `target_util`.
  * Actual arrival = scheduled ETA + N(0, eta_noise_sd) (ETA uncertainty).
  * Service work (berth-hours) = 0.35 * TEU / port productivity * LogNormal noise + 4h
    (about 35% of a ship's capacity is exchanged per call; 4h for mooring/unmooring).
  * Productivity per hour: 0 during storms (crane wind limit), 0.6 during labour action,
    0.8 at night, otherwise 1.
  * Queue discipline: first-come-first-served at anchorage; no new berthing during storms.

For each port call we record (a) what a planner can observe 48h before the scheduled ETA
-> model features, and (b) what actually happened -> labels/outcomes (never used as features).
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Peak season Aug-Nov, post-Lunar-New-Year lull in Feb. Amplitude deliberately moderate: with
# +/-20% swings the smaller ports' queues became unstable for months (rho -> 1), which turned
# the problem into "predict the month" -- an early version of the simulator had exactly this bug.
SEASON = {1: 0.97, 2: 0.91, 3: 0.94, 4: 0.97, 5: 1.00, 6: 1.00,
          7: 1.03, 8: 1.09, 9: 1.12, 10: 1.12, 11: 1.06, 12: 1.00}
DOW = {0: 1.05, 1: 1.05, 2: 1.05, 3: 1.05, 4: 1.05, 5: 0.88, 6: 0.87}  # Mon=0, quieter weekends

# Columns that describe the realised outcome. They must never be used as model inputs.
OUTCOME_COLUMNS = ["actual_arrival", "berth_start", "departure", "planned_service_hours", "wait_hours",
                   "berth_hours", "turnaround_hours", "excess_hours", "storm_during_stay",
                   "labor_during_stay", "congested"]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _simulate_weather(rng, days, regions, s):
    """Daily max wind (knots) per region: winter-windy seasonal mean + AR(1) weather systems."""
    doy = np.arange(days)
    out = {}
    for r in regions:
        mean = 16 + 6 * np.cos(2 * np.pi * (doy - 15) / 365)
        x = np.zeros(days)
        for d in range(1, days):
            x[d] = 0.6 * x[d - 1] + rng.normal(0, 6)
        wind = np.clip(mean + x, 0, None)
        forecast = np.clip(wind + rng.normal(0, s["forecast_wind_sd_knots"], days), 0, None)
        hourly_storm = np.zeros(days * 24, dtype=bool)
        for d in range(days):
            if wind[d] > s["severe_storm_wind_knots"]:
                hourly_storm[d * 24:(d + 1) * 24] = True
            elif wind[d] > s["storm_wind_knots"]:
                hourly_storm[d * 24 + 8:d * 24 + 20] = True  # 12-hour shutdown
        out[r] = {"wind": wind, "forecast": forecast, "storm_h": hourly_storm}
    return out


def _episodes(rng, days, n, min_len, max_len, warmup):
    """Random multi-day episodes (labour action / schedule surges) as an hourly boolean mask."""
    mask = np.zeros(days * 24, dtype=bool)
    for _ in range(n):
        start = rng.integers(warmup, days - max_len)
        length = rng.integers(min_len, max_len + 1)
        mask[start * 24:(start + length) * 24] = True
    return mask


def _simulate_port(rng, code, p, s, hours, start, weather):
    c = p["berths"]
    teu_sigma = 0.35
    mean_service = 0.35 * p["mean_teu"] / p["teu_per_hour"] + 4
    base_rate = p["target_util"] * c / mean_service  # vessels per hour

    ts = start + pd.to_timedelta(np.arange(hours), unit="h")
    labor_h = _episodes(rng, hours // 24, s["labor_episodes_per_port"], 2, 4, 14)
    surge_h = _episodes(rng, hours // 24, s["surge_episodes_per_port"], 3, 3, 0)
    rate = (base_rate * ts.month.map(SEASON).to_numpy() * ts.day_of_week.map(DOW).to_numpy()
            * np.where(surge_h, s["surge_multiplier"], 1.0))

    counts = rng.poisson(rate)
    sched = np.repeat(np.arange(hours), counts) + rng.random(counts.sum())
    n = len(sched)
    actual = np.clip(sched + rng.normal(0, s["eta_noise_sd_hours"], n), 0, hours - 1)
    teu = rng.lognormal(np.log(p["mean_teu"]) - teu_sigma ** 2 / 2, teu_sigma, n).round()
    work = 0.35 * teu / p["teu_per_hour"] * rng.lognormal(-0.02, 0.2, n) + 4

    storm_h = weather[p["region"]]["storm_h"]
    order = np.argsort(actual)
    berth_start = np.full(n, np.nan)
    departure = np.full(n, np.nan)
    queue_len = np.zeros(hours)
    occupied = np.zeros(hours)
    queue, berths, ptr = deque(), {}, 0

    for h in range(hours):
        while ptr < n and actual[order[ptr]] <= h:  # vessels that have reached anchorage
            queue.append(order[ptr])
            ptr += 1
        storm = storm_h[h]
        if not storm:
            while queue and len(berths) < c:
                v = queue.popleft()
                berth_start[v] = h
                berths[v] = work[v]
        prod = 0.0 if storm else (s["labor_productivity"] if labor_h[h] else 1.0) * \
            (s["night_productivity"] if h % 24 < 6 else 1.0)
        for v in list(berths):
            berths[v] -= prod
            if berths[v] <= 0:
                departure[v] = h + 1
                del berths[v]
        queue_len[h] = len(queue)
        occupied[h] = len(berths)

    df = pd.DataFrame({
        "port": code, "sched_eta_h": sched, "actual_arrival": actual, "vessel_teu": teu,
        "berth_start": berth_start, "departure": departure, "planned_service_hours": work,
    })
    df["wait_hours"] = df["berth_start"] - df["actual_arrival"]
    df["berth_hours"] = df["departure"] - df["berth_start"]
    df["turnaround_hours"] = df["departure"] - df["actual_arrival"]
    # Excess port time = time lost vs. the plan: anchorage wait + berth overrun beyond the
    # terminal's planned service time (overrun comes from storms, labour action, night shifts).
    df["excess_hours"] = df["wait_hours"] + (df["berth_hours"] - df["planned_service_hours"]).clip(lower=0)
    state = {"queue_len": queue_len, "occupied": occupied, "labor_h": labor_h,
             "storm_h": storm_h, "surge_h": surge_h, "berths": c}
    return df, state


def _observables(rng, df, state, weather_r, s, cfg, start):
    """Features a planner can know at t0 = scheduled ETA - horizon. Uses only data <= t0."""
    horizon = cfg["prediction"]["horizon_hours"]
    thr = cfg["label"]["excess_threshold_hours"]
    c = state["berths"]
    sched_sorted = np.sort(df["sched_eta_h"].to_numpy())
    done = df.dropna(subset=["departure"]).sort_values("departure")
    dep_t = done["departure"].to_numpy()
    dep_late = (done["excess_hours"] > thr).to_numpy()
    hours = len(state["queue_len"])

    rows = []
    for t_eta in df["sched_eta_h"].to_numpy():
        t0 = t_eta - horizon
        i0 = int(np.floor(t0)) - 1
        if i0 < 168:
            rows.append({})
            continue
        lo, hi = np.searchsorted(dep_t, [t0 - 30 * 24, t0])
        hist = dep_late[lo:hi].mean() if hi > lo else np.nan
        day = int(t_eta // 24)
        fc = weather_r["forecast"][day:min(day + 2, len(weather_r["forecast"]))].max()
        lab_window = state["labor_h"][int(max(t_eta - 12, 0)):int(min(t_eta + 48, hours))].any()
        alert = rng.random() < (s["labor_alert_recall"] if lab_window else s["labor_false_alert"])
        rows.append({
            "queue_at_prediction": state["queue_len"][i0],
            "berths_occupied_at_prediction": state["occupied"][i0],
            "berth_util_7d": state["occupied"][i0 - 168:i0].mean() / c,
            "sched_arrivals_ahead": int(np.searchsorted(sched_sorted, t_eta) - np.searchsorted(sched_sorted, t0)),
            "sched_arrivals_window": int(np.searchsorted(sched_sorted, t_eta + 24)
                                         - np.searchsorted(sched_sorted, t_eta - 24) - 1),
            "port_hist_delay_rate": hist,
            "forecast_wind_max": fc,
            "labor_alert": int(alert),
        })
    obs = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, obs], axis=1)

    # Realised disruption during the stay (analysis only -- NOT a feature).
    def _any(mask, a, b):
        if np.isnan(b):
            return False
        return bool(mask[int(a):int(min(b, hours - 1)) + 1].any())
    out["storm_during_stay"] = [_any(state["storm_h"], a, b) for a, b in zip(out.actual_arrival, out.departure)]
    out["labor_during_stay"] = [_any(state["labor_h"], a, b) for a, b in zip(out.actual_arrival, out.departure)]
    return out


def simulate(cfg: dict | None = None, seed: int | None = None, **overrides) -> tuple[pd.DataFrame, dict]:
    """Run the simulation and return (sampled labelled dataset, metadata)."""
    cfg = cfg or load_config()
    s = {**cfg["simulation"], **overrides}
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    start = pd.Timestamp(s["start_date"])
    days = s["days"]
    hours = days * 24
    regions = sorted({p["region"] for p in cfg["ports"].values()})
    weather = _simulate_weather(rng, days, regions, s)

    frames, meta_ports = [], {}
    for code, p in cfg["ports"].items():
        df, state = _simulate_port(rng, code, p, s, hours, start, weather)
        df = _observables(rng, df, state, weather[p["region"]], s, cfg, start)
        df["capacity_berths"] = p["berths"]
        frames.append(df)
        meta_ports[code] = {
            "vessels_simulated": int(len(df)),
            "mean_berth_utilisation": round(float(state["occupied"].mean() / p["berths"]), 3),
            "max_queue": int(state["queue_len"].max()),
            "storm_hours": int(state["storm_h"].sum()),
            "labor_hours": int(state["labor_h"].sum()),
        }
    full = pd.concat(frames, ignore_index=True)

    # ---- Ground truth -------------------------------------------------------------------
    # congested = the call lost more than `excess_threshold_hours` versus its plan.
    med = full.groupby("port")["turnaround_hours"].median()
    full["congested"] = (full["excess_hours"] > cfg["label"]["excess_threshold_hours"]).astype(int)

    # ---- Sample the labelled dataset ----------------------------------------------------
    eligible = full[(full.sched_eta_h >= s["warmup_days"] * 24 + 48) &
                    (full.sched_eta_h <= hours - 96) & full.departure.notna()]
    sample = eligible.sample(n=min(s["n_records"], len(eligible)), random_state=int(rng.integers(1e9)))
    sample = sample.sort_values("sched_eta_h").reset_index(drop=True)

    # Missing values from data-feed outages (AIS gaps, forecast feed down, etc.).
    for col in ["queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d", "forecast_wind_max"]:
        sample.loc[rng.random(len(sample)) < s["missing_rate"], col] = np.nan

    sample.insert(0, "call_id", [f"C{i:04d}" for i in range(len(sample))])
    sample["sched_eta"] = start + pd.to_timedelta(sample["sched_eta_h"], unit="h")
    sample["prediction_time"] = sample["sched_eta"] - pd.Timedelta(hours=cfg["prediction"]["horizon_hours"])
    sample["vessel_teu"] = sample["vessel_teu"].astype(int)

    meta = {
        "seed": cfg["seed"] if seed is None else seed,
        "overrides": overrides,
        "vessels_simulated_total": int(len(full)),
        "records_sampled": int(len(sample)),
        "congestion_rate_full": round(float(full["congested"].mean()), 3),
        "congestion_rate_sample": round(float(sample["congested"].mean()), 3),
        "port_median_turnaround_h": med.round(1).to_dict(),
        "ports": meta_ports,
        "true_causal_drivers": [
            "arrival pressure vs berth capacity (scheduled arrivals, current queue, utilisation)",
            "storms halting berth operations (forecast wind is a noisy proxy)",
            "labour action reducing productivity (alert is a noisy proxy)",
            "seasonal / weekday arrival-rate variation and schedule surges",
            "vessel size (longer berth occupation -> more exposure to storm / labour slowdowns)",
        ],
    }
    return sample, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "port_calls.csv"))
    args = ap.parse_args()
    df, meta = simulate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["call_id", "port", "capacity_berths", "sched_eta", "prediction_time", "vessel_teu",
            "queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d",
            "sched_arrivals_ahead", "sched_arrivals_window", "port_hist_delay_rate",
            "forecast_wind_max", "labor_alert", "sched_eta_h"] + OUTCOME_COLUMNS
    df[cols].to_csv(out, index=False)
    with open(out.with_name("simulation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {len(df)} records -> {out}")
    print(json.dumps({k: meta[k] for k in ["vessels_simulated_total", "congestion_rate_full",
                                           "congestion_rate_sample"]}, indent=2))
    print(json.dumps(meta["ports"], indent=2))


if __name__ == "__main__":
    main()
