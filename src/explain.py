"""Per-prediction explanations and recommended actions.

For the production logistic-regression model, the contribution of feature j (in log-odds,
relative to the average training call) is  coef_j * z_j  where z_j is the standardised value.
For a linear model with independent features this is exactly the SHAP value, so no
approximation is involved.

Contributions are summed into operational *factors* (traffic, weather, labour, ...). The traffic
features are correlated (queue, scheduled arrivals, occupancy, pressure index); the production model
constrains all of them to non-negative coefficients (features.MONOTONE_INCREASING), so neither a
single feature nor a factor can ever claim that "more traffic lowers risk".
"""
from __future__ import annotations

import pandas as pd

from src.features import FEED_FIELDS, RAW_INPUTS

GROUPS = {
    "traffic": ["queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d",
                "sched_arrivals_ahead", "sched_arrivals_window", "pressure_index", "window_arrivals_per_berth"],
    "capacity": ["capacity_berths"],
    "weather": ["forecast_wind_max", "forecast_storm"],
    "labour": ["labor_alert"],
    # 30-day delay history is positively correlated with congestion on its own (r=0.12); once the live
    # queue state is known it adds little (a queue's future depends on its present state). Its
    # coefficient is constrained to be >= 0 and it is reported inside the "port" factor.
    "port": ["port_hist_delay_rate"],
    "vessel": ["vessel_teu"],
    "timing": ["eta_hour", "eta_weekend"],
    "season": ["month_sin", "month_cos", "peak_season"],
}

ACTIONS = {
    "traffic": "Request a later berth window or slow-steam to arrive after the peak (just-in-time arrival "
               "saves fuel and emissions instead of idling at anchorage).",
    "weather": "Add a weather buffer to the customer ETA and check whether arrival can be moved outside the storm window.",
    "labour": "Monitor the labour situation; line up contingency (alternative port or inland routing) now.",
    "port": "Port-specific risk is high; compare alternative ports (see 'Compare ports').",
    "capacity": "Little berth slack at this port; pre-book a berth window with the terminal.",
    "vessel": "Large call: confirm berth and crane allocation with the terminal in advance.",
    "season": "Peak-season demand: build extra slack into downstream (rail/truck/warehouse) bookings.",
    "timing": "Consider shifting the arrival slot away from the night shift / weekend.",
}
DATA_ACTION = "Live data incomplete: refresh AIS / forecast feeds before acting on this prediction."


def group_of(feature: str) -> str:
    if feature.startswith("port_"):
        return "port"
    for g, cols in GROUPS.items():
        if feature in cols:
            return g
    return "other"


def contributions(model, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Log-odds contribution of every engineered feature, per row (rows x features)."""
    X = model.transform(X_raw)
    scaler, lr = model.clf[0], model.clf[-1]
    return (X - scaler.mean_) / scaler.scale_ * lr.coef_[0]


def factor_contributions(model, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Contributions summed per operational factor (rows x factors)."""
    c = contributions(model, X_raw)
    return c.T.groupby(c.columns.map(group_of)).sum().T


def _fmt(raw, col, spec):
    v = raw.get(col)
    return "n/a (feed missing)" if pd.isna(v) else format(v, spec)


def _sentence(group, contrib, raw, f, port_name, medians):
    up = contrib > 0
    if group == "traffic":
        parts = [f"{raw['sched_arrivals_ahead']:.0f} vessels due in the 48h before ETA"]
        if pd.notna(raw.get("queue_at_prediction")):
            parts.append(f"{raw['queue_at_prediction']:.0f} waiting at anchorage")
        if pd.notna(raw.get("berths_occupied_at_prediction")):
            parts.append(f"{raw['berths_occupied_at_prediction']:.0f}/{f['capacity_berths']:.0f} berths occupied")
        if pd.notna(raw.get("berth_util_7d")):
            parts.append(f"7-day utilisation {raw['berth_util_7d']:.0%}")
        missing = [n for n, c in [("anchorage count", "queue_at_prediction"),
                                  ("berth occupancy", "berths_occupied_at_prediction"),
                                  ("7-day utilisation", "berth_util_7d")] if pd.isna(raw.get(c))]
        txt = f"{'heavy' if up else 'light'} traffic: " + ", ".join(parts)
        return txt + (f" [unknown: {', '.join(missing)}]" if missing else "")
    if group == "weather":
        if pd.isna(raw.get("forecast_wind_max")):
            return "weather forecast unavailable (assumed typical)"
        w = raw["forecast_wind_max"]
        return (f"storm risk: forecast wind {w:.0f} kn (cranes stop above 32 kn)" if f["forecast_storm"]
                else f"{'breezy' if up else 'calm'} forecast: {w:.0f} kn max wind")
    if group == "labour":
        return "labour-action alert active" if f["labor_alert"] else "no labour-action alert"
    if group == "capacity":
        return (f"small port ({f['capacity_berths']:.0f} berths): little slack to absorb bunching" if up
                else f"larger port ({f['capacity_berths']:.0f} berths) absorbs bunching")
    if group == "vessel":
        size = "large" if f["vessel_teu"] > medians["vessel_teu"] else "small"
        return f"{size} vessel ({f['vessel_teu']:,.0f} TEU)"
    if group == "timing":
        return f"arrival around {int(f['eta_hour']):02d}:00 on a {'weekend' if f['eta_weekend'] else 'weekday'}"
    if group == "season":
        m = pd.to_datetime(raw["sched_eta"]).strftime("%B")
        return f"seasonal demand ({m}{', peak season' if f['peak_season'] else ''})"
    if group == "port":
        return (f"port-specific performance at {port_name} "
                f"({f['port_hist_delay_rate']:.0%} of calls lost >12h in the last 30 days)")
    return group


def explain_row(model, raw_row: pd.Series, top_k: int = 3) -> dict:
    """Full, human-readable prediction for a single port call."""
    X_raw = raw_row[RAW_INPUTS].to_frame().T.reset_index(drop=True)
    for c in X_raw.columns:
        if c not in ("port", "sched_eta"):
            X_raw[c] = pd.to_numeric(X_raw[c])
    raw = X_raw.iloc[0]
    prob = float(model.predict_proba(X_raw)[0])
    conf = model.confidence(X_raw).iloc[0]
    wait = model.predict_wait(X_raw).iloc[0]
    f = model.transform(X_raw).iloc[0]
    fc = factor_contributions(model, X_raw).iloc[0]
    port_name = getattr(model, "port_names_", {}).get(raw["port"], raw["port"])

    agg = pd.DataFrame({"group": fc.index, "contribution": fc.values})
    agg = agg[agg.contribution.abs() > 0.05].sort_values("contribution", ascending=False)
    agg["text"] = [_sentence(g, c, raw, f, port_name, model.prep.medians_)
                   for g, c in zip(agg.group, agg.contribution)]

    drivers = agg[agg.contribution > 0].head(top_k)
    mitigating = agg[agg.contribution < 0].sort_values("contribution").head(2)

    band = "HIGH" if prob >= 0.5 else "MODERATE" if prob >= 0.2 else "LOW"
    act = prob >= model.threshold_
    actions = []
    if act:
        for g in drivers.group:
            if g in ACTIONS and ACTIONS[g] not in actions:
                actions.append(ACTIONS[g])
    n_miss = int(raw[FEED_FIELDS].isna().sum())
    if n_miss:
        actions.append(DATA_ACTION)

    caveats = []
    if conf["agreement"] < 0.6:
        caveats.append("estimate is unstable: models refitted on resampled history"
                       + (" and plausible values of the missing feeds" if n_miss else "") + " disagree")
    if conf["familiarity"] < 0.8:
        caveats.append("situation is unlike the training data (extrapolation)")
    if n_miss:
        caveats.append(f"{n_miss} live input(s) missing and imputed with typical values")

    p_txt = ">99%" if prob > 0.99 else "<1%" if prob < 0.01 else f"{prob:.0%}"
    reasons = "; ".join(drivers.text) if len(drivers) else "no strong risk drivers"
    summary = (f"{band} congestion risk ({p_txt}, confidence {conf['confidence']:.0f}%). "
               f"Main drivers: {reasons}. Expected anchorage wait ~{wait['wait_p50']:.0f}h "
               f"(80% range {wait['wait_p10']:.0f}-{wait['wait_p90']:.0f}h).")
    return {
        "probability": prob, "probability_text": p_txt, "risk_band": band, "action_recommended": bool(act),
        "decision_threshold": model.threshold_,
        "confidence": float(conf["confidence"]),
        "confidence_components": {k: float(conf[k]) for k in ("agreement", "completeness", "familiarity")},
        "expected_wait_hours": {k: float(wait[k]) for k in ("wait_p10", "wait_p50", "wait_p90")},
        "drivers": drivers[["text", "group", "contribution"]].to_dict("records"),
        "mitigating": mitigating[["text", "group", "contribution"]].to_dict("records"),
        "all_factors": agg[["text", "group", "contribution"]].to_dict("records"),
        "actions": actions, "caveats": caveats, "summary": summary,
    }


def global_importance(model, X_raw: pd.DataFrame) -> pd.Series:
    """Mean |log-odds contribution| per operational factor over a dataset."""
    return factor_contributions(model, X_raw).abs().mean().sort_values(ascending=False)
