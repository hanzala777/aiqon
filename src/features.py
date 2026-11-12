"""Leakage-safe feature engineering.

Every feature is computable at prediction time (scheduled ETA - 48h) from sources a port
planner actually has: port-call schedules / pre-arrival notices, AIS anchorage counts, berth
occupancy, weather forecasts, labour-relations alerts and the port's own recent history.
Realised outcomes (actual arrival, wait, storm during stay, ...) are excluded by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PORTS = ["HUB", "MID_A", "MID_B", "SML_A", "SML_B"]

# Raw fields that come from live feeds and may be missing (AIS gaps, forecast outages).
FEED_FIELDS = ["queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d", "forecast_wind_max"]

# Raw observables the system needs as input (what a caller of the API must provide).
RAW_INPUTS = ["port", "capacity_berths", "sched_eta", "vessel_teu", "queue_at_prediction",
              "berths_occupied_at_prediction", "berth_util_7d", "sched_arrivals_ahead",
              "sched_arrivals_window", "port_hist_delay_rate", "forecast_wind_max", "labor_alert"]

STORM_FORECAST_KNOTS = 30  # forecast slightly below the 32 kn crane limit -> "storm risk"

# Domain knowledge: more of any of these can never LOWER congestion risk. The production logistic
# regression constrains their coefficients to be >= 0. Without the constraint, correlated inputs
# picked up opposite signs in 500 records (e.g. "more berths occupied -> lower risk", and vessel
# size acting as a proxy for port size: "bigger ship -> lower risk").
MONOTONE_INCREASING = ["queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d",
                       "sched_arrivals_ahead", "sched_arrivals_window", "pressure_index",
                       "window_arrivals_per_berth", "vessel_teu", "forecast_wind_max", "forecast_storm",
                       "labor_alert", "port_hist_delay_rate", "peak_season"]


def validate_raw(df: pd.DataFrame) -> None:
    """Reject impossible inputs with a clear message instead of silently predicting on them.

    NaN is allowed only in the live-feed fields (FEED_FIELDS): a missing feed is a normal outage.
    """
    absent = [c for c in RAW_INPUTS if c not in df.columns]
    if absent:
        raise ValueError(f"invalid input: missing column(s) {absent}")
    errors = []
    unknown = sorted(set(df["port"].astype(str)) - set(PORTS))
    if unknown:
        errors.append(f"unknown port code(s) {unknown}; expected one of {PORTS}")
    try:
        pd.to_datetime(df["sched_eta"])
    except (ValueError, TypeError):
        errors.append("sched_eta is not a valid date-time (e.g. '2025-12-20 06:00')")
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in RAW_INPUTS if c not in ("port", "sched_eta")}
    for c, s in num.items():
        if c not in FEED_FIELDS and s.isna().any():
            errors.append(f"{c} is required (got a missing or non-numeric value)")

    def check(mask, msg):
        if bool(mask.any()):
            errors.append(msg)

    cap = num["capacity_berths"]
    check(cap <= 0, "capacity_berths must be > 0")
    check(num["vessel_teu"] <= 0, "vessel_teu must be > 0")
    for c in ("queue_at_prediction", "berths_occupied_at_prediction", "sched_arrivals_ahead", "sched_arrivals_window"):
        check(num[c] < 0, f"{c} cannot be negative")
    check(num["berths_occupied_at_prediction"] > cap, "berths_occupied_at_prediction cannot exceed the port's berth count")
    for c in ("berth_util_7d", "port_hist_delay_rate"):
        check((num[c] < 0) | (num[c] > 1), f"{c} must be a share between 0 and 1")
    check((num["forecast_wind_max"] < 0) | (num["forecast_wind_max"] > 150), "forecast_wind_max must be 0-150 knots")
    check(num["labor_alert"].notna() & ~num["labor_alert"].isin([0, 1]), "labor_alert must be 0 or 1")
    if errors:
        raise ValueError("invalid input: " + "; ".join(errors))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw observables -> model features (no fitting, row-wise only). Inputs are validated first."""
    validate_raw(df)
    eta = pd.to_datetime(df["sched_eta"])
    cap = df["capacity_berths"].astype(float)
    X = pd.DataFrame(index=df.index)
    # Traffic / capacity pressure
    X["queue_at_prediction"] = df["queue_at_prediction"]
    X["berths_occupied_at_prediction"] = df["berths_occupied_at_prediction"]
    X["berth_util_7d"] = df["berth_util_7d"]
    X["sched_arrivals_ahead"] = df["sched_arrivals_ahead"]
    X["sched_arrivals_window"] = df["sched_arrivals_window"]
    X["capacity_berths"] = cap
    # Demand that must be served before / around our vessel, per berth: the core congestion signal.
    X["pressure_index"] = (df["queue_at_prediction"] + df["berths_occupied_at_prediction"]
                           + df["sched_arrivals_ahead"]) / cap
    X["window_arrivals_per_berth"] = df["sched_arrivals_window"] / cap
    # Vessel
    X["vessel_teu"] = df["vessel_teu"]
    # Weather / labour
    X["forecast_wind_max"] = df["forecast_wind_max"]
    X["forecast_storm"] = (df["forecast_wind_max"] >= STORM_FORECAST_KNOTS).astype(float)
    X.loc[df["forecast_wind_max"].isna(), "forecast_storm"] = np.nan
    X["labor_alert"] = df["labor_alert"]
    # Port history
    X["port_hist_delay_rate"] = df["port_hist_delay_rate"]
    # Calendar / seasonality
    X["eta_hour"] = eta.dt.hour
    X["eta_weekend"] = (eta.dt.day_of_week >= 5).astype(int)
    X["month_sin"] = np.sin(2 * np.pi * eta.dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * eta.dt.month / 12)
    X["peak_season"] = eta.dt.month.isin([8, 9, 10, 11]).astype(int)
    # Port identity (captures port-specific productivity not explained by size)
    for p in PORTS:
        X[f"port_{p}"] = (df["port"] == p).astype(int)
    return X


class Preprocessor(BaseEstimator, TransformerMixin):
    """Median imputation fitted on TRAINING data only.

    Missing-value indicator columns were tried and removed: in this data missingness is random
    (feed outages), so the indicators only added noise (CV AUC 0.859 with vs 0.863 without) and
    showed up as spurious "drivers" in explanations. Missingness is instead surfaced through the
    confidence score (completeness component) and flagged in the explanation text.
    """

    def fit(self, X_raw: pd.DataFrame, y=None):
        X = engineer(X_raw)
        self.medians_ = X.median(numeric_only=True)
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = engineer(X_raw).fillna(self.medians_)
        return X[self.feature_names_].astype(float)


def n_missing(X_raw: pd.DataFrame) -> np.ndarray:
    return X_raw[FEED_FIELDS].isna().sum(axis=1).to_numpy()
