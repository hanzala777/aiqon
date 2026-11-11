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


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw observables -> model features (no fitting, row-wise only)."""
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
