"""Smoke tests: leakage guard, output ranges, reproducibility, explanation sanity."""
import numpy as np
import pandas as pd
import pytest

from src.explain import explain_row
from src.features import RAW_INPUTS, Preprocessor
from src.model import CongestionModel
from src.simulate import OUTCOME_COLUMNS, load_config, simulate
from src.train import chrono_split


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    df, _ = simulate(cfg)
    return cfg, df


@pytest.fixture(scope="module")
def model(data):
    cfg, df = data
    tr, va, te = chrono_split(df, cfg)
    return CongestionModel().fit(tr, va, cfg), te


def test_no_outcome_leakage(data):
    _, df = data
    X = Preprocessor().fit(df).transform(df)
    assert not set(OUTCOME_COLUMNS) & set(X.columns)
    assert not set(OUTCOME_COLUMNS) & set(RAW_INPUTS)


def test_features_ignore_outcomes(data):
    """Scrambling every outcome column must not change the feature matrix."""
    _, df = data
    prep = Preprocessor().fit(df)
    scrambled = df.copy()
    for c in OUTCOME_COLUMNS:
        scrambled[c] = np.random.default_rng(0).permutation(scrambled[c].to_numpy())
    pd.testing.assert_frame_equal(prep.transform(df), prep.transform(scrambled))


def test_simulation_reproducible():
    cfg = load_config()
    a, _ = simulate(cfg)
    b, _ = simulate(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_chronological_split(data):
    cfg, df = data
    tr, va, te = chrono_split(df, cfg)
    assert tr.sched_eta.max() <= va.sched_eta.min() <= te.sched_eta.min()


def test_output_ranges(model):
    m, te = model
    p = m.predict_proba(te)
    c = m.confidence(te)
    w = m.predict_wait(te)
    assert ((p >= 0) & (p <= 1)).all()
    assert c.confidence.between(0, 100).all()
    assert (w.wait_p10 <= w.wait_p50).all() and (w.wait_p50 <= w.wait_p90).all()


def test_missing_data_lowers_confidence(model):
    m, te = model
    row = te.iloc[[0]].copy()
    full = m.confidence(row).confidence.iloc[0]
    row[["queue_at_prediction", "berths_occupied_at_prediction", "berth_util_7d", "forecast_wind_max"]] = np.nan
    assert m.confidence(row).confidence.iloc[0] < full


def test_more_traffic_more_risk(model):
    m, te = model
    row = te.iloc[0].copy()
    lo, hi = row.copy(), row.copy()
    lo[["queue_at_prediction", "sched_arrivals_ahead"]] = [0, 2]
    hi[["queue_at_prediction", "sched_arrivals_ahead"]] = [15, 30]
    assert explain_row(m, hi)["probability"] > explain_row(m, lo)["probability"]


def test_explanation_structure(model):
    m, te = model
    res = explain_row(m, te.iloc[0])
    assert {"probability", "confidence", "drivers", "actions", "summary"} <= res.keys()
    assert isinstance(res["summary"], str) and len(res["summary"]) > 20
