"""Tests: leakage guard, output ranges, reproducibility, explanation sanity, input validation,
monotonic risk, confidence behaviour, decision threshold, wait intervals, CLI and demo app."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.explain import explain_row
from src.features import FEED_FIELDS, MONOTONE_INCREASING, RAW_INPUTS, Preprocessor, engineer
from src.model import CongestionModel
from src.simulate import OUTCOME_COLUMNS, ROOT, load_config, simulate
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


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize("col,value", [("queue_at_prediction", -5), ("berth_util_7d", 1.8),
                                       ("berths_occupied_at_prediction", 99), ("labor_alert", 5),
                                       ("port", "XYZ"), ("sched_eta", "not-a-date"), ("vessel_teu", 0)])
def test_invalid_inputs_rejected(data, col, value):
    _, df = data
    row = df.iloc[[0]].copy()
    row[col] = row[col].astype(object)
    row.loc[row.index[0], col] = value
    with pytest.raises(ValueError, match="invalid input"):
        engineer(row)


def test_missing_feeds_are_valid_input(data):
    _, df = data
    row = df.iloc[[0]].copy()
    row[FEED_FIELDS] = np.nan
    assert not engineer(row).empty


# ---------------------------------------------------------------- model behaviour
def test_domain_signs_respected(model):
    m, _ = model
    coef = pd.Series(m.clf[-1].coef_[0], index=m.prep.feature_names_)
    assert (coef[MONOTONE_INCREASING] >= 0).all()


@pytest.mark.parametrize("col,values", [("berths_occupied_at_prediction", [0, 2, 4, 6]),
                                        ("vessel_teu", [1000, 4000, 9000]),
                                        ("port_hist_delay_rate", [0.0, 0.3, 0.6])])
def test_more_of_a_risk_factor_never_lowers_risk(model, col, values):
    m, te = model
    rows = pd.concat([te.iloc[[0]].assign(**{col: v}) for v in values], ignore_index=True)
    rows["berths_occupied_at_prediction"] = rows["berths_occupied_at_prediction"].clip(upper=rows["capacity_berths"])
    assert (np.diff(m.predict_proba(rows)) >= -1e-12).all()


def test_losing_all_feeds_lowers_confidence_for_every_call(model):
    m, te = model
    blind = te.copy()
    blind[FEED_FIELDS] = np.nan
    assert (m.confidence(blind).confidence.to_numpy() < m.confidence(te).confidence.to_numpy()).all()


def test_confidence_is_not_a_restatement_of_risk(model):
    m, te = model
    rho = pd.Series(m.confidence(te).confidence.to_numpy()).corr(
        pd.Series(np.abs(m.predict_proba(te) - 0.5)), method="spearman")
    assert abs(rho) < 0.8


def test_threshold_is_expected_cost_optimal(model, data):
    m, _ = model
    cfg, _ = data
    c_fn, c_fp = cfg["decision"]["cost_missed_delay"], cfg["decision"]["cost_false_alarm"]
    assert m.threshold_ == pytest.approx(c_fp / (c_fp + c_fn))


def test_wait_interval_is_conformalised(model, data):
    m, te = model
    cfg, df = data
    _, va, _ = chrono_split(df, cfg)
    w = m.predict_wait(va)
    covered = ((va.wait_hours >= w.wait_p10) & (va.wait_hours <= w.wait_p90)).mean()
    assert covered >= 0.8 - 1e-9  # split-conformal guarantee on the calibration slice


# ---------------------------------------------------------------- evaluation artefacts
def test_reported_holdout_metrics_reproducible():
    from src.evaluate import cls_metrics
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    d = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    got = cls_metrics(d.congested.to_numpy(), d.p_logreg.to_numpy())
    for k in ("roc_auc", "pr_auc", "brier"):
        assert got[k] == metrics["models"]["logreg_production"][k]


# ---------------------------------------------------------------- CLI and demo app
def _cli(*args):
    return subprocess.run([sys.executable, "-m", "src.predict", "--port", "MID_B", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_cli_predicts_and_rejects_bad_input():
    ok = _cli("--eta", "2025-10-20 06:00", "--queue", "6", "--wind", "35", "--json")
    assert ok.returncode == 0 and 0 <= json.loads(ok.stdout)["probability"] <= 1
    for bad in (["--eta", "not-a-date"], ["--eta", "2025-10-20 06:00", "--util7d", "1.5"],
                ["--eta", "2025-10-20 06:00", "--occupied", "40"], ["--eta", "2025-10-20 06:00", "--labor", "5"]):
        r = _cli(*bad)
        assert r.returncode == 2 and "Traceback" not in r.stderr


def test_app_renders_and_handles_bad_eta():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.text_input[0].set_value("tomorrow-ish")
    at.button[0].click().run()
    assert not at.exception and at.error
