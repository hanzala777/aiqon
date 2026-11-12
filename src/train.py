"""Model training: chronological split, baselines, calibrated logistic regression (production),
LightGBM challenger, confidence components and quantile wait-time models.

Outputs
  models/model.joblib            fitted CongestionModel (used by predict.py / app.py)
  reports/test_predictions.csv   per-call test-set predictions of every model (for evaluate.py)
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src.features import n_missing
from src.model import CongestionModel, RuleBaseline
from src.simulate import ROOT, load_config

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def chrono_split(df: pd.DataFrame, cfg: dict):
    """Train on the past, calibrate on the next slice, test on the most recent calls."""
    df = df.sort_values("sched_eta").reset_index(drop=True)
    n = len(df)
    a = int(n * cfg["split"]["train_frac"])
    b = int(n * (cfg["split"]["train_frac"] + cfg["split"]["val_frac"]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def run(df: pd.DataFrame, cfg: dict, seed: int = 0):
    """Fit everything on train/val, return the model and a test-prediction frame."""
    train, val, test = chrono_split(df, cfg)
    model = CongestionModel().fit(train, val, cfg, seed=seed)
    rule = RuleBaseline().fit(train, train["congested"])

    out = test[["call_id", "port", "sched_eta", "congested", "wait_hours", "storm_during_stay",
                "labor_during_stay"]].copy()
    Xte = model.transform(test)
    out["p_logreg"] = model.predict_proba(test)  # production model
    out["p_lgbm"] = model.challenger_cal.predict_proba(Xte)[:, 1]
    out["score_rule"] = rule.score(test)
    out["pred_rule"] = rule.predict(test)
    out = out.join(model.confidence(test)).join(model.predict_wait(test))
    out["n_missing"] = n_missing(test)
    info = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
            "train_period": [str(train.sched_eta.min()), str(train.sched_eta.max())],
            "test_period": [str(test.sched_eta.min()), str(test.sched_eta.max())],
            "rule_pressure_threshold": rule.threshold_,
            "decision_threshold": model.threshold_, "theoretical_threshold": model.theoretical_threshold_,
            "val_cost_threshold_for_reference": model.val_cost_threshold_,
            "wait_interval_conformal_adjustment_hours": round(model.wait_conformal_, 3),
            "logreg_coefficients_standardised": dict(zip(model.prep.feature_names_,
                                                         np.round(model.clf[-1].coef_[0], 3).tolist()))}
    return model, out, info


def load_data(path=None) -> pd.DataFrame:
    return pd.read_csv(path or ROOT / "data" / "port_calls.csv", parse_dates=["sched_eta", "prediction_time"])


def main():
    cfg = load_config()
    df = load_data()
    model, out, info = run(df, cfg, seed=cfg["seed"])
    # Latest observed state per port -> default inputs for the demo's "compare ports" view.
    model.port_snapshots_ = (df.sort_values("sched_eta").groupby("port").tail(1)
                             .set_index("port")[[c for c in df.columns if c in
                                                 ["capacity_berths", "queue_at_prediction",
                                                  "berths_occupied_at_prediction", "berth_util_7d",
                                                  "sched_arrivals_ahead", "sched_arrivals_window",
                                                  "port_hist_delay_rate"]]].to_dict("index"))
    model.port_names_ = {k: v["name"] for k, v in cfg["ports"].items()}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "model.joblib")
    out.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)
    with open(ROOT / "reports" / "train_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("logreg")}, indent=2))


if __name__ == "__main__":
    main()
