"""Model classes: rule-of-thumb baseline and the production CongestionModel
(calibrated logistic regression + LightGBM challenger + confidence + wait-time quantiles)."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEED_FIELDS, Preprocessor, engineer, n_missing

warnings.filterwarnings("ignore", message="X does not have valid feature names")

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
N_BOOTSTRAP = 20


class RuleBaseline:
    """Operational rule of thumb: flag if demand per berth is high OR a storm is forecast.

    The pressure threshold is tuned on training data (max F1) so the comparison is fair.
    """

    def fit(self, X_raw, y):
        p = engineer(X_raw)["pressure_index"].fillna(engineer(X_raw)["pressure_index"].median())
        grid = np.quantile(p, np.linspace(0.3, 0.95, 40))
        storm = engineer(X_raw)["forecast_storm"].fillna(0).to_numpy()
        scores = [f1_score(y, ((p.to_numpy() > t) | (storm == 1)).astype(int)) for t in grid]
        self.threshold_ = float(grid[int(np.argmax(scores))])
        self.fill_ = float(p.median())
        return self

    def score(self, X_raw):
        F = engineer(X_raw)
        return (F["pressure_index"].fillna(self.fill_) / self.threshold_
                + F["forecast_storm"].fillna(0)).to_numpy()  # continuous score for ROC/PR

    def predict(self, X_raw):
        return (self.score(X_raw) > 1).astype(int)


class CongestionModel:
    """Calibrated congestion probability + confidence + expected wait range."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.prep = Preprocessor().fit(train)
        Xtr, Xva = self.prep.transform(train), self.prep.transform(val)
        ytr, yva = train["congested"].to_numpy(), val["congested"].to_numpy()

        # Production classifier: L2-regularised logistic regression. In rolling-origin CV it beat
        # LightGBM on every fold (AUC 0.86 vs 0.82) -- with ~350 training rows the boosted trees
        # overfit. Calibrated on the later-in-time validation slice with Platt scaling (isotonic
        # would overfit ~75 calibration points); this also absorbs some seasonal base-rate drift.
        self.clf = make_logreg().fit(Xtr, ytr)
        self.calibrated = CalibratedClassifierCV(FrozenEstimator(self.clf), method="sigmoid").fit(Xva, yva)

        # Challenger: gradient boosting (kept for comparison + SHAP interaction analysis).
        self.challenger = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed).fit(Xtr, ytr)
        self.challenger_cal = CalibratedClassifierCV(FrozenEstimator(self.challenger), method="sigmoid").fit(Xva, yva)

        # Bootstrap ensemble -> model-agreement component of confidence.
        rng = np.random.default_rng(seed)
        self.ensemble = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(Xtr), len(Xtr))
            self.ensemble.append(make_logreg().fit(Xtr.iloc[idx], ytr[idx]))

        # Expected anchorage wait: quantile regressors (P10 / P50 / P90).
        wtr = train["wait_hours"].to_numpy()
        self.quantiles = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200,
                                               learning_rate=0.03, num_leaves=7, min_child_samples=15,
                                               verbose=-1, random_state=seed).fit(Xtr, wtr)
                          for q in (0.1, 0.5, 0.9)}

        # Out-of-distribution reference: kNN distance in standardised feature space.
        self.ood_cols = [c for c in self.prep.feature_names_ if not c.startswith(("port_", "missing_"))]
        self.scaler = StandardScaler().fit(Xtr[self.ood_cols])
        Ztr = self.scaler.transform(Xtr[self.ood_cols])
        self.nn = NearestNeighbors(n_neighbors=6).fit(Ztr)
        d, _ = self.nn.kneighbors(Ztr)
        self.d95_ = float(np.quantile(d[:, 1:].mean(axis=1), 0.95))  # exclude self-match

        # Decision threshold that minimises expected operational cost on validation data.
        pva = self.calibrated.predict_proba(Xva)[:, 1]
        c_fn, c_fp = cfg["decision"]["cost_missed_delay"], cfg["decision"]["cost_false_alarm"]
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        costs = [c_fn * ((pva < t) & (yva == 1)).sum() + c_fp * ((pva >= t) & (yva == 0)).sum() for t in grid]
        self.threshold_ = float(grid[int(np.argmin(costs))])
        self.theoretical_threshold_ = c_fp / (c_fp + c_fn)

        # Context for explanations / the demo.
        self.train_medians_ = self.prep.medians_
        self.feature_ranges_ = Xtr.quantile([0.05, 0.5, 0.95]).to_dict()
        return self

    # ---------------------------------------------------------------- inference
    def transform(self, X_raw):
        return self.prep.transform(X_raw)

    def predict_proba(self, X_raw) -> np.ndarray:
        return self.calibrated.predict_proba(self.transform(X_raw))[:, 1]

    def confidence(self, X_raw) -> pd.DataFrame:
        """Confidence (0-100) = how much to trust the probability, NOT how extreme it is.

        * agreement    : 1 - spread of a 20-model bootstrap ensemble (model/data uncertainty)
        * completeness : share of live-feed inputs that were actually observed (not imputed)
        * familiarity  : is this situation similar to what the model was trained on?
        """
        X = self.transform(X_raw)
        boot = np.column_stack([m.predict_proba(X)[:, 1] for m in self.ensemble])
        std = boot.std(axis=1)
        agreement = np.clip(1 - std / 0.15, 0, 1)
        completeness = 1 - n_missing(X_raw) / len(FEED_FIELDS)
        d, _ = self.nn.kneighbors(self.scaler.transform(X[self.ood_cols]), n_neighbors=5)
        dist = d.mean(axis=1)
        familiarity = np.where(dist <= self.d95_, 1.0, np.clip(1 - (dist - self.d95_) / self.d95_, 0, 1))
        conf = 100 * (0.5 * agreement + 0.25 * completeness + 0.25 * familiarity)
        return pd.DataFrame({"confidence": conf.round(1), "ensemble_std": std, "agreement": agreement,
                             "completeness": completeness, "familiarity": familiarity}, index=X_raw.index)

    def predict_wait(self, X_raw) -> pd.DataFrame:
        X = self.transform(X_raw)
        q = {k: np.clip(m.predict(X), 0, None) for k, m in self.quantiles.items()}
        lo, mid, hi = np.sort(np.column_stack([q[0.1], q[0.5], q[0.9]]), axis=1).T  # avoid crossing
        return pd.DataFrame({"wait_p10": lo, "wait_p50": mid, "wait_p90": hi}, index=X_raw.index)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
