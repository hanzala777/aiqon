"""Model classes: rule-of-thumb baseline and the production CongestionModel
(sign-constrained, calibrated logistic regression + LightGBM challenger + confidence + wait-time intervals)."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEED_FIELDS, MONOTONE_INCREASING, Preprocessor, engineer

warnings.filterwarnings("ignore", message="X does not have valid feature names")

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
N_BOOTSTRAP = 20
N_IMPUTATIONS = 25            # plausible values drawn from history for each missing live feed
AGREEMENT_LOGIT_SCALE = 2.0   # ensemble spread (sd in log-odds) at which model agreement reaches 0:
                              # around a 50% call the models' 80% range then spans ~7%-93%
WAIT_INTERVAL_COVERAGE = 0.80 # target coverage of the conformalised P10-P90 wait interval


class SignConstrainedLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2-regularised logistic regression with optional sign constraints on coefficients.

    Same objective as sklearn's LogisticRegression(penalty="l2", C=C): C * sum(log-loss) + 0.5 * ||w||^2,
    intercept unpenalised, solved with L-BFGS-B so that coefficients of the features named in
    `increasing` are kept >= 0. Feature names are read from the column names of X (the pipeline's
    scaler outputs a DataFrame), so constraints follow the features wherever they sit.
    """

    def __init__(self, C=1.0, increasing=(), max_iter=3000):
        self.C = C
        self.increasing = increasing
        self.max_iter = max_iter

    def fit(self, X, y):
        names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        t = (y == self.classes_[-1]).astype(float)
        inc = set(self.increasing)
        bounds = [(0.0, None) if n in inc else (None, None) for n in names] + [(None, None)]

        def objective(wb):
            w, b = wb[:-1], wb[-1]
            z = X @ w + b
            p = expit(z)
            loss = np.sum(np.logaddexp(0.0, z) - t * z)
            return self.C * loss + 0.5 * w @ w, np.append(self.C * (X.T @ (p - t)) + w, self.C * np.sum(p - t))

        res = minimize(objective, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "gtol": 1e-8})
        self.coef_ = res.x[:-1][None, :]
        self.intercept_ = np.array([res.x[-1]])
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = names
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


def make_logreg():
    return make_pipeline(StandardScaler().set_output(transform="pandas"),
                         SignConstrainedLogisticRegression(C=0.5, increasing=tuple(MONOTONE_INCREASING)))


class RuleBaseline:
    """Operational rule of thumb: flag if demand per berth is high OR a storm is forecast.

    The pressure threshold is tuned on training data (max F1) so the comparison is fair.
    """

    def fit(self, X_raw, y):
        F = engineer(X_raw)
        p = F["pressure_index"].fillna(F["pressure_index"].median())
        grid = np.quantile(p, np.linspace(0.3, 0.95, 40))
        storm = F["forecast_storm"].fillna(0).to_numpy()
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

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int = 0, full: bool = True):
        """`full=False` fits only the calibrated classifier (used for fast rolling-origin evaluation)."""
        self.cfg = cfg
        self.prep = Preprocessor().fit(train)
        Xtr, Xva = self.prep.transform(train), self.prep.transform(val)
        ytr, yva = train["congested"].to_numpy(), val["congested"].to_numpy()

        # Production classifier: L2-regularised logistic regression with domain sign constraints
        # (see features.MONOTONE_INCREASING). In rolling-origin evaluation it matched or beat LightGBM
        # on every fold: with ~350 training rows the boosted trees overfit. Calibrated on the later-in-time
        # validation slice with Platt scaling (isotonic would overfit ~75 calibration points).
        self.clf = make_logreg().fit(Xtr, ytr)
        self.calibrated = CalibratedClassifierCV(FrozenEstimator(self.clf), method="sigmoid").fit(Xva, yva)

        # Decision threshold. With calibrated probabilities the expected-cost-minimising rule is
        # "act if p >= c_fp / (c_fp + c_fn)". It does not depend on the small validation slice: the
        # empirical cost-minimising threshold on 75 validation calls ranged 0.06-0.39 under bootstrap
        # resampling, so it is kept for reporting only.
        c_fn, c_fp = cfg["decision"]["cost_missed_delay"], cfg["decision"]["cost_false_alarm"]
        self.theoretical_threshold_ = c_fp / (c_fp + c_fn)
        self.threshold_ = self.theoretical_threshold_
        pva = self.calibrated.predict_proba(Xva)[:, 1]
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        costs = [c_fn * ((pva < t) & (yva == 1)).sum() + c_fp * ((pva >= t) & (yva == 0)).sum() for t in grid]
        self.val_cost_threshold_ = float(grid[int(np.argmin(costs))])
        self.train_medians_ = self.prep.medians_
        if not full:
            return self

        # Challenger: gradient boosting (kept for comparison + SHAP interaction analysis).
        self.challenger = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed).fit(Xtr, ytr)
        self.challenger_cal = CalibratedClassifierCV(FrozenEstimator(self.challenger), method="sigmoid").fit(Xva, yva)

        # Bootstrap ensemble -> model-agreement component of confidence.
        rng = np.random.default_rng(seed)
        self.ensemble = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(Xtr), len(Xtr))
            self.ensemble.append(make_logreg().fit(Xtr.iloc[idx], ytr[idx]))

        # Donor values for missing live feeds: complete training calls of the same port (hot-deck
        # multiple imputation). Fixed at fit time so confidence is deterministic and batch-independent.
        rng_d = np.random.default_rng(seed + 1)
        complete = train.dropna(subset=FEED_FIELDS)
        self.donors_ = {p: g[FEED_FIELDS].to_numpy(float)[rng_d.integers(0, len(g), N_IMPUTATIONS)]
                        for p, g in complete.groupby("port")}
        self.donors_all_ = complete[FEED_FIELDS].to_numpy(float)[rng_d.integers(0, len(complete), N_IMPUTATIONS)]

        # Expected anchorage wait: quantile regressors (P10 / P50 / P90), then split-conformal
        # calibration on the validation slice so the P10-P90 band reaches its nominal 80% coverage
        # (the raw quantile band covered only 67% of test waits).
        wtr = train["wait_hours"].to_numpy()
        self.quantiles = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200,
                                               learning_rate=0.03, num_leaves=7, min_child_samples=15,
                                               verbose=-1, random_state=seed).fit(Xtr, wtr)
                          for q in (0.1, 0.5, 0.9)}
        lo, _, hi = self._raw_wait(Xva)
        wva = val["wait_hours"].to_numpy()
        scores = np.maximum(lo - wva, wva - hi)
        level = min(1.0, np.ceil((len(scores) + 1) * WAIT_INTERVAL_COVERAGE) / len(scores))
        self.wait_conformal_ = float(np.quantile(scores, level, method="higher"))

        # Out-of-distribution reference: kNN distance in standardised feature space.
        self.ood_cols = [c for c in self.prep.feature_names_ if not c.startswith(("port_", "missing_"))]
        self.scaler = StandardScaler().fit(Xtr[self.ood_cols])
        Ztr = self.scaler.transform(Xtr[self.ood_cols])
        self.nn = NearestNeighbors(n_neighbors=6).fit(Ztr)
        d, _ = self.nn.kneighbors(Ztr)
        self.d95_ = float(np.quantile(d[:, 1:].mean(axis=1), 0.95))  # exclude self-match

        # Context for explanations / the demo.
        self.feature_ranges_ = Xtr.quantile([0.05, 0.5, 0.95]).to_dict()
        return self

    # ---------------------------------------------------------------- inference
    def transform(self, X_raw):
        return self.prep.transform(X_raw)

    def predict_proba(self, X_raw) -> np.ndarray:
        return self.calibrated.predict_proba(self.transform(X_raw))[:, 1]

    def _variants(self, X_raw):
        """Rows to score for confidence: complete rows as-is; rows with missing live feeds are
        expanded into N_IMPUTATIONS copies whose missing values come from historical donor calls."""
        raw = X_raw.reset_index(drop=True)
        miss = raw[FEED_FIELDS].isna().to_numpy()
        has_miss = miss.any(axis=1)
        parts, owners = [raw.loc[~has_miss]], [np.flatnonzero(~has_miss)]
        if has_miss.any():
            rows = raw.loc[has_miss]
            rep = rows.loc[rows.index.repeat(N_IMPUTATIONS)].copy()
            donors = np.vstack([self.donors_.get(p, self.donors_all_) for p in rows["port"]])
            vals = rep[FEED_FIELDS].to_numpy(dtype=float, copy=True)  # writable copy (pandas copy-on-write)
            gap = np.isnan(vals)
            vals[gap] = donors[gap]
            rep[FEED_FIELDS] = vals
            parts.append(rep)
            owners.append(np.repeat(np.flatnonzero(has_miss), N_IMPUTATIONS))
        return pd.concat(parts, ignore_index=True), np.concatenate(owners), miss

    def confidence(self, X_raw) -> pd.DataFrame:
        """Confidence (0-100) = how much to trust the probability, NOT how extreme it is.

        * agreement    : 1 - spread (sd, in log-odds) of the predictions of 20 bootstrap-refitted models
                         x plausible values of any missing live feed. Measured on the log-odds scale so
                         it is not mechanically small for extreme probabilities and large near 50%.
                         Missing an influential feed widens the spread (multiple-imputation logic).
        * completeness : share of live-feed inputs that were actually observed
        * familiarity  : is this situation similar to what the model was trained on? (k-NN distance,
                         averaged over the imputed variants for rows with missing feeds)
        """
        V, owners, miss = self._variants(X_raw)
        n = len(X_raw)
        XV = self.transform(V)
        logits = np.column_stack([m.decision_function(XV) for m in self.ensemble])
        cnt = np.bincount(owners, minlength=n) * logits.shape[1]
        s1 = np.bincount(owners, weights=logits.sum(axis=1), minlength=n)
        s2 = np.bincount(owners, weights=(logits ** 2).sum(axis=1), minlength=n)
        sd = np.sqrt(np.clip(s2 / cnt - (s1 / cnt) ** 2, 0, None))
        agreement = np.clip(1 - sd / AGREEMENT_LOGIT_SCALE, 0, 1)
        completeness = 1 - miss.sum(axis=1) / len(FEED_FIELDS)
        d, _ = self.nn.kneighbors(self.scaler.transform(XV[self.ood_cols]), n_neighbors=5)
        dist = d.mean(axis=1)
        fam_v = np.where(dist <= self.d95_, 1.0, np.clip(1 - (dist - self.d95_) / self.d95_, 0, 1))
        familiarity = np.bincount(owners, weights=fam_v, minlength=n) / np.bincount(owners, minlength=n)
        conf = 100 * (0.5 * agreement + 0.25 * completeness + 0.25 * familiarity)
        return pd.DataFrame({"confidence": conf.round(1), "ensemble_logit_sd": sd, "agreement": agreement,
                             "completeness": completeness, "familiarity": familiarity}, index=X_raw.index)

    def _raw_wait(self, X):
        q = {k: np.clip(m.predict(X), 0, None) for k, m in self.quantiles.items()}
        return np.sort(np.column_stack([q[0.1], q[0.5], q[0.9]]), axis=1).T  # avoid crossing

    def predict_wait(self, X_raw) -> pd.DataFrame:
        lo, mid, hi = self._raw_wait(self.transform(X_raw))
        e = getattr(self, "wait_conformal_", 0.0)
        lo, hi = np.minimum(np.clip(lo - e, 0, None), mid), np.maximum(hi + e, mid)
        return pd.DataFrame({"wait_p10": lo, "wait_p50": mid, "wait_p90": hi}, index=X_raw.index)
