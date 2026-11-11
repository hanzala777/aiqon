"""Evaluation: metrics with uncertainty, calibration, breakdowns, confidence validation, figures.

Reads models/model.joblib + reports/test_predictions.csv (from train.py) and data/port_calls.csv.
Writes reports/metrics.json and reports/figures/*.png.
"""
from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)

from src.explain import global_importance
from src.features import Preprocessor
from src.model import LGB_PARAMS, RuleBaseline, make_logreg
from src.simulate import ROOT, load_config
from src.train import chrono_split, load_data

warnings.filterwarnings("ignore")
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Logistic regression (production)": "#1f6f8b", "LightGBM (challenger)": "#e07a3f",
          "Rule of thumb": "#8a8a8a"}


def boot_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)]


def cls_metrics(y, score, pred=None, prob=True):
    m = {"roc_auc": round(roc_auc_score(y, score), 3), "roc_auc_95ci": boot_ci(y, score, roc_auc_score),
         "pr_auc": round(average_precision_score(y, score), 3)}
    if prob:
        m["brier"] = round(brier_score_loss(y, score), 3)
    if pred is not None:
        m.update(precision=round(precision_score(y, pred, zero_division=0), 3),
                 recall=round(recall_score(y, pred), 3), f1=round(f1_score(y, pred), 3))
    return m


def rolling_cv(df, cfg, folds=5):
    """Rolling-origin evaluation: train on the first 50%+10%k of time, test on the next 10%."""
    df = df.sort_values("sched_eta").reset_index(drop=True)
    n, res = len(df), {"logreg": [], "lgbm": [], "rule": []}
    for k in range(folds):
        a, b = int(n * (0.5 + 0.1 * k)), int(n * (0.6 + 0.1 * k))
        tr, te = df.iloc[:a], df.iloc[a:b]
        prep = Preprocessor().fit(tr)
        Xtr, Xte, y = prep.transform(tr), prep.transform(te), te.congested.to_numpy()
        res["logreg"].append(roc_auc_score(y, make_logreg().fit(Xtr, tr.congested).predict_proba(Xte)[:, 1]))
        res["lgbm"].append(roc_auc_score(y, lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr, tr.congested).predict_proba(Xte)[:, 1]))
        res["rule"].append(roc_auc_score(y, RuleBaseline().fit(tr, tr.congested).score(te)))
    return {k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
                "folds": [round(x, 3) for x in v]} for k, v in res.items()}


def main():
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ROOT / "models" / "model.joblib")
    d = pd.read_csv(ROOT / "reports" / "test_predictions.csv", parse_dates=["sched_eta"])
    df = load_data()
    _, _, test = chrono_split(df, cfg)
    y = d.congested.to_numpy()
    thr = model.threshold_
    d["pred"] = (d.p_logreg >= thr).astype(int)

    M = {"test_size": int(len(d)), "test_prevalence": round(float(y.mean()), 3), "decision_threshold": thr}
    M["models"] = {
        "logreg_production": cls_metrics(y, d.p_logreg.to_numpy(), d.pred),
        "lgbm_challenger": cls_metrics(y, d.p_lgbm.to_numpy(), (d.p_lgbm >= thr).astype(int)),
        "rule_of_thumb": cls_metrics(y, d.score_rule.to_numpy(), d.pred_rule, prob=False),
    }
    M["rolling_origin_cv_auc"] = rolling_cv(df, cfg)

    # Operational cost at the chosen threshold vs simple policies.
    c_fn, c_fp = cfg["decision"]["cost_missed_delay"], cfg["decision"]["cost_false_alarm"]
    cost = lambda pred: int(c_fn * ((pred == 0) & (y == 1)).sum() + c_fp * ((pred == 1) & (y == 0)).sum())
    M["operational_cost_test"] = {"model_at_threshold": cost(d.pred.to_numpy()),
                                  "rule_of_thumb": cost(d.pred_rule.to_numpy()),
                                  "never_act": cost(np.zeros_like(y)), "always_act": cost(np.ones_like(y)),
                                  "costs": {"missed_delay": c_fn, "false_alarm": c_fp}}
    # With ~20 positive test cases one missed delay moves cost by 10 -> report uncertainty.
    rng = np.random.default_rng(0)
    pm, pr = d.pred.to_numpy(), d.pred_rule.to_numpy()
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        cm = c_fn * ((pm[i] == 0) & (y[i] == 1)).sum() + c_fp * ((pm[i] == 1) & (y[i] == 0)).sum()
        cr = c_fn * ((pr[i] == 0) & (y[i] == 1)).sum() + c_fp * ((pr[i] == 1) & (y[i] == 0)).sum()
        diffs.append(cm - cr)
    M["operational_cost_test"]["model_minus_rule_95ci"] = [float(np.quantile(diffs, 0.025)),
                                                           float(np.quantile(diffs, 0.975))]

    # Wait-time regression.
    w = d.wait_hours.to_numpy()
    M["wait_regression"] = {"mae_p50_hours": round(float(np.abs(w - d.wait_p50).mean()), 2),
                            "mae_baseline_median_hours": round(float(np.abs(w - np.median(df.wait_hours)).mean()), 2),
                            "p10_p90_coverage": round(float(((w >= d.wait_p10) & (w <= d.wait_p90)).mean()), 3)}

    # Breakdowns.
    def sub(mask):
        yy, pp = y[mask], d.p_logreg.to_numpy()[mask]
        return {"n": int(mask.sum()), "prevalence": round(float(yy.mean()), 2),
                "roc_auc": round(roc_auc_score(yy, pp), 3) if 0 < yy.mean() < 1 else None,
                "recall": round(recall_score(yy, d.pred.to_numpy()[mask], zero_division=0), 2) if yy.sum() else None}
    M["by_port"] = {p: sub((d.port == p).to_numpy()) for p in sorted(d.port.unique())}
    M["by_storm_during_stay"] = {"storm": sub(d.storm_during_stay.to_numpy()),
                                 "no_storm": sub(~d.storm_during_stay.to_numpy())}

    # Is the confidence score meaningful? Error should fall as confidence rises.
    d["conf_band"] = pd.qcut(d.confidence.rank(method="first"), 3, labels=["low", "medium", "high"])
    d["sq_err"] = (d.p_logreg - y) ** 2
    d["correct"] = (d.pred == y)
    M["confidence_validation"] = (d.groupby("conf_band", observed=True)
                                  .agg(n=("correct", "size"), accuracy=("correct", "mean"), brier=("sq_err", "mean"),
                                       mean_conf=("confidence", "mean")).round(3).reset_index().to_dict("records"))

    # Global explanation: which factors drive predictions (production model).
    imp = global_importance(model, test)
    M["global_factor_importance"] = imp.round(3).to_dict()

    # Interaction check with LightGBM SHAP interaction values (does traffic x weather matter?).
    import shap
    Xte = model.transform(test)
    inter = shap.TreeExplainer(model.challenger).shap_interaction_values(Xte)
    inter = inter[1] if isinstance(inter, list) else inter
    mean_abs = np.abs(inter).mean(axis=0)
    names = list(Xte.columns)
    pairs = sorted(((mean_abs[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)[:5]
    M["top_lgbm_interactions"] = [{"pair": f"{a} x {b}", "mean_abs_shap": round(float(v), 4)} for v, a, b in pairs]

    with open(ROOT / "reports" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2, default=str)

    make_figures(d, y, df, model, imp, cfg)
    print(json.dumps({k: M[k] for k in ["test_size", "test_prevalence", "models", "rolling_origin_cv_auc",
                                        "operational_cost_test", "wait_regression", "confidence_validation"]},
                     indent=1, default=str))


def make_figures(d, y, df, model, imp, cfg):
    series = [("Logistic regression (production)", d.p_logreg), ("LightGBM (challenger)", d.p_lgbm),
              ("Rule of thumb", d.score_rule)]

    # 1. ROC + PR + calibration
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for name, s in series:
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr, color=COLORS[name], lw=1.4, label=f"{name.split(' (')[0]} ({roc_auc_score(y, s):.2f})")
        pr, rc, _ = precision_recall_curve(y, s)
        ax[1].plot(rc, pr, color=COLORS[name], lw=1.4)
    ax[0].plot([0, 1], [0, 1], ":", color="#bbb", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test, AUC)")
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    ax[1].axhline(y.mean(), ls=":", color="#bbb", lw=1)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for name, s in series[:2]:
        fr, mp = calibration_curve(y, s, n_bins=5, strategy="quantile")
        ax[2].plot(mp, fr, "o-", color=COLORS[name], lw=1.4, ms=3)
    ax[2].plot([0, 1], [0, 1], ":", color="#bbb", lw=1)
    ax[2].set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration (test, 5 bins)")
    fig.tight_layout()
    fig.savefig(FIG / "performance.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Global factor importance + empirical traffic x weather interaction + confidence validation
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    imp = imp[imp > 0.01][::-1]
    ax[0].barh(imp.index, imp.values, color="#1f6f8b")
    ax[0].set(xlabel="mean |log-odds contribution|", title="What drives predictions")

    tr = df.copy()
    pres = (tr.queue_at_prediction.fillna(0) + tr.berths_occupied_at_prediction.fillna(tr.berths_occupied_at_prediction.median())
            + tr.sched_arrivals_ahead) / tr.capacity_berths
    tr["pressure_band"] = pd.qcut(pres, 3, labels=["low", "medium", "high"])
    tr["storm"] = np.where(tr.forecast_wind_max >= 30, "storm forecast", "no storm")
    tab = tr.groupby(["pressure_band", "storm"], observed=True).congested.agg(["mean", "size"]).reset_index()
    xs = np.arange(3)
    for k, (lab, col) in enumerate([("no storm", "#9ecae1"), ("storm forecast", "#08519c")]):
        t = tab[tab.storm == lab].set_index("pressure_band").reindex(["low", "medium", "high"])
        ax[1].bar(xs + (k - 0.5) * 0.38, t["mean"], 0.38, color=col, label=lab)
        for x, (m, n) in zip(xs + (k - 0.5) * 0.38, zip(t["mean"], t["size"])):
            if pd.notna(n):
                ax[1].text(x, (m if pd.notna(m) else 0) + 0.02, f"n={int(n)}", ha="center", fontsize=5)
    ax[1].set_xticks(xs, ["low", "medium", "high"])
    ax[1].set(xlabel="Demand per berth (tercile)", ylabel="Congestion rate", title="Traffic x weather (all data)")
    ax[1].legend(fontsize=6, frameon=False)

    cv = d.groupby("conf_band", observed=True).agg(acc=("correct", "mean"), brier=("sq_err", "mean"))
    ax[2].bar(cv.index.astype(str), cv["acc"], color="#1f6f8b")
    for i, (a, b) in enumerate(zip(cv["acc"], cv["brier"])):
        ax[2].text(i, a + 0.02, f"{a:.0%}\nBrier {b:.2f}", ha="center", fontsize=6)
    ax[2].set(ylim=(0, 1.15), xlabel="Confidence tercile (test)", ylabel="Accuracy at threshold",
              title="Does confidence mean anything?")
    fig.tight_layout()
    fig.savefig(FIG / "explanations.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Expected cost vs threshold (test) -- links probability to the decision
    c_fn, c_fp = cfg["decision"]["cost_missed_delay"], cfg["decision"]["cost_false_alarm"]
    ts = np.linspace(0.02, 0.98, 97)
    p = d.p_logreg.to_numpy()
    costs = [c_fn * ((p < t) & (y == 1)).sum() + c_fp * ((p >= t) & (y == 0)).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(ts, costs, color="#1f6f8b", lw=1.4)
    ax.axvline(model.threshold_, color="#e07a3f", ls="--", lw=1, label=f"chosen on validation ({model.threshold_:.2f})")
    ax.set(xlabel="Decision threshold", ylabel="Total cost (test)",
           title=f"Cost: missed delay={c_fn}, false alarm={c_fp}")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cost_curve.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
