"""Robustness experiments -> reports/experiments.json

1. Label-definition sensitivity: congestion threshold 6h / 12h / 24h of excess port time.
2. Data-quality stress: more missing feed data, noisier ETAs, both.
3. Seed variance: the whole pipeline (simulate -> train -> test) on 5 different random worlds,
   to show how much the headline numbers move by chance alone.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.simulate import ROOT, load_config, simulate
from src.train import run

warnings.filterwarnings("ignore")


def score(df, cfg, seed=0):
    _, out, _ = run(df, cfg, seed=seed)
    y = out.congested
    return {"prevalence": round(float(y.mean()), 3),
            "auc_logreg": round(roc_auc_score(y, out.p_logreg), 3),
            "auc_lgbm": round(roc_auc_score(y, out.p_lgbm), 3),
            "auc_rule": round(roc_auc_score(y, out.score_rule), 3),
            "pr_auc_logreg": round(average_precision_score(y, out.p_logreg), 3),
            "mean_confidence": round(float(out.confidence.mean()), 1)}


def main():
    cfg = load_config()
    base, _ = simulate(cfg)
    R = {}

    R["label_threshold"] = {}
    for thr in (6, 12, 24):
        df = base.copy()
        df["congested"] = (df["excess_hours"] > thr).astype(int)
        R["label_threshold"][f"{thr}h"] = score(df, cfg)
        print("label", thr, R["label_threshold"][f"{thr}h"])

    R["data_quality"] = {}
    for name, kw in {"baseline (5% missing, ETA sd 6h)": {},
                     "15% missing": {"missing_rate": 0.15},
                     "ETA noise sd 12h": {"eta_noise_sd_hours": 12.0},
                     "15% missing + ETA sd 12h": {"missing_rate": 0.15, "eta_noise_sd_hours": 12.0}}.items():
        df, _ = simulate(cfg, **kw)
        R["data_quality"][name] = score(df, cfg)
        print("quality", name, R["data_quality"][name])

    seeds = {}
    for s in (1, 2, 3, 4, 5):
        df, _ = simulate(cfg, seed=s)
        seeds[s] = score(df, cfg, seed=s)
        print("seed", s, seeds[s])
    agg = {k: {"mean": round(float(np.mean([v[k] for v in seeds.values()])), 3),
               "std": round(float(np.std([v[k] for v in seeds.values()])), 3)}
           for k in ["prevalence", "auc_logreg", "auc_lgbm", "auc_rule"]}
    R["seed_variance"] = {"per_seed": seeds, "summary": agg}

    with open(ROOT / "reports" / "experiments.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
