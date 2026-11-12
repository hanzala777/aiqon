"""Predict congestion risk for one port call from the command line.

Example:
  python -m src.predict --port MID_B --eta "2025-12-20 06:00" --teu 4200 --queue 6 \
      --occupied 6 --ahead 18 --window 17 --util7d 0.9 --wind 34 --labor 0

Any live input you omit is treated as MISSING (and imputed), which lowers the confidence --
exactly what would happen if an AIS or forecast feed were down. Port-level context (berth
count, 30-day delay rate) defaults to the port's latest known state.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.explain import explain_row
from src.simulate import ROOT


def load_model(path=None):
    return joblib.load(path or ROOT / "models" / "model.joblib")


def build_row(model, port, eta, teu, queue=None, occupied=None, ahead=None, window=None,
              util7d=None, wind=None, labor=0, hist=None) -> pd.Series:
    """Assemble one raw input row. Raises ValueError on an unknown port or unparseable ETA; value
    ranges are checked by features.validate_raw when the row is scored."""
    if port not in model.port_snapshots_:
        raise ValueError(f"invalid input: unknown port {port!r}; expected one of {sorted(model.port_snapshots_)}")
    try:
        eta_ts = pd.Timestamp(eta)
    except (ValueError, TypeError):
        raise ValueError(f"invalid input: ETA {eta!r} is not a date-time; use e.g. '2025-12-20 06:00'") from None
    if pd.isna(eta_ts):
        raise ValueError("invalid input: ETA is required")
    snap = model.port_snapshots_[port]
    nan = np.nan
    return pd.Series({
        "port": port, "capacity_berths": snap["capacity_berths"], "sched_eta": eta_ts,
        "vessel_teu": teu,
        "queue_at_prediction": nan if queue is None else queue,
        "berths_occupied_at_prediction": nan if occupied is None else occupied,
        "berth_util_7d": nan if util7d is None else util7d,
        "sched_arrivals_ahead": snap["sched_arrivals_ahead"] if ahead is None else ahead,
        "sched_arrivals_window": snap["sched_arrivals_window"] if window is None else window,
        "port_hist_delay_rate": snap["port_hist_delay_rate"] if hist is None else hist,
        "forecast_wind_max": nan if wind is None else wind,
        "labor_alert": labor,
    })


def main():
    ap = argparse.ArgumentParser(description="Port congestion risk for a single call")
    ap.add_argument("--port", required=True, choices=["HUB", "MID_A", "MID_B", "SML_A", "SML_B"])
    ap.add_argument("--eta", required=True, help="scheduled ETA, e.g. '2025-12-20 06:00'")
    ap.add_argument("--teu", type=float, default=3000)
    ap.add_argument("--queue", type=float, help="vessels waiting at anchorage now")
    ap.add_argument("--occupied", type=float, help="berths occupied now")
    ap.add_argument("--ahead", type=float, help="vessels scheduled to arrive before ETA (next 48h)")
    ap.add_argument("--window", type=float, help="vessels scheduled within +/-24h of ETA")
    ap.add_argument("--util7d", type=float, help="7-day berth utilisation (0-1)")
    ap.add_argument("--wind", type=float, help="forecast max wind (knots) for arrival window")
    ap.add_argument("--labor", type=int, default=0, choices=[0, 1], help="1 if a labour-action alert is active")
    ap.add_argument("--hist", type=float, help="share of calls delayed >12h at this port, last 30 days")
    ap.add_argument("--json", action="store_true", help="print full JSON output")
    a = ap.parse_args()

    model = load_model()
    try:  # invalid ETA / out-of-range values -> clean usage error (exit code 2), no traceback
        row = build_row(model, a.port, a.eta, a.teu, a.queue, a.occupied, a.ahead, a.window,
                        a.util7d, a.wind, a.labor, a.hist)
        res = explain_row(model, row)
    except ValueError as e:
        ap.error(str(e))
    if a.json:
        print(json.dumps(res, indent=2, default=float))
        return
    print("\n" + res["summary"])
    print(f"\nDecision threshold (expected-cost optimal): {res['decision_threshold']:.2f} -> "
          f"{'ACTION RECOMMENDED' if res['action_recommended'] else 'no action needed'}")
    print("\nRisk drivers (log-odds contribution):")
    for d in res["drivers"]:
        print(f"  + {d['contribution']:+.2f}  {d['text']}")
    for d in res["mitigating"]:
        print(f"  - {d['contribution']:+.2f}  {d['text']}")
    if res["actions"]:
        print("\nRecommended actions:")
        for x in res["actions"]:
            print(f"  * {x}")
    if res["caveats"]:
        print("\nCaveats: " + "; ".join(res["caveats"]))


if __name__ == "__main__":
    main()
