"""Streamlit demo:  streamlit run app.py"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explain import explain_row
from src.predict import build_row
from src.simulate import ROOT
from src.train import chrono_split, load_data
from src.simulate import load_config

st.set_page_config(page_title="Port congestion risk", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "model.joblib")
    cfg = load_config()
    df = load_data()
    _, _, test = chrono_split(df, cfg)
    metrics = json.load(open(ROOT / "reports" / "metrics.json"))
    return model, cfg, test.reset_index(drop=True), metrics


model, cfg, test, metrics = load()
PORTS = list(cfg["ports"])
BAND_COLOR = {"HIGH": "#c0392b", "MODERATE": "#e67e22", "LOW": "#27ae60"}

SCENARIOS = {
    "Custom": None,
    "Quiet day at the hub": dict(port="HUB", eta="2025-05-14 14:00", teu=5000, queue=0, occupied=6, ahead=8,
                                 window=9, util7d=0.55, wind=12, labor=0),
    "Peak season + storm at Midport Bravo": dict(port="MID_B", eta="2025-10-20 06:00", teu=4200, queue=6, occupied=6,
                                                 ahead=22, window=20, util7d=0.92, wind=35, labor=0),
    "Feed outage at Smallport Coral": dict(port="SML_A", eta="2025-09-10 02:00", teu=1800, queue=None, occupied=None,
                                           ahead=12, window=12, util7d=None, wind=None, labor=0),
    "Labour alert + storm at Smallport Delta": dict(port="SML_B", eta="2025-10-08 10:00", teu=1500, queue=2, occupied=3,
                                                    ahead=9, window=9, util7d=0.8, wind=36, labor=1),
}


def factor_chart(res):
    f = pd.DataFrame(res["all_factors"]).sort_values("contribution")
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(f) + 0.6))
    ax.barh(f.group, f.contribution, color=np.where(f.contribution > 0, "#c0392b", "#27ae60"))
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("contribution to risk (log-odds vs. an average call)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_result(res):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div style='font-size:0.9rem;color:#666'>Congestion risk</div>"
                f"<div style='font-size:2.2rem;font-weight:700;color:{BAND_COLOR[res['risk_band']]}'>"
                f"{res['probability_text']}</div><div style='color:{BAND_COLOR[res['risk_band']]}'>{res['risk_band']}</div>",
                unsafe_allow_html=True)
    c2.metric("Confidence", f"{res['confidence']:.0f}%")
    w = res["expected_wait_hours"]
    c3.metric("Expected anchorage wait", f"{w['wait_p50']:.0f} h", f"80% range {w['wait_p10']:.0f}-{w['wait_p90']:.0f} h",
              delta_color="off")
    c4.metric("Decision", "ACT" if res["action_recommended"] else "Monitor",
              f"threshold {res['decision_threshold']:.2f} (expected-cost optimal)", delta_color="off")
    st.info(res["summary"])
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why? Factor contributions**")
        st.pyplot(factor_chart(res), clear_figure=True)
    with right:
        if res["actions"]:
            st.markdown("**Recommended actions**")
            for a in res["actions"]:
                st.markdown(f"- {a}")
        st.markdown("**Confidence breakdown**")
        cc = res["confidence_components"]
        st.markdown(f"- Model agreement (20 bootstrap models x plausible values of missing feeds): **{cc['agreement']:.0%}**\n"
                    f"- Live data completeness: **{cc['completeness']:.0%}**\n"
                    f"- Similarity to training data: **{cc['familiarity']:.0%}**")
        for c in res["caveats"]:
            st.warning(c)


st.title("Port congestion & delay risk")
st.caption("Predicts, 48h before scheduled arrival, whether a port call will lose >12h versus plan "
           "(anchorage wait + berth overrun). Trained on a simulated dataset; see the report for assumptions.")

tab1, tab2, tab3, tab4 = st.tabs(["Predict a call", "Compare ports", "Test-set explorer", "Model performance"])

# ------------------------------------------------------------------ tab 1
with tab1:
    sc = st.selectbox("Scenario preset", list(SCENARIOS))
    s = SCENARIOS[sc] or dict(port="MID_A", eta="2025-11-05 08:00", teu=3000, queue=3, occupied=6, ahead=14,
                              window=14, util7d=0.8, wind=20, labor=0)
    with st.form("inputs"):
        a, b, c = st.columns(3)
        port = a.selectbox("Port", PORTS, index=PORTS.index(s["port"]),
                           format_func=lambda p: f"{p} - {cfg['ports'][p]['name']} ({cfg['ports'][p]['berths']} berths)")
        eta = a.text_input("Scheduled ETA", s["eta"])
        teu = a.number_input("Vessel size (TEU)", 300, 20000, int(s["teu"]), step=100)
        ahead = b.number_input("Vessels scheduled to arrive in the 48h before ETA", 0, 80, int(s["ahead"]))
        window = b.number_input("Vessels scheduled within +/-24h of ETA", 0, 80, int(s["window"]))
        labor = b.checkbox("Labour-action alert", bool(s["labor"]))
        st.markdown("**Live feeds** (untick to simulate a feed outage -> value is imputed, confidence drops)")
        d1, d2, d3, d4 = st.columns(4)
        q_on = d1.checkbox("AIS anchorage count", s["queue"] is not None)
        queue = d1.number_input("Vessels waiting", 0, 100, int(s["queue"] or 0))
        o_on = d2.checkbox("Berth occupancy", s["occupied"] is not None)
        occupied = d2.number_input("Berths occupied", 0, 20, int(s["occupied"] or 0))
        u_on = d3.checkbox("7-day utilisation", s["util7d"] is not None)
        util = d3.slider("Utilisation", 0.0, 1.0, float(s["util7d"] or 0.7))
        w_on = d4.checkbox("Weather forecast", s["wind"] is not None)
        wind = d4.slider("Max wind forecast (kn)", 0, 60, int(s["wind"] or 18))
        go = st.form_submit_button("Predict", type="primary")
    try:  # malformed ETA or impossible values -> friendly message instead of a stack trace
        row = build_row(model, port, eta, teu, queue if q_on else None, occupied if o_on else None, ahead, window,
                        util if u_on else None, wind if w_on else None, int(labor))
        res1 = explain_row(model, row)
    except ValueError as e:
        st.error(f"Cannot predict: {str(e).removeprefix('invalid input: ')}")
    else:
        st.session_state["last_inputs"] = dict(eta=eta, teu=teu, wind=wind if w_on else None, labor=int(labor))
        show_result(res1)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Same vessel and ETA, evaluated at every port using each port's **latest observed traffic state**. "
                "Use this to judge whether diverting or re-timing is worth it.")
    li = st.session_state.get("last_inputs", dict(eta="2025-11-05 08:00", teu=3000, wind=20, labor=0))
    rows = []
    for p in PORTS:
        snap = model.port_snapshots_[p]
        r = explain_row(model, build_row(model, p, li["eta"], li["teu"], snap["queue_at_prediction"],
                                         snap["berths_occupied_at_prediction"], None, None, snap["berth_util_7d"],
                                         li["wind"], li["labor"]))
        rows.append({"Port": f"{p} - {cfg['ports'][p]['name']}", "Risk": r["probability"],
                     "Band": r["risk_band"], "Confidence": round(r["confidence"]),
                     "Expected wait (h)": round(r["expected_wait_hours"]["wait_p50"]),
                     "Main driver": r["drivers"][0]["text"] if r["drivers"] else "-"})
    cmp_df = pd.DataFrame(rows).sort_values("Risk")
    st.dataframe(cmp_df.style.format({"Risk": "{:.0%}"}), hide_index=True, width="stretch")
    st.caption(f"Inputs carried over from 'Predict a call': ETA {li['eta']}, {li['teu']} TEU, "
               f"wind {li['wind']} kn, labour alert {li['labor']}. In production each port would use its own "
               "regional forecast and live AIS state.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("Held-out **test** calls (the most recent 15% of the timeline, never seen in training). "
                "Pick one to compare the prediction with what actually happened.")
    test_disp = test.assign(label=lambda d: d.call_id + "  " + d.port + "  " + d.sched_eta.dt.strftime("%Y-%m-%d %H:%M")
                            + np.where(d.congested == 1, "  (actually congested)", "  (actually on time)"))
    pick = st.selectbox("Port call", test_disp.label)
    r0 = test_disp[test_disp.label == pick].iloc[0]
    res = explain_row(model, r0)
    show_result(res)
    st.markdown(f"**What actually happened:** waited **{r0.wait_hours:.0f} h** at anchorage, "
                f"excess port time **{r0.excess_hours:.0f} h** -> "
                f"{'**congested**' if r0.congested else '**not congested**'}"
                f"{' (storm hit during the stay)' if r0.storm_during_stay else ''}"
                f"{' (labour action during the stay)' if r0.labor_during_stay else ''}. "
                f"Prediction was {'correct' if (res['probability'] >= res['decision_threshold']) == bool(r0.congested) else '**wrong**'}.")

# ------------------------------------------------------------------ tab 4
with tab4:
    ro, m = metrics["rolling_origin"], metrics["models"]
    keys = ["logreg_production", "lgbm_challenger", "rule_of_thumb"]
    st.markdown(f"Primary evaluation: **rolling-origin**, {ro['n_calls']} out-of-time calls ({ro['folds']} folds, "
                f"{ro['prevalence']:.0%} congested): the full system is refitted on the past and predicts the next "
                f"slice of the timeline. Final hold-out: the production model on the last {metrics['test_size']} calls.")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic regression (production)", "LightGBM (challenger)", "Rule of thumb"],
        "ROC-AUC": [ro["models"][k]["roc_auc"] for k in keys],
        "95% CI": [str(ro["models"][k]["roc_auc_95ci"]) for k in keys],
        "PR-AUC": [ro["models"][k]["pr_auc"] for k in keys],
        f"Hold-out AUC ({metrics['test_size']} calls)": [m[k]["roc_auc"] for k in keys],
    }), hide_index=True)
    st.image(str(ROOT / "reports" / "figures" / "performance.png"))
    st.image(str(ROOT / "reports" / "figures" / "explanations.png"))
    st.image(str(ROOT / "reports" / "figures" / "cost_curve.png"), width=480)
    st.markdown("**Key limitation:** all data is simulated. Numbers show the method works on a plausible world, "
                "not that it will reach this accuracy on a real port without retraining on real AIS / port-call data.")
