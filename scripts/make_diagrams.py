"""Draw the two diagrams used in the technical report (reports/figures/).

    python scripts/make_diagrams.py

Both are drawn in millimetres on a canvas as wide as the report's text column, so font sizes are
the printed sizes. Each diagram is meant to be readable without the surrounding text.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
WIDTH = 172  # mm, text width of the A4 report

INK, GREY, LIGHT, SHADE, ACCENT = "#1a1a1a", "#5f5f5f", "#e9e9e9", "#f3f3f3", "#b5532a"


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Palatino Linotype" if "Palatino Linotype" in names else "DejaVu Serif"


plt.rcParams.update({"font.family": _font(), "font.size": 7, "text.color": INK, "hatch.linewidth": 0.6})


def canvas(h):
    fig = plt.figure(figsize=(WIDTH / 25.4, h / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(h, 0)  # y grows downwards
    ax.axis("off")
    return fig, ax


def arrow(ax, x1, y1, x2, y2, ls="-", color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=color,
                                 lw=0.7, ls=ls, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=7, **kw):
    kw.setdefault("va", "top")
    kw.setdefault("linespacing", 1.3)
    ax.text(x, y, s, size=size, **kw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def node(ax, x, y, w, h, title, sub=None, fill="white", lw=0.7, bold=True, tsize=7.0, ssize=6.1):
    """Box with a centred title and an optional grey italic second part (e.g. the source file)."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=INK, lw=lw, zorder=2))
    line = lambda size: size * 0.3528 * 1.25  # noqa: E731  (line height in mm)
    nt, ns = title.count("\n") + 1, (sub.count("\n") + 1 if sub else 0)
    total = nt * line(tsize) + (0.6 + ns * line(ssize) if sub else 0)
    y0 = y + (h - total) / 2 + 0.3
    ax.text(x + w / 2, y0, title, ha="center", va="top", size=tsize, weight="bold" if bold else "normal",
            linespacing=1.2, zorder=3)
    if sub:
        ax.text(x + w / 2, y0 + nt * line(tsize) + 0.6, sub, ha="center", va="top", size=ssize, color=GREY,
                style="italic", linespacing=1.2, zorder=3)


def architecture():
    """Two lanes: the offline training pipeline and the prediction path for one port call."""
    fig, ax = canvas(84)
    band = "#f5f5f5"

    # ---- lane 1: training (offline)
    ax.add_patch(Rectangle((0, 0.5), WIDTH, 34.5, fc=band, ec="none"))
    text(ax, 2, 2.0, "TRAINING", size=7.4, weight="bold")
    text(ax, 21.0, 2.25, "offline, run once with  python run_all.py", size=6.6, color=GREY, style="italic")
    w, g, y, h = 24.6, 4.68, 8.5, 13.0
    train = [("Assumptions", "config.yaml"), ("Port simulator", "simulate.py"),
             ("500 labelled\ncalls", "port_calls.csv"), ("Features", "features.py"),
             ("Train and\ncalibrate", "train.py, model.py"), ("Saved model", "model.joblib")]
    xs = [0.5 + i * (w + g) for i in range(len(train))]
    for i, (x, (t, s)) in enumerate(zip(xs, train)):
        node(ax, x, y, w, h, t, s, fill="white", lw=1.0 if i == 5 else 0.7)
        if i < len(train) - 1:
            arrow(ax, x + w + 0.4, y + h / 2, x + w + g - 0.4, y + h / 2)
    ex, ew = xs[2], xs[4] + w - xs[2]
    node(ax, ex, 25.5, ew, 7.5, "Evaluation: rolling refits on past data (250 unseen calls), hold-out, 27 tests",
         "evaluate.py, experiments.py", fill="white", bold=False, tsize=6.6, ssize=5.9)
    arrow(ax, xs[4] + w / 2, y + h + 0.4, xs[4] + w / 2, 25.1, ls=(0, (2, 1.5)), color=GREY)

    # ---- lane 2: prediction for one call
    top2 = 39.0
    ax.add_patch(Rectangle((0, top2), WIDTH, 44.5, fc=band, ec="none"))
    text(ax, 2, top2 + 1.5, "PREDICTION", size=7.4, weight="bold")
    text(ax, 24.5, top2 + 1.75, "for one port call, 48 h before its arrival", size=6.6, color=GREY, style="italic")

    cy = [51.0, 59.5, 68.0, 76.5]          # centres of the four parallel rows
    rh = 7.0
    mid = (cy[0] + cy[-1]) / 2
    node(ax, 0.5, mid - 13, 27, 26, "New port call",
         "port, ETA, ship size,\nships waiting,\nberths busy,\nwind forecast,\nstrike alert", ssize=6.2)
    node(ax, 33.0, mid - 7, 22, 14, "Check inputs,\nbuild features", "features.py")
    arrow(ax, 27.9, mid, 32.6, mid)

    # trained models used at prediction time (dashed group)
    gx, gw = 59.5, 49.0
    ax.add_patch(Rectangle((gx, cy[0] - rh / 2 - 4.2), gw, cy[-1] - cy[0] + rh + 5.4, fc="none", ec=GREY,
                           lw=0.7, ls=(0, (3, 2)), zorder=1))
    text(ax, gx + 1.5, cy[0] - rh / 2 - 3.6, "trained models, loaded from model.joblib", size=6.0, color=GREY,
         style="italic")
    models = ["Calibrated logistic regression", "20 resampled models + similarity check",
              "Contribution of each input (explain.py)", "Quantile models for waiting time"]
    outs = ["Risk (probability)", "Confidence (0–100)", "Main causes + action", "Expected wait range"]
    for c, mtxt, otxt in zip(cy, models, outs):
        arrow(ax, 55.4, mid, 60.6, c)
        node(ax, 61.0, c - rh / 2, 46.0, rh, mtxt, bold=False, tsize=6.7)
        arrow(ax, 107.4, c, 111.6, c)
        node(ax, 112.0, c - rh / 2, 29.0, rh, otxt, fill=LIGHT, tsize=6.8)
        arrow(ax, 141.4, c, 145.6, c)
    node(ax, 146.0, cy[0] - rh / 2, 25.5, cy[-1] - cy[0] + rh, "Result shown",
         "as text or JSON\n(predict.py)\n\nin the Streamlit\ndemo app\n(app.py)", fill=LIGHT, lw=1.0, ssize=6.2)

    # link: saved model (lane 1) -> trained models (lane 2)
    sx = xs[5] + w / 2
    gy = cy[0] - rh / 2 - 4.2
    ax.plot([sx, sx, gx + gw / 2], [y + h + 0.4, 37.0, 37.0], color=INK, lw=0.7, zorder=4)
    arrow(ax, gx + gw / 2, 37.0, gx + gw / 2, gy - 0.3)
    save(fig, "architecture.png")


def ground_truth():
    """Timeline of one port call with a worked example of the label."""
    fig, ax = canvas(46)
    axis_y, bar_y, bh = 31.0, 22.5, 5.0
    # before / after zones
    ax.add_patch(Rectangle((0, 1), 44, 44, fc=SHADE, ec="none"))
    text(ax, 2, 2.6, "BEFORE: known at prediction time", size=7.0, weight="bold")
    text(ax, 2, 7.0, "schedule and ships due, ships\nwaiting, berths busy, wind\nforecast, strike alert",
         size=6.6, color=GREY)
    text(ax, 2, 16.6, "= the model's inputs", size=6.6, style="italic", color=GREY)
    text(ax, 47, 2.6, "AFTER: not known yet, used only to build the label", size=7.0, weight="bold")

    ax.plot([1, 134], [axis_y, axis_y], color=INK, lw=0.7)
    marks = [(11, "prediction made\n(ETA − 48 h)"), (50, "scheduled\nETA"), (60, "actual\narrival"),
             (85, "berth free,\nservice starts"), (118, "planned\nfinish"), (130, "departure")]
    for x, lab in marks:
        ax.plot([x, x], [axis_y - 1, axis_y + 1], color=INK, lw=0.7)
        text(ax, x, axis_y + 2.2, lab, size=6.4, ha="center", linespacing=1.15)
    ax.plot([11, 11], [axis_y - 9, axis_y], color=ACCENT, lw=1.2)

    ax.add_patch(Rectangle((60, bar_y), 25, bh, fc="white", ec=ACCENT, lw=0.8, hatch="//////", hatchcolor=ACCENT))
    ax.add_patch(Rectangle((85, bar_y), 33, bh, fc=LIGHT, ec=INK, lw=0.7))
    ax.add_patch(Rectangle((118, bar_y), 12, bh, fc="white", ec=ACCENT, lw=0.8, hatch="//////", hatchcolor=ACCENT))
    text(ax, 72.5, bar_y - 1.0, "waiting at anchorage\n10 h", size=6.6, color=ACCENT, ha="center", va="bottom",
         linespacing=1.1)
    text(ax, 101.5, bar_y + bh / 2, "planned service 20 h", size=6.6, ha="center", va="center")
    text(ax, 124, bar_y - 1.0, "overrun\n4 h", size=6.6, color=ACCENT, ha="center", va="bottom", linespacing=1.1)
    text(ax, 60, 13.0, "hatched parts = delay that counts", size=6.4, color=ACCENT, style="italic")

    # worked example of the label
    bx, by, bw, bhh = 138.0, 7.0, 34.0, 36.0
    ax.add_patch(Rectangle((bx, by), bw, bhh, fc="white", ec=INK, lw=0.7))
    text(ax, bx + 2, by + 1.6, "Label for this call", size=7.2, weight="bold")
    text(ax, bx + 2, by + 6.4, "excess port time\n= waiting + overrun\n= 10 h + 4 h = 14 h", size=6.7)
    text(ax, bx + 2, by + 18.0, "14 h is more than 12 h, so", size=6.7)
    text(ax, bx + 2, by + 22.2, "CONGESTED (label 1)", size=7.2, weight="bold", color=ACCENT)
    text(ax, bx + 2, by + 27.4, "12 h or less would be\nnot congested (label 0)", size=6.4, color=GREY)
    save(fig, "ground_truth.png")


def main():
    architecture()
    ground_truth()


if __name__ == "__main__":
    main()
