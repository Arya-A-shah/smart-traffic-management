"""All graphs (matplotlib -> PNG). Used by the dashboard page and by the game."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from algorithms import ALGORITHMS, NAMES  # noqa: E402

DIR_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7"]          # N E S W
ALGO_COLORS = {"fixed": "#9ca3af", "actuated": "#60a5fa", "max_pressure": "#34d399", "rl": "#f472b6"}
BG, FG, GRID = "#111827", "#e5e7eb", "#374151"


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _empty(msg):
    fig, ax = plt.subplots(figsize=(7, 3))
    _style(ax, "")
    ax.text(0.5, 0.5, msg, color=FG, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    return _png(fig)


# ------------------------------------------------------------------ live run
def live_queues(run):
    ts = run.timeseries
    if len(ts) < 2:
        return _empty("Start the game to see live data")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _style(ax, f"Queue length per approach  ·  {ALGORITHMS[run.algo]['name']}")
    t = [r["t"] for r in ts]
    for k in range(4):
        ax.plot(t, [r["queue"][k] for r in ts], color=DIR_COLORS[k], label=NAMES[k], linewidth=1.8)
    # shade emergency overrides
    for d in run.decisions:
        if d["source"] == "emergency":
            ax.axvline(d["t"], color="#ef4444", alpha=0.35, linewidth=2)
    ax.set_xlabel("time (s)", color=FG); ax.set_ylabel("vehicles stopped", color=FG)
    leg = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, ncol=4, loc="upper left")
    leg.set_title("red lines = ambulance override", prop={"size": 8})
    leg.get_title().set_color("#fca5a5")
    return _png(fig)


def live_waits(run):
    vs = run.vehicles
    if not vs:
        return _empty("No vehicle has crossed the junction yet")
    types = ["car", "bike", "auto", "bus", "truck", "ambulance"]
    avg = []
    for ty in types:
        w = [v["wait"] for v in vs if v["type"] == ty]
        avg.append(sum(w) / len(w) if w else 0)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    _style(ax, "Average waiting time by vehicle type (s)")
    cols = ["#94a3b8"] * 5 + ["#ef4444"]
    bars = ax.bar(types, avg, color=cols, width=0.6)
    for b, a in zip(bars, avg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{a:.1f}", ha="center",
                va="bottom", color=FG, fontsize=9)
    return _png(fig)


# ------------------------------------------------------------------ across runs in the game
def runs_compare(runs):
    runs = [r for r in runs if r.vehicles]
    if not runs:
        return _empty("Play the game with different algorithms to compare runs")
    runs = runs[-8:]
    labels = [f"#{r.n} {ALGORITHMS[r.algo]['name']}" + ("" if r.ev_priority else "\n(no amb. rule)") for r in runs]
    s = [r.summary() for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _style(axes[0], "Avg waiting time, normal vehicles (s)")
    _style(axes[1], "Avg ambulance waiting time (s)")
    cols = [ALGO_COLORS[r.algo] for r in runs]
    axes[0].barh(labels, [x["avg_wait"] for x in s], color=cols)
    axes[1].barh(labels, [x["amb_wait"] or 0 for x in s], color=cols)
    for ax in axes:
        ax.invert_yaxis()
    axes[1].set_yticklabels([])
    fig.tight_layout()
    return _png(fig)


# ------------------------------------------------------------------ quick fair comparison
def compare_chart(results, title_extra=""):
    algos = [r["algo"] for r in results]
    names = [ALGORITHMS[a]["name"] for a in algos]
    cols = [ALGO_COLORS[a] for a in algos]
    fig = plt.figure(figsize=(11, 6.2), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.35)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    for ax, key, title in ((ax1, "avg_wait", "Avg wait, normal vehicles (s)"),
                           (ax2, "amb_wait", "Avg ambulance wait (s)"),
                           (ax3, "max_queue", "Longest queue (vehicles)")):
        _style(ax, title)
        vals = [(r[key] or 0) for r in results]
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.65)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        best = min(range(len(vals)), key=lambda i: vals[i])
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}" + (" ★" if i == best else ""),
                    ha="center", va="bottom", color=FG, fontsize=9)
    ax4 = fig.add_subplot(gs[1, :])
    _style(ax4, "Total vehicles stopped at the junction over time (same traffic for every algorithm)")
    for r, c, n in zip(results, cols, names):
        t = [row[0] for row in r["series"]]
        ax4.plot(t, [sum(row[1:]) for row in r["series"]], color=c, label=n, linewidth=1.8)
    ax4.set_xlabel("time (s)", color=FG)
    ax4.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, ncol=4)
    fig.suptitle("Same traffic, different algorithms" + title_extra, color=FG, fontsize=14, x=0.01, ha="left")
    return _png(fig)
