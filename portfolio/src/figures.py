"""Figures + integrated robustness scorecard. Run after analysis.py."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import analysis as AN
import data
import engine as E

OUT = AN.OUT
FIG = os.path.join(OUT, "figures")
TAB = AN.TAB
os.makedirs(FIG, exist_ok=True)

# Validated categorical palette (fixed order) + text tokens; light surface
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PAL = {"VT100": "#2a78d6", "A: VT90/G10": "#eb6834", "B: VT80/G10/VGIT10": "#1baf7a",
       "C: VT80/G10/SGOV10": "#eda100", "VT80/VGIT20": "#e87ba4", "VT75/G15/VGIT10": "#008300",
       "VT70/G15/VGIT15": "#6250d6"}
LS = {"VT100": "-", "A: VT90/G10": "-", "B: VT80/G10/VGIT10": "--", "C: VT80/G10/SGOV10": ":",
      "VT80/VGIT20": "-.", "VT75/G15/VGIT10": "--", "VT70/G15/VGIT15": "-."}
SEQ = LinearSegmentedColormap.from_list("seq", ["#dbe8f8", "#2a78d6", "#0d3a73"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2491b", "#f2c3ad", "#e8e8e6", "#a9c8ef", "#1c5fae"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.8,
})

SHOW = ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10", "VT75/G15/VGIT10", "VT70/G15/VGIT15"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def port_returns(R, names):
    W = np.array([AN.CANDIDATES[n] for n in names])
    o = E.run(R, W, "A")
    return pd.DataFrame(o["ret"], index=R.index, columns=names)


def fig_growth(R):
    P = port_returns(R, SHOW)
    W = AN.INIT * (1 + P).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for c in SHOW:
        ax.plot(W.index, W[c], color=PAL[c], ls=LS[c], label=f"{c}  (${W[c].iloc[-1]/1e6:.1f}M)")
    ax.set_yscale("log")
    ax.axvspan(W.index[0], pd.Timestamp("2008-07-31"), color="#f1f0ec", zorder=0)
    ax.text(pd.Timestamp("1973-06-30"), W.values.max() * 0.8, "proxy era (simulated)", color=INK2, fontsize=8.5)
    ax.text(pd.Timestamp("2009-06-30"), W.values.max() * 0.8, "VT ETF era", color=INK2, fontsize=8.5)
    ax.set_title("Growth of $80,000, 1972-2025 (annual rebalance, log scale)")
    ax.set_ylabel("Portfolio value (USD)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "01_growth_1972_2025.png")
    # modern ETF era
    R2 = R.loc["2009-01-01":]
    P = port_returns(R2, SHOW)
    W = AN.INIT * (1 + P).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in SHOW:
        ax.plot(W.index, W[c], color=PAL[c], ls=LS[c], label=f"{c}  (${W[c].iloc[-1]/1e3:.0f}k)")
    ax.set_title("Growth of $80,000, 2009-2025 (investable-ETF era, annual rebalance)")
    ax.set_ylabel("Portfolio value (USD)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "02_growth_2009_2025.png")


def fig_dd_roll(R):
    names = ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "VT70/G15/VGIT15"]
    P = port_returns(R, names)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for c in names:
        dd = E.drawdown_series(P[c].values)
        axs[0].plot(P.index, dd * 100, color=PAL[c], ls=LS[c], lw=1.4, label=c)
        r36 = np.exp(np.log1p(P[c]).rolling(36).sum() / 3) - 1
        axs[1].plot(P.index, r36 * 100, color=PAL[c], ls=LS[c], lw=1.4, label=c)
    axs[0].set_title("Drawdown from prior peak (%)")
    axs[1].set_title("Rolling 36-month annualized return (%)")
    axs[1].axhline(0, color=INK2, lw=0.8)
    axs[0].legend(loc="lower left", fontsize=8.5, ncol=2)
    save(fig, "03_drawdown_rolling36.png")


def fig_corr(R):
    C = R[AN.A].corr()
    roll = pd.read_csv(os.path.join(TAB, "rolling36_corr.csv"), index_col=0, parse_dates=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    im = axs[0].imshow(C.values, cmap=DIV, vmin=-1, vmax=1)
    lab = ["VT", "Gold", "VGIT", "T-bill"]
    axs[0].set_xticks(range(4), lab)
    axs[0].set_yticks(range(4), lab)
    axs[0].grid(False)
    for i in range(4):
        for j in range(4):
            axs[0].text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=9)
    axs[0].set_title("Monthly correlation 1972-2025")
    cols = {"VT-Gold": "#eb6834", "VT-VGIT": "#1baf7a", "VT-TBILL": "#eda100", "Gold-VGIT": "#6250d6"}
    ls = {"VT-Gold": "-", "VT-VGIT": "--", "VT-TBILL": ":", "Gold-VGIT": "-."}
    for c, col in cols.items():
        axs[1].plot(roll.index, roll[c], color=col, ls=ls[c], lw=1.4, label=c)
    axs[1].axhline(0, color=INK2, lw=0.8)
    axs[1].set_title("Rolling 36-month correlation (regime-dependent)")
    axs[1].legend(fontsize=8.5, ncol=4, loc="lower left")
    save(fig, "04_correlations.png")


def fig_frontier(R):
    G = E.weight_grid(0.025)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (a, b), title in [(axs[0], ("1972", "2025"), "1972-2025"), (axs[1], ("1972", "2024"), "1972-2024 (ex-2025)")]:
        Rw = AN.slice_(R, a, b)
        m, _ = AN.grid_eval(Rw, G)
        sc = ax.scatter(m.Vol * 100, m.CAGR * 100, c=G[:, 1] * 100, cmap=SEQ, s=10, alpha=0.9, lw=0)
        # Pareto front (max CAGR for given vol)
        order = np.argsort(m.Vol.values)
        best, front = -1, []
        for i in order:
            if m.CAGR.values[i] > best:
                best = m.CAGR.values[i]
                front.append(i)
        ax.plot(m.Vol.values[front] * 100, m.CAGR.values[front] * 100, color=INK, lw=1.2, label="Pareto front (vol, CAGR)")
        for k in ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10", "VT70/G15/VGIT15"]:
            w = np.array(AN.CANDIDATES[k])
            j = int(np.argmin(np.abs(G - w).sum(1)))
            ax.scatter(m.Vol[j] * 100, m.CAGR[j] * 100, s=60, facecolor=PAL[k], edgecolor=SURF, lw=2, zorder=5)
            off = {"VT70/G15/VGIT15": (-60, 8), "B: VT80/G10/VGIT10": (6, 4)}.get(k, (5, -10))
            ax.annotate(k.split(":")[0] if ":" in k else k, (m.Vol[j] * 100, m.CAGR[j] * 100),
                        xytext=off, textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel("Annualized volatility (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title(f"Grid of 2.5%-step portfolios, {title}")
        ax.legend(loc="lower right", fontsize=8)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8)
    cb.set_label("Gold weight (%)")
    save(fig, "05_efficient_frontier.png")
    # CAGR vs MaxDD
    Rw = AN.slice_(R, "1972", "2025")
    m, _ = AN.grid_eval(Rw, G)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sc = ax.scatter(-m.MaxDD * 100, m.CAGR * 100, c=G[:, 2] * 100 + G[:, 3] * 100, cmap=SEQ, s=10, lw=0)
    for k in ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10", "VT70/G15/VGIT15"]:
        w = np.array(AN.CANDIDATES[k])
        j = int(np.argmin(np.abs(G - w).sum(1)))
        ax.scatter(-m.MaxDD[j] * 100, m.CAGR[j] * 100, s=60, facecolor=PAL[k], edgecolor=SURF, lw=2, zorder=5)
        off = {"VT70/G15/VGIT15": (-60, 8), "B: VT80/G10/VGIT10": (6, 4)}.get(k, (5, -10))
        ax.annotate(k.split(":")[0] if ":" in k else k, (-m.MaxDD[j] * 100, m.CAGR[j] * 100),
                    xytext=off, textcoords="offset points", fontsize=8)
    ax.set_xlabel("Maximum drawdown (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("CAGR vs max drawdown, 1972-2025 (color = total Treasury weight)")
    fig.colorbar(sc, ax=ax).set_label("VGIT + T-bill weight (%)")
    save(fig, "06_cagr_vs_mdd.png")


def fig_regimes():
    G = pd.read_csv(os.path.join(TAB, "regime_annualized.csv"), index_col=0)
    cols = ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10", "VT80/VGIT20",
            "VT85/G15", "VT75/G15/VGIT10", "VT70/G15/VGIT15", "asset GOLD", "asset VGIT", "asset TBILL"]
    M = G[cols].T * 100
    fig, ax = plt.subplots(figsize=(11, 5.4))
    lim = 30
    im = ax.imshow(M.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    import textwrap
    ax.set_xticks(range(M.shape[1]), ["\n".join(textwrap.wrap(c, 13)) for c in M.columns], fontsize=8)
    ax.set_yticks(range(M.shape[0]), [c.replace("asset ", "[asset] ") for c in M.index], fontsize=8.5)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("Annualized return by historical regime (%), annual rebalance")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("% per year (clipped at ±30)")
    save(fig, "07_regime_heatmap.png")


def fig_rebal():
    T = pd.read_csv(os.path.join(TAB, "rebalancing_comparison.csv"))
    B = pd.read_csv(os.path.join(TAB, "rebalancing_premium_bootstrap.csv"))
    pols = ["none", "A", "S", "Q", "M", "A_band5", "abs5", "rel25"]
    labels = ["Buy&hold", "Annual", "Semiann.", "Quarterly", "Monthly", "Annual+5pp band", "5pp threshold", "25% relative"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.45})
    x = np.arange(len(pols))
    for k, off, col in [("A: VT90/G10", -0.2, PAL["A: VT90/G10"]), ("B: VT80/G10/VGIT10", 0.2, PAL["B: VT80/G10/VGIT10"])]:
        s = T[(T.Window == "1972-2025") & (T.Portfolio == k)].set_index("Policy").loc[pols]
        base = s.loc["none", "CAGR"]
        axs[0].bar(x + off, (s.CAGR - base) * 1e4, width=0.38, color=col, label=k)
        for xi, (v, n) in enumerate(zip((s.CAGR - base) * 1e4, s.Trades)):
            axs[0].text(xi + off, v + (2 if v >= 0 else -6), f"{int(n)}", ha="center", fontsize=7, color=INK2)
    axs[0].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    axs[0].set_ylabel("CAGR vs buy-and-hold (bp/yr)")
    axs[0].set_title("1972-2025 rebalancing effect (labels = trade legs)")
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].legend(fontsize=8)
    b = B[B.Policy == "A_band5"].set_index("Portfolio")
    y = np.arange(len(b))
    axs[1].hlines(y, b.p05 * 1e4, b.p95 * 1e4, color="#2a78d6", lw=2)
    axs[1].scatter(b["median"] * 1e4, y, color="#2a78d6", s=40, zorder=3, label="median")
    axs[1].scatter(b.mean_premium * 1e4, y, color="#eb6834", marker="D", s=30, zorder=3, label="mean")
    axs[1].set_yticks(y, b.index, fontsize=8)
    axs[1].axvline(0, color=INK2, lw=0.8)
    axs[1].set_xlabel("Rebalancing premium vs buy-and-hold (bp/yr), 5-95% band")
    axs[1].set_title("Block-bootstrap 30y paths: annual+5pp band")
    axs[1].legend(fontsize=8)
    save(fig, "08_rebalancing.png")


def fig_mc():
    M = pd.read_csv(os.path.join(TAB, "montecarlo.csv"))
    names = ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10", "VT75/G15/VGIT10", "VT70/G15/VGIT15"]
    meths = list(M.Method.unique())
    fig, axs = plt.subplots(1, len(meths), figsize=(13, 4.4), sharey=True)
    for ax, me in zip(axs, meths):
        s = M[M.Method == me].set_index("Portfolio").loc[names]
        y = np.arange(len(names))[::-1]
        ax.hlines(y, s["P10_$"] / 1e3, s["P90_$"] / 1e3, color=[PAL[n] for n in names], lw=3)
        ax.scatter(s["Median_$"] / 1e3, y, color=[PAL[n] for n in names], s=40, edgecolor=SURF, lw=1.5, zorder=3)
        ax.scatter(s["P5_$"] / 1e3, y, marker="|", color=INK, s=60, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 500, 1000, 2000, 4000])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(me, fontsize=9.5)
        ax.set_xlabel("Nominal value at 2055 ($k, log)")
        ax.axvline(1000, color=INK2, lw=0.7, ls=":")
        ax.axvline(500, color=INK2, lw=0.7, ls=":")
    axs[0].set_yticks(np.arange(len(names))[::-1], names, fontsize=8.5)
    fig.suptitle("Monte Carlo 30y terminal wealth from $80k: median (dot), P10-P90 (bar), P5 (tick)", fontweight="bold")
    save(fig, "09_montecarlo_ranges.png")


def fig_mc_hist(R):
    inputs = AN.mc_inputs(R, 10_000, 360)
    X = inputs["2 Fwd-calibrated bootstrap"][:, :, :4]
    names = ["VT100", "A: VT90/G10", "B: VT80/G10/VGIT10", "VT70/G15/VGIT15"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.logspace(np.log10(3e4), np.log10(5e6), 70)
    for n in names:
        rp = AN.mc_simulate_paths(X, np.array(AN.CANDIDATES[n], float))
        fin = AN.INIT * np.prod(1 + rp, 1)
        h, e = np.histogram(fin, bins=bins, density=False)
        ax.step(e[1:], h / len(fin) * 100, where="pre", color=PAL[n], ls=LS[n], lw=1.6, label=f"{n} (median ${np.median(fin)/1e3:.0f}k)")
    ax.set_xscale("log")
    ax.axvline(5e5, color=INK2, lw=0.7, ls=":")
    ax.axvline(1e6, color=INK2, lw=0.7, ls=":")
    ax.set_xlabel("Nominal terminal wealth after 30y (USD, log)")
    ax.set_ylabel("% of paths per bin")
    ax.set_title("Terminal-wealth distribution, forward-calibrated block bootstrap (10,000 paths)")
    ax.legend(fontsize=8.5)
    save(fig, "10_montecarlo_hist.png")


def fig_gold():
    Gs = pd.read_csv(os.path.join(TAB, "gold_sensitivity.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    wins = ["1972-2025", "1980-2025", "2000-2025", "2009-2025", "2016-2025"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for w, c in zip(wins, cols):
        for suffix, ls in [(w, "-"), (w.replace("2025", "2024"), "--")]:
            s = Gs[(Gs.Family == "VT+Gold") & (Gs.Window == suffix)]
            axs[0].plot(s.Gold * 100, s.Sharpe, color=c, ls=ls, marker="o", ms=3, label=suffix if ls == "-" else None)
            axs[1].plot(s.Gold * 100, s.CAGR * 100, color=c, ls=ls, marker="o", ms=3, label=suffix if ls == "-" else None)
    axs[0].set_title("Sharpe vs gold weight (VT + gold); dashed = ex-2025")
    axs[1].set_title("CAGR vs gold weight (VT + gold); dashed = ex-2025")
    for ax in axs:
        ax.set_xlabel("Gold weight (%)")
    axs[0].legend(fontsize=8)
    save(fig, "11_gold_sensitivity.png")


def fig_walkforward():
    Wt = pd.read_csv(os.path.join(TAB, "walkforward_weights.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = {"max Sharpe": "#2a78d6", "max CE3": "#eb6834", "max CAGR|MDD<=35%": "#1baf7a", "max Sharpe (rolling 20y)": "#6250d6"}
    ls = {"max Sharpe": "-", "max CE3": "--", "max CAGR|MDD<=35%": ":", "max Sharpe (rolling 20y)": "-."}
    for k, c in cols.items():
        g = Wt[k].str.extract(r"G(\d+)")[0].fillna(0).astype(float)
        ax.step(Wt.index, g, where="post", color=c, ls=ls[k], label=k)
    ax.set_ylabel("Gold weight chosen (%)")
    ax.set_title("Walk-forward: gold weight chosen each January using only past data")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "12_walkforward_gold.png")


def scorecard(R):
    """Integrated scorecard: historical windows (incl. ex-2025) + MC methods + regimes."""
    names = list(AN.CANDIDATES)
    W = np.array([AN.CANDIDATES[n] for n in names])
    rows = {n: {} for n in names}
    for wname, (a, b) in {**AN.WINDOWS, **AN.WINDOWS_X25}.items():
        Rw = AN.slice_(R, a, b)
        o = E.run(Rw, W, "A")
        m = E.metrics(o["ret"], Rw.RF.values)
        for j, n in enumerate(names):
            rows[n][f"CAGR {wname}"] = m.CAGR[j]
            rows[n][f"Sharpe {wname}"] = m.Sharpe[j]
            rows[n][f"MDD {wname}"] = m.MaxDD[j]
    S = pd.DataFrame(rows).T
    M = pd.read_csv(os.path.join(TAB, "montecarlo.csv"))
    for me in M.Method.unique():
        s = M[M.Method == me].set_index("Portfolio")
        S[f"MC med {me}"] = s["Median_$"]
        S[f"MC P5 {me}"] = s["P5_$"]
        S[f"MC P(MDD>40) {me}"] = s["P(MDD>40%)"]
    G = pd.read_csv(os.path.join(TAB, "regime_annualized.csv"), index_col=0)
    for rg in G.index:
        S[f"Regime {rg}"] = G.loc[rg, names].values
    # summary statistics
    cagr_cols = [c for c in S if c.startswith("CAGR ")]
    sh_cols = [c for c in S if c.startswith("Sharpe ")]
    med_cols = [c for c in S if c.startswith("MC med")]
    p5_cols = [c for c in S if c.startswith("MC P5")]
    reg_cols = [c for c in S if c.startswith("Regime")]
    summ = pd.DataFrame({
        "Median hist CAGR (14 windows)": S[cagr_cols].median(1),
        "Min hist CAGR": S[cagr_cols].min(1),
        "Median Sharpe (14 windows)": S[sh_cols].median(1),
        "Worst MDD (any window)": S[[c for c in S if c.startswith("MDD ")]].min(1),
        "Max CAGR regret (pp)": (S[cagr_cols].max(0) - S[cagr_cols]).max(1) * 100,
        "MC median: worst-method ratio to best": (S[med_cols] / S[med_cols].max(0)).min(1),
        "MC median: mean ratio to best": (S[med_cols] / S[med_cols].max(0)).mean(1),
        "MC P5: mean ratio to best": (S[p5_cols] / S[p5_cols].max(0)).mean(1),
        "MC P(MDD>40%) avg": S[[c for c in S if c.startswith("MC P(MDD")]].mean(1),
        "Worst regime ann. return": S[reg_cols].min(1),
        "n ETFs": (W > 0).sum(1),
    }, index=names)
    S.to_csv(os.path.join(TAB, "scorecard_full.csv"))
    summ.to_csv(os.path.join(TAB, "scorecard_summary.csv"))
    return summ


if __name__ == "__main__":
    R, _ = data.build_panel(verbose=False)
    fig_growth(R)
    fig_dd_roll(R)
    fig_corr(R)
    fig_frontier(R)
    fig_regimes()
    fig_rebal()
    fig_mc()
    fig_mc_hist(R)
    fig_gold()
    fig_walkforward()
    s = scorecard(R)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(s.round(3).to_string())
