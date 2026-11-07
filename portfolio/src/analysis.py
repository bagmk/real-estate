"""Full analysis pipeline: backtests, regimes, correlations, rebalancing, optimization,
walk-forward, bootstrap, Monte Carlo, stress tests. Writes tables + figures to ../output.

Run:  python3 analysis.py            (about 2-4 minutes on a laptop)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import data
import engine as E

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
SEED = 20261106
INIT = 80_000.0
A = E.ASSETS

NAMED = {
    "VT100": [1.0, 0, 0, 0],
    "A: VT90/G10": [0.9, 0.1, 0, 0],
    "B: VT80/G10/VGIT10": [0.8, 0.1, 0.1, 0],
    "C: VT80/G10/SGOV10": [0.8, 0.1, 0, 0.1],
    "VT80/VGIT20": [0.8, 0, 0.2, 0],
    "VT60/VGIT40": [0.6, 0, 0.4, 0],
}

WINDOWS = {
    "1972-2025": ("1972", "2025"), "1980-2025": ("1980", "2025"), "2000-2025": ("2000", "2025"),
    "2009-2025": ("2009", "2025"), "2010-2019": ("2010", "2019"), "2016-2025": ("2016", "2025"),
    "2020-2025": ("2020", "2025"),
}
WINDOWS_X25 = {k.replace("2025", "2024"): (a, "2024") for k, (a, b) in WINDOWS.items() if b == "2025"}

# Historical regime episodes (dates chosen from NBER peaks/troughs, CPI and yield history;
# defined by macro characteristics, not by portfolio outcomes).
REGIMES = {
    "1 Deflationary recession": [("2000-09", "2002-09"), ("2020-02", "2020-03")],
    "2 Stagflation": [("1973-01", "1974-12"), ("1980-01", "1982-07")],
    "3 Inflation shock": [("1977-01", "1980-03"), ("2021-04", "2022-09")],
    "4 Tech-led bull": [("1995-01", "1999-12"), ("2023-01", "2024-12")],
    "5 Gold bull / weak equity": [("2001-01", "2011-08")],
    "6 Rising rates": [("1994-01", "1994-12"), ("2022-01", "2023-10")],
    "7 Financial crisis": [("2007-11", "2009-02"), ("1987-09", "1987-11")],
    "8 Normal expansion": [("1983-01", "1989-12"), ("2003-04", "2007-10"), ("2012-01", "2019-12")],
}

RES = {}


def slice_(R, a, b):
    return R.loc[f"{a}-01-01" if len(a) == 4 else a: f"{b}-12-31" if len(b) == 4 else b]


def bt(Rw, W, policy="A"):
    o = E.run(Rw, np.array(W), policy)
    m = E.metrics(o["ret"], Rw.RF.values, Rw.CPI.values, INIT)
    m["Turnover_total"] = o["turnover"].sum(0)
    m["Trades"] = o["trades"]
    return m, o


def ce_return(r, gamma):
    """Annualized certainty-equivalent return of CRRA utility on monthly returns."""
    r = np.asarray(r)
    if gamma == 1:
        return np.exp(np.log1p(r).mean(0) * 12) - 1
    u = ((1 + r) ** (1 - gamma)).mean(0)
    return u ** (12 / (1 - gamma)) - 1


# ----------------------------------------------------------------------------- 1. data
def sec_data():
    R, S = data.build_panel(verbose=True)
    v, vr, dd = data.validate()
    v.to_csv(os.path.join(TAB, "proxy_validation.csv"), index=False)
    seg = {}
    for c in S.columns:
        g = S[c].groupby((S[c] != S[c].shift()).cumsum())
        seg[c] = [(x.iloc[0], f"{x.index[0]:%Y-%m}", f"{x.index[-1]:%Y-%m}") for _, x in g]
    RES["segments"] = seg
    RES["validation"] = v.assign(start=v.start.astype(str), end=v.end.astype(str)).to_dict("records")
    RES["gold_avg_vol_ratio"] = vr
    RES["damodaran_gold_corr"] = float(dd.corr().loc["dam_gold", "our_gold"])
    stats = pd.DataFrame({
        "CAGR": (1 + R[A]).prod() ** (12 / len(R)) - 1,
        "Vol": R[A].std() * np.sqrt(12),
        "MaxDD": [E.drawdown_series(R[a]).min() for a in A],
    })
    stats.to_csv(os.path.join(TAB, "asset_stats_1972_2025.csv"))
    RES["asset_stats"] = stats.round(4).to_dict()
    R.to_csv(os.path.join(TAB, "monthly_returns_panel.csv"))
    S.to_csv(os.path.join(TAB, "monthly_source_flags.csv"))
    return R, S


# ----------------------------------------------------------------------------- 2. windows
def sec_windows(R):
    rows = []
    for wname, (a, b) in {**WINDOWS, **WINDOWS_X25}.items():
        Rw = slice_(R, a, b)
        m, _ = bt(Rw, list(NAMED.values()))
        m.insert(0, "Portfolio", list(NAMED.keys()))
        m.insert(0, "Window", wname)
        rows.append(m)
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(os.path.join(TAB, "window_backtests.csv"), index=False)
    RES["windows"] = T.round(4).to_dict("records")
    return T


# ----------------------------------------------------------------------------- 3. correlations
def sec_corr(R):
    X = R[A]
    out = {"full": X.corr()}
    vt = X["VT"]
    infl_yoy = (1 + R.CPI).rolling(12).apply(np.prod, raw=True) - 1
    y5 = data._yahoo_monthly("FVX", field="close").reindex(R.index) / 100
    dy = y5.diff(12)
    wealth = (1 + vt).cumprod()
    vt_dd = wealth / wealth.cummax() - 1
    masks = {
        "VT worst-decile months": vt <= vt.quantile(0.10),
        "VT drawdown > 20%": vt_dd < -0.20,
        "CPI YoY > 5%": infl_yoy > 0.05,
        "CPI YoY < 2.5%": infl_yoy < 0.025,
        "5y yield +100bp (12m)": dy > 0.01,
        "5y yield -100bp (12m)": dy < -0.01,
        "1972-1999": (R.index.year < 2000),
        "2000-2020": (R.index.year >= 2000) & (R.index.year <= 2020),
        "2021-2025": R.index.year >= 2021,
    }
    cond = []
    for k, m in masks.items():
        c = X[m].corr()
        cond.append({"Condition": k, "N_months": int(m.sum()),
                     "VT-Gold": c.loc["VT", "GOLD"], "VT-VGIT": c.loc["VT", "VGIT"],
                     "VT-TBILL": c.loc["VT", "TBILL"], "Gold-VGIT": c.loc["GOLD", "VGIT"],
                     "Mean VT %/m": X.loc[m, "VT"].mean() * 100,
                     "Mean Gold %/m": X.loc[m, "GOLD"].mean() * 100,
                     "Mean VGIT %/m": X.loc[m, "VGIT"].mean() * 100,
                     "Mean TBILL %/m": X.loc[m, "TBILL"].mean() * 100})
    C = pd.DataFrame(cond)
    C.to_csv(os.path.join(TAB, "conditional_correlations.csv"), index=False)
    out["full"].to_csv(os.path.join(TAB, "correlation_full.csv"))
    # hedge hit-rate in VT worst-decile months
    m = masks["VT worst-decile months"]
    hit = {a: float((X.loc[m, a] > 0).mean()) for a in ["GOLD", "VGIT", "TBILL"]}
    # in worst-decile months, split by inflation regime
    split = []
    for lab, im in [("CPI YoY>4%", infl_yoy > 0.04), ("CPI YoY<=4%", infl_yoy <= 0.04)]:
        mm = m & im
        split.append({"Inflation": lab, "N": int(mm.sum()),
                      **{f"{a} mean %": X.loc[mm, a].mean() * 100 for a in ["VT", "GOLD", "VGIT", "TBILL"]},
                      **{f"{a} hit": (X.loc[mm, a] > 0).mean() for a in ["GOLD", "VGIT"]}})
    Sp = pd.DataFrame(split)
    Sp.to_csv(os.path.join(TAB, "crash_months_by_inflation.csv"), index=False)
    roll = pd.DataFrame({
        "VT-Gold": X.VT.rolling(36).corr(X.GOLD), "VT-VGIT": X.VT.rolling(36).corr(X.VGIT),
        "VT-TBILL": X.VT.rolling(36).corr(X.TBILL), "Gold-VGIT": X.GOLD.rolling(36).corr(X.VGIT)})
    roll.to_csv(os.path.join(TAB, "rolling36_corr.csv"))
    RES["corr_full"] = out["full"].round(3).to_dict()
    RES["corr_cond"] = C.round(3).to_dict("records")
    RES["crash_hit"] = hit
    RES["crash_by_infl"] = Sp.round(3).to_dict("records")
    RES["roll_corr_range"] = {c: [float(roll[c].min()), float(roll[c].max())] for c in roll}
    return out["full"], C, roll


# ----------------------------------------------------------------------------- 4. regimes
def sec_regimes(R, extra=None):
    ports = dict(NAMED)
    if extra:
        ports.update(extra)
    rows = []
    for rg, eps in REGIMES.items():
        for a, b in eps:
            Rw = R.loc[a: pd.Timestamp(b) + pd.offsets.MonthEnd(0)]
            o = E.run(Rw, np.array(list(ports.values())), "A")
            cum = np.prod(1 + o["ret"], 0) - 1
            dd = [E.drawdown_series(o["ret"][:, j]).min() for j in range(len(ports))]
            assets = (1 + Rw[A]).prod() - 1
            infl = np.prod(1 + Rw.CPI.values) - 1
            rows.append({"Regime": rg, "Episode": f"{a}..{b}", "Months": len(Rw),
                         **{f"ret {k}": v for k, v in zip(ports, cum)},
                         **{f"mdd {k}": v for k, v in zip(ports, dd)},
                         **{f"asset {k}": assets[k] for k in A}, "CPI": infl})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, "regime_episodes.csv"), index=False)
    # aggregate per regime: annualized return (geometric across that regime's months)
    agg = []
    for rg, eps in REGIMES.items():
        Rw = pd.concat([R.loc[a: pd.Timestamp(b) + pd.offsets.MonthEnd(0)] for a, b in eps])
        o = E.run(Rw, np.array(list(ports.values())), "A")
        ann = np.prod(1 + o["ret"], 0) ** (12 / len(Rw)) - 1
        agg.append({"Regime": rg, "Months": len(Rw), **dict(zip(ports, ann)),
                    **{f"asset {k}": (1 + Rw[k]).prod() ** (12 / len(Rw)) - 1 for k in A}})
    G = pd.DataFrame(agg).set_index("Regime")
    G.to_csv(os.path.join(TAB, "regime_annualized.csv"))
    RES["regime_episodes"] = T.round(4).to_dict("records")
    RES["regime_ann"] = G.round(4).reset_index().to_dict("records")
    return T, G


# ----------------------------------------------------------------------------- 5. rebalancing
POLICIES = ["none", "A", "S", "Q", "M", "A_band5", "abs5", "rel25"]


def sec_rebal(R):
    rows = []
    test = {k: NAMED[k] for k in ["A: VT90/G10", "B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10"]}
    test["VT70/G15/VGIT15"] = [0.7, 0.15, 0.15, 0]
    for wname in ["1972-2025", "1980-2025", "2000-2025", "2009-2025", "2010-2019", "2016-2025"]:
        a, b = WINDOWS[wname]
        Rw = slice_(R, a, b)
        for pol in POLICIES:
            m, o = bt(Rw, list(test.values()), pol)
            m.insert(0, "Policy", pol)
            m.insert(0, "Portfolio", list(test))
            m.insert(0, "Window", wname)
            m["TC_paid_$_per_$80k"] = None
            rows.append(m)
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(os.path.join(TAB, "rebalancing_comparison.csv"), index=False)
    RES["rebal"] = T.drop(columns="TC_paid_$_per_$80k").round(4).to_dict("records")

    # Diversification-return decomposition (Booth-Fama/Willenbrock):
    # DR ~= 0.5*(sum w_i s_i^2 - s_p^2), annualized from monthly
    dec = []
    for k, w in test.items():
        w = np.array(w)
        X = R[A].values
        s2 = X.var(0, ddof=1) * 12
        sp2 = (X @ w).var(ddof=1) * 12
        dr = 0.5 * (w @ s2 - sp2)
        dec.append({"Portfolio": k, "Theoretical_div_return": dr})
    RES["div_return_theory"] = dec

    # Stationary block bootstrap of the rebalancing premium (Annual vs None, abs5 vs None)
    rng = np.random.default_rng(SEED)
    X = R[A].values
    T_ = len(X)
    n_b = 2000
    horizon = 360
    idx = E.stationary_bootstrap_idx(T_, horizon, 24, rng, n_b)
    prem = {k: {"A": [], "abs5": [], "A_band5": []} for k in test}
    W = np.array(list(test.values()))
    months = np.tile(np.arange(1, 13), horizon // 12 + 1)[:horizon]
    for i in range(n_b):
        Xi = X[idx[i]]
        E.simulate.months = months
        base = E.simulate(Xi, W, "none")["ret"]
        g0 = np.prod(1 + base, 0) ** (12 / horizon)
        for pol in ["A", "abs5", "A_band5"]:
            rr = E.simulate(Xi, W, pol)["ret"]
            g1 = np.prod(1 + rr, 0) ** (12 / horizon)
            for j, k in enumerate(test):
                prem[k][pol].append(g1[j] - g0[j])
    bs = []
    for k in test:
        for pol in ["A", "abs5", "A_band5"]:
            x = np.array(prem[k][pol])
            bs.append({"Portfolio": k, "Policy": pol, "mean_premium": x.mean(), "median": np.median(x),
                       "p05": np.quantile(x, 0.05), "p95": np.quantile(x, 0.95), "P(>0)": (x > 0).mean()})
    B = pd.DataFrame(bs)
    B.to_csv(os.path.join(TAB, "rebalancing_premium_bootstrap.csv"), index=False)
    RES["rebal_bootstrap"] = B.round(5).to_dict("records")
    return T, B


# ----------------------------------------------------------------------------- 6. optimization
def grid_eval(Rw, G, policy="A"):
    o = E.run(Rw, G, policy)
    m = E.metrics(o["ret"], Rw.RF.values, Rw.CPI.values, INIT)
    m["CE3"] = ce_return(o["ret"], 3)
    m["CE5"] = ce_return(o["ret"], 5)
    return m, o["ret"]


def optimize_table(m, G):
    """Return dict objective -> weight vector (or None if infeasible)."""
    res = {}
    res["A: max CAGR"] = G[m.CAGR.idxmax()]
    res["B1: max Sharpe"] = G[m.Sharpe.idxmax()]
    res["B2: max Sortino"] = G[m.Sortino.idxmax()]
    for lim in [0.20, 0.25, 0.30, 0.35, 0.40]:
        ok = m.MaxDD >= -lim
        res[f"C: max CAGR | MDD<={int(lim*100)}%"] = G[m.CAGR[ok].idxmax()] if ok.any() else None
    # D: min CVaR subject to CAGR within 0.5pp of the grid max
    ok = m.CAGR >= m.CAGR.max() - 0.005
    res["D: min CVaR | CAGR>=max-0.5pp"] = G[m.CVaR95_m[ok].idxmax()]
    res["D': min CVaR (unconstrained)"] = G[m.CVaR95_m.idxmax()]
    return res


def sec_optimize(R):
    G = E.weight_grid(0.025)
    RES["grid_size"] = len(G)
    tabs = []
    evals = {}
    for wname in ["1972-2025", "1972-2024", "1980-2025", "1980-2024", "2000-2025", "2000-2024",
                  "2009-2025", "2009-2024", "2010-2019", "2016-2025", "2016-2024"]:
        a, b = wname.split("-")
        Rw = slice_(R, a, b)
        m, _ = grid_eval(Rw, G)
        evals[wname] = m
        for obj, w in optimize_table(m, G).items():
            row = {"Window": wname, "Objective": obj}
            if w is None:
                row.update({"Weights": "INFEASIBLE"})
            else:
                j = int(np.where((G == w).all(1))[0][0])
                row.update({"Weights": E.wlabel(w), **dict(zip(A, w)),
                            "CAGR": m.CAGR[j], "Vol": m.Vol[j], "MaxDD": m.MaxDD[j],
                            "Sharpe": m.Sharpe[j], "Sortino": m.Sortino[j], "CVaR95_m": m.CVaR95_m[j]})
            tabs.append(row)
    T = pd.DataFrame(tabs)
    T.to_csv(os.path.join(TAB, "optimization_by_window.csv"), index=False)
    RES["opt"] = T.round(4).to_dict("records")
    # min achievable MDD in each window (feasibility report)
    RES["min_mdd"] = {k: float(v.MaxDD.max()) for k, v in evals.items()}

    # ---- E: robust allocation = minimax regret of CE(gamma=3) over all windows (incl. ex-2025)
    # plus regime aggregates; complexity penalty 0.05%/yr per ETF beyond two.
    wins = list({**WINDOWS, **WINDOWS_X25}.keys())
    CE = pd.DataFrame({k: grid_eval(slice_(R, *k.split("-")), G)[0].CE3 for k in wins})
    reg = []
    for rg, eps in REGIMES.items():
        Rw = pd.concat([R.loc[a: pd.Timestamp(b) + pd.offsets.MonthEnd(0)] for a, b in eps])
        o = E.run(Rw, G, "A")
        reg.append(pd.Series(ce_return(o["ret"], 3), name=rg))
    CEr = pd.concat(reg, axis=1)
    regret_w = CE.max(0) - CE
    regret_r = CEr.max(0) - CEr
    n_etf = (G > 1e-9).sum(1)
    penalty = 0.0005 * np.maximum(n_etf - 2, 0)
    score = pd.concat([regret_w, regret_r], axis=1)
    worst = score.max(1) + penalty
    avg = score.mean(1) + penalty
    combo = 0.5 * worst / worst.std() + 0.5 * avg / avg.std()
    rob = pd.DataFrame({"weights": [E.wlabel(g) for g in G], "max_regret": worst, "mean_regret": avg,
                        "combo": combo, "n_etf": n_etf})
    rob = rob.sort_values("combo")
    rob.head(25).to_csv(os.path.join(TAB, "robust_top25.csv"))
    RES["robust_top10"] = rob.head(10).round(5).to_dict("records")
    RES["robust_best"] = G[rob.index[0]].tolist()
    RES["robust_minimax"] = G[worst.idxmin()].tolist()
    # complexity check: best 2-ETF robust vs best overall
    two = rob[rob.n_etf <= 2].iloc[0]
    RES["robust_best_2etf"] = {"weights": two.weights, "combo": float(two.combo),
                               "max_regret": float(two.max_regret), "mean_regret": float(two.mean_regret)}
    return G, evals, T, rob


def sec_bootstrap_weights(R, G, n_b=500):
    """Weight stability: re-optimize on stationary-bootstrap resamples of 1972-2025."""
    rng = np.random.default_rng(SEED + 1)
    X = R[A].values
    rf = R.RF.values
    T_ = len(X)
    idx = E.stationary_bootstrap_idx(T_, T_, 24, rng, n_b)
    months = np.tile(np.arange(1, 13), T_ // 12 + 1)[:T_]
    picks = {"max Sharpe": [], "max CE3": [], "max CAGR|MDD<=35%": [], "max CAGR": []}
    for i in range(n_b):
        E.simulate.months = months
        r = E.simulate(X[idx[i]], G, "A")["ret"]
        ex = r - rf[idx[i]][:, None]
        sh = ex.mean(0) / ex.std(0)
        w = np.cumprod(1 + r, 0)
        mdd = (w / np.maximum.accumulate(np.maximum(w, 1), 0) - 1).min(0)
        cagr = w[-1] ** (12 / T_) - 1
        ce = ce_return(r, 3)
        picks["max Sharpe"].append(G[np.argmax(sh)])
        picks["max CE3"].append(G[np.argmax(ce)])
        ok = mdd >= -0.35
        picks["max CAGR|MDD<=35%"].append(G[np.argmax(np.where(ok, cagr, -9))] if ok.any() else np.full(4, np.nan))
        picks["max CAGR"].append(G[np.argmax(cagr)])
    rows = []
    for k, v in picks.items():
        v = np.array(v)
        for j, a in enumerate(A):
            q = np.nanquantile(v[:, j], [0.1, 0.5, 0.9])
            rows.append({"Objective": k, "Asset": a, "mean": np.nanmean(v[:, j]), "p10": q[0], "median": q[1], "p90": q[2]})
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(TAB, "bootstrap_weight_stability.csv"), index=False)
    RES["boot_weights"] = B.round(3).to_dict("records")
    return B, picks


# ----------------------------------------------------------------------------- 7. walk-forward
def sec_walkforward(R, G):
    """Expanding-window walk-forward: each December, choose weights on data up to that month,
    hold for the next 12 months (annual rebalance to the new target). OOS 1990-2025."""
    years = range(1990, 2026)
    rules = ["max Sharpe", "max CE3", "max CAGR|MDD<=35%", "max Sharpe (rolling 20y)"]
    chosen = {k: {} for k in rules}
    for y in years:
        train = R.loc[: f"{y-1}-12-31"]
        m, _ = grid_eval(train, G)
        chosen["max Sharpe"][y] = G[m.Sharpe.idxmax()]
        chosen["max CE3"][y] = G[m.CE3.idxmax()]
        ok = m.MaxDD >= -0.35
        chosen["max CAGR|MDD<=35%"][y] = G[m.CAGR[ok].idxmax()] if ok.any() else G[m.MaxDD.idxmax()]
        tr20 = R.loc[f"{y-20}-01-01": f"{y-1}-12-31"]
        m20, _ = grid_eval(tr20, G)
        chosen["max Sharpe (rolling 20y)"][y] = G[m20.Sharpe.idxmax()]
    oos = {}
    for k in rules:
        rets = []
        for y in years:
            Ry = R.loc[f"{y}-01-01": f"{y}-12-31"]
            o = E.run(Ry, chosen[k][y][None, :], "M" if False else "none")  # rebalanced at start of year
            rets.append(pd.Series(o["ret"][:, 0], index=Ry.index))
        oos[k] = pd.concat(rets)
    Roos = R.loc["1990-01-01":"2025-12-31"]
    for k, w in NAMED.items():
        o = E.run(Roos, np.array([w]), "A")
        oos[k] = pd.Series(o["ret"][:, 0], index=Roos.index)
    O = pd.DataFrame(oos)
    m = E.metrics(O.values, Roos.RF.values, Roos.CPI.values, INIT)
    m.index = O.columns
    m["CE3"] = ce_return(O.values, 3)
    m.to_csv(os.path.join(TAB, "walkforward_oos_1990_2025.csv"))
    Wt = pd.DataFrame({k: {y: E.wlabel(w) for y, w in v.items()} for k, v in chosen.items()})
    Wt.to_csv(os.path.join(TAB, "walkforward_weights.csv"))
    gold_path = pd.DataFrame({k: {y: w[1] for y, w in v.items()} for k, v in chosen.items()})
    RES["wf_metrics"] = m.round(4).reset_index().to_dict("records")
    RES["wf_weights_sample"] = Wt.loc[[1990, 2000, 2010, 2020, 2025]].to_dict()
    return O, m, Wt, gold_path


# ----------------------------------------------------------------------------- 8. Monte Carlo
def mc_simulate_paths(X, W, policy_months=12):
    """X : (n, T, N) monthly returns; W : (N,). Annual rebalance. Returns (n,T) port returns."""
    n, T, N = X.shape
    h = np.tile(W, (n, 1)).astype(float)
    rp = np.empty((n, T))
    for t in range(T):
        v0 = h.sum(1)
        h = h * (1 + X[:, t])
        v1 = h.sum(1)
        if (t + 1) % policy_months == 0:
            dw = np.abs(h / v1[:, None] - W).sum(1)
            v1 = v1 * (1 - E.TC_BPS / 1e4 * dw)
            h = W[None, :] * v1[:, None]
        rp[:, t] = v1 / v0 - 1
    return rp


FWD = {  # forward-looking annual geometric return assumptions (nominal), see report
    "VT": 0.060, "GOLD": 0.030, "VGIT": 0.044, "TBILL": 0.0325, "CPI": 0.0275}


def mc_inputs(R, n=10_000, T=360):
    rng = np.random.default_rng(SEED + 2)
    cols = A + ["CPI"]
    H = R[cols].values
    Th = len(H)
    out = {}
    # (1) historical stationary block bootstrap (mean block 24m)
    idx = E.stationary_bootstrap_idx(Th, T, 24, rng, n)
    out["1 Hist. block bootstrap"] = H[idx]
    # (2) forward-calibrated bootstrap: same blocks, each asset's log-mean shifted to FWD
    L = np.log1p(H)
    shift = np.log1p(np.array([FWD[c] for c in cols])) / 12 - L.mean(0)
    out["2 Fwd-calibrated bootstrap"] = np.expm1(L[idx] + shift)
    # (3) regime-switching: Markov chain on empirical states, months sampled within state
    infl_yoy = (1 + R.CPI).rolling(12).apply(np.prod, raw=True) - 1
    w = (1 + R.VT).cumprod()
    dd = w / w.cummax() - 1
    state = np.where(dd < -0.15, 2, np.where(infl_yoy.fillna(0.03) > 0.05, 1, 0))
    P = np.zeros((3, 3))
    for a, b in zip(state[:-1], state[1:]):
        P[a, b] += 1
    P = P / P.sum(1, keepdims=True)
    pools = [np.where(state == s)[0] for s in range(3)]
    st = np.empty((n, T), int)
    st[:, 0] = rng.choice(3, n, p=np.bincount(state, minlength=3) / len(state))
    u = rng.random((n, T))
    cP = P.cumsum(1)
    for t in range(1, T):
        st[:, t] = (u[:, t][:, None] > cP[st[:, t - 1]]).sum(1)
    Xr = np.empty((n, T, len(cols)))
    for s in range(3):
        msk = st == s
        Xr[msk] = H[pools[s][rng.integers(0, len(pools[s]), msk.sum())]]
    out["3 Regime-switching"] = Xr
    RES["mc_regime_P"] = P.round(4).tolist()
    RES["mc_regime_counts"] = np.bincount(state, minlength=3).tolist()
    # (4) parametric multivariate-t with parameter + correlation uncertainty
    vol = np.array([0.16, 0.16, 0.055, 0.008, 0.012])
    mu_sd = np.array([0.015, 0.03, 0.0075, 0.01, 0.0075])
    base = np.array([FWD[c] for c in cols])
    C_neg = np.array([[1, 0.10, -0.20, 0.0, -0.05], [0.10, 1, 0.30, 0.0, 0.10], [-0.20, 0.30, 1, 0.15, -0.20],
                      [0.0, 0.0, 0.15, 1, 0.30], [-0.05, 0.10, -0.20, 0.30, 1]])
    C_pos = C_neg.copy()
    C_pos[0, 2] = C_pos[2, 0] = 0.35
    C_pos[0, 1] = C_pos[1, 0] = 0.20
    C_pos[2, 4] = C_pos[4, 2] = -0.35
    Lneg, Lpos = np.linalg.cholesky(C_neg), np.linalg.cholesky(C_pos)
    regime_pos = rng.random(n) < 0.5  # half the paths live in an inflationary +corr world
    mu_path = base + rng.standard_normal((n, len(cols))) * mu_sd
    df_ = 5
    z = rng.standard_normal((n, T, len(cols)))
    chi = rng.chisquare(df_, (n, T, 1))
    z = z / np.sqrt(chi / df_) * np.sqrt((df_ - 2) / df_)
    eps = np.where(regime_pos[:, None, None], z @ Lpos.T, z @ Lneg.T)
    sig_m = vol / np.sqrt(12)
    mu_log_m = np.log1p(mu_path)[:, None, :] / 12
    out["4 Parametric t + param. uncertainty"] = np.expm1(mu_log_m + eps * sig_m)
    return out


def sec_montecarlo(R, ports, n=10_000, T=360):
    inputs = mc_inputs(R, n, T)
    rows = []
    dists = {}
    for meth, X in inputs.items():
        Xa, cpi = X[:, :, :4], X[:, :, 4]
        defl = np.prod(1 + cpi, 1)
        for k, w in ports.items():
            rp = mc_simulate_paths(Xa, np.array(w, float))
            wealth = INIT * np.cumprod(1 + rp, 1)
            fin = wealth[:, -1]
            real = fin / defl
            pk = np.maximum.accumulate(np.concatenate([np.full((n, 1), INIT), wealth], 1), 1)[:, 1:]
            mdd = (wealth / pk - 1).min(1)
            rows.append({"Method": meth, "Portfolio": k,
                         "Median_$": np.median(fin), "P10_$": np.quantile(fin, 0.10), "P90_$": np.quantile(fin, 0.90),
                         "P5_$": np.quantile(fin, 0.05), "Median_real_$": np.median(real),
                         "P10_real_$": np.quantile(real, 0.10), "P5_real_$": np.quantile(real, 0.05),
                         "P(>=$500k)": (fin >= 5e5).mean(), "P(>=$1M)": (fin >= 1e6).mean(),
                         "P(real>=$500k)": (real >= 5e5).mean(), "P(real>=$1M)": (real >= 1e6).mean(),
                         "P(MDD>30%)": (mdd < -0.30).mean(), "P(MDD>40%)": (mdd < -0.40).mean(),
                         "E[MDD]": mdd.mean(), "Median_CAGR": (np.median(fin) / INIT) ** (1 / 30) - 1})
            dists[(meth, k)] = fin
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(TAB, "montecarlo.csv"), index=False)
    RES["mc"] = M.round(4).to_dict("records")
    return M, dists


# ----------------------------------------------------------------------------- 9. stress
STRESS = {
    "A Equity crash": {"VT": -0.40, "GOLD": 0.15, "VGIT": 0.10, "TBILL": 0.04},
    "B Stagflation shock": {"VT": -0.25, "GOLD": 0.20, "VGIT": -0.15, "TBILL": 0.04},
    "C Gold bear": {"VT": 0.15, "GOLD": -0.30, "VGIT": 0.03, "TBILL": 0.04},
    "D Joint stock/bond selloff": {"VT": -0.30, "GOLD": -0.10, "VGIT": -0.15, "TBILL": 0.04},
}
# recovery paths for the 5 years after the shock (annual returns, then flat monthly)
RECOVERY = {
    "V-shaped": {"VT": [0.35, 0.10, 0.07, 0.07, 0.07], "GOLD": [-0.05, 0.02, 0.02, 0.02, 0.02],
                 "VGIT": [0.02, 0.03, 0.04, 0.04, 0.04], "TBILL": [0.03, 0.03, 0.03, 0.03, 0.03]},
    "Slow": {"VT": [0.06, 0.06, 0.06, 0.06, 0.06], "GOLD": [0.02] * 5, "VGIT": [0.04] * 5, "TBILL": [0.03] * 5},
    "L-shaped then catch-up": {"VT": [0.0, 0.0, 0.0, 0.15, 0.15], "GOLD": [0.03] * 5, "VGIT": [0.04] * 5,
                               "TBILL": [0.03] * 5},
}


def sec_stress(ports):
    rows, trades, rec = [], [], []
    for sn, sh in STRESS.items():
        s = np.array([sh[a] for a in A])
        for k, w in ports.items():
            w = np.array(w, float)
            h0 = INIT * w
            h1 = h0 * (1 + s)
            V = h1.sum()
            rows.append({"Scenario": sn, "Portfolio": k, "Loss_%": V / INIT - 1, "Remaining_$": V})
            tgt = w * V
            trades.append({"Scenario": sn, "Portfolio": k,
                           **{f"trade {a} $": tgt[i] - h1[i] for i, a in enumerate(A)},
                           "Turnover_$": 0.5 * np.abs(tgt - h1).sum()})
            # recovery: shock happens over months 1-6, recovery path after; compare policies
            for rn, rp in RECOVERY.items():
                monthly = np.vstack([np.repeat((1 + np.array(rp[a])) ** (1 / 12) - 1, 12) for a in A]).T
                shock_m = np.tile((1 + s) ** (1 / 6) - 1, (6, 1))
                path = np.vstack([shock_m, monthly])  # 66 months; calendar: shock Jul-Dec
                res = {}
                for pol in ["none", "A", "A_band5", "abs5", "rel25"]:
                    E.simulate.months = (np.arange(len(path)) + 6) % 12 + 1  # starts in July
                    o = E.simulate(path, w[None, :], pol)
                    res[pol] = INIT * np.prod(1 + o["ret"][:, 0])
                rec.append({"Scenario": sn, "Recovery": rn, "Portfolio": k,
                            **{f"V_5.5y {p}": v for p, v in res.items()},
                            "Annual - None $": res["A"] - res["none"], "Thresh5 - Annual $": res["abs5"] - res["A"]})
    S1, S2, S3 = pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(rec)
    S1.to_csv(os.path.join(TAB, "stress_loss.csv"), index=False)
    S2.to_csv(os.path.join(TAB, "stress_trades.csv"), index=False)
    S3.to_csv(os.path.join(TAB, "stress_recovery.csv"), index=False)
    RES["stress"] = S1.round(4).to_dict("records")
    RES["stress_trades"] = S2.round(0).to_dict("records")
    RES["stress_recovery"] = S3.round(0).to_dict("records")
    return S1, S2, S3


# ----------------------------------------------------------------------------- 2026 OOS check
def sec_2026(ports):
    """2026 YTD (Jan..last complete month) was never used for any selection: pure OOS."""
    R26, _ = data.build_panel(start="2026-01-31", end="2026-10-31", verbose=False)
    o = E.run(R26, np.array(list(ports.values())), "none")
    cum = np.prod(1 + o["ret"], 0) - 1
    RES["oos_2026"] = {"months": f"{R26.index[0]:%Y-%m}..{R26.index[-1]:%Y-%m}",
                       "assets": ((1 + R26[A]).prod() - 1).round(4).to_dict(),
                       "ports": dict(zip(ports, np.round(cum, 4)))}


def main():
    R, S = sec_data()
    T = sec_windows(R)
    sec_corr(R)
    sec_rebal(R)
    G, evals, OT, rob = sec_optimize(R)
    sec_bootstrap_weights(R, G)
    sec_walkforward(R, G)
    json.dump(RES, open(os.path.join(OUT, "results_partial.json"), "w"), indent=1, default=str)
    return R, G


# ----------------------------------------------------------------------------- 10. gold & pairwise
def sec_gold(R):
    """Gold-weight sensitivity across windows and break-even gold return haircut."""
    rows = []
    golds = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for base, lab in [("VT", "VT+Gold"), ("VT+IT10", "VT+Gold+VGIT10")]:
        W = []
        for g in golds:
            W.append([1 - g, g, 0, 0] if base == "VT" else [0.9 - g, g, 0.1, 0])
        for wname, (a, b) in {**WINDOWS, **WINDOWS_X25}.items():
            Rw = slice_(R, a, b)
            o = E.run(Rw, np.array(W), "A")
            m = E.metrics(o["ret"], Rw.RF.values)
            for j, g in enumerate(golds):
                rows.append({"Family": lab, "Gold": g, "Window": wname, "CAGR": m.CAGR[j],
                             "Sharpe": m.Sharpe[j], "MaxDD": m.MaxDD[j], "CE3": ce_return(o["ret"][:, j], 3)})
    Gs = pd.DataFrame(rows)
    Gs.to_csv(os.path.join(TAB, "gold_sensitivity.csv"), index=False)
    # break-even haircut: annual gold log-return reduction making X% gold no better than 0%
    be = []
    full = slice_(R, "1972", "2025")
    for g in [0.05, 0.10, 0.15, 0.20]:
        for wname, (a, b) in [("1972-2025", ("1972", "2025")), ("2000-2025", ("2000", "2025")),
                              ("1980-2024", ("1980", "2024"))]:
            Rw = slice_(R, a, b).copy()
            res = {}
            for crit in ["CAGR", "Sharpe"]:
                lo, hi = -0.10, 0.15
                for _ in range(40):
                    mid = (lo + hi) / 2
                    X = Rw.copy()
                    X["GOLD"] = np.expm1(np.log1p(Rw.GOLD) - mid / 12)
                    o = E.run(X, np.array([[1, 0, 0, 0], [1 - g, g, 0, 0]]), "A")
                    m = E.metrics(o["ret"], X.RF.values)
                    better = m[crit][1] > m[crit][0]
                    lo, hi = (mid, hi) if better else (lo, mid)
                gold_cagr = (1 + Rw.GOLD).prod() ** (12 / len(Rw)) - 1
                res[crit] = (1 + gold_cagr) * np.exp(-mid) - 1
            be.append({"Gold weight": g, "Window": wname, "Hist gold CAGR": gold_cagr,
                       "Breakeven gold CAGR (CAGR)": res["CAGR"], "Breakeven gold CAGR (Sharpe)": res["Sharpe"]})
    BE = pd.DataFrame(be)
    BE.to_csv(os.path.join(TAB, "gold_breakeven.csv"), index=False)
    RES["gold_breakeven"] = BE.round(4).to_dict("records")
    return Gs, BE


def sec_pairwise(R, n_b=2000):
    """Stationary-bootstrap CIs for differences in Sharpe and CAGR between portfolio pairs."""
    pairs = [("A: VT90/G10", "VT100"), ("B: VT80/G10/VGIT10", "A: VT90/G10"),
             ("C: VT80/G10/SGOV10", "A: VT90/G10"), ("B: VT80/G10/VGIT10", "C: VT80/G10/SGOV10"),
             ("VT80/VGIT20", "VT100")]
    out = []
    for wname in ["1972-2025", "2000-2025", "1980-2024"]:
        Rw = slice_(R, *wname.split("-"))
        names = list(NAMED)
        o = E.run(Rw, np.array(list(NAMED.values())), "A")
        r = o["ret"]
        rf = Rw.RF.values
        rng = np.random.default_rng(SEED + 7)
        idx = E.stationary_bootstrap_idx(len(r), len(r), 24, rng, n_b)
        for p1, p2 in pairs:
            i, j = names.index(p1), names.index(p2)
            d_sh, d_cg = [], []
            for b in range(n_b):
                rb = r[idx[b]]
                ex = rb - rf[idx[b]][:, None]
                sh = ex.mean(0) / ex.std(0) * np.sqrt(12)
                cg = np.exp(np.log1p(rb).mean(0) * 12) - 1
                d_sh.append(sh[i] - sh[j])
                d_cg.append(cg[i] - cg[j])
            ex = r - rf[:, None]
            sh = ex.mean(0) / ex.std(0) * np.sqrt(12)
            cg = np.exp(np.log1p(r).mean(0) * 12) - 1
            d_sh, d_cg = np.array(d_sh), np.array(d_cg)
            out.append({"Window": wname, "Pair": f"{p1} minus {p2}",
                        "dSharpe": sh[i] - sh[j], "dSharpe_p05": np.quantile(d_sh, 0.05),
                        "dSharpe_p95": np.quantile(d_sh, 0.95), "P(dSharpe>0)": (d_sh > 0).mean(),
                        "dCAGR": cg[i] - cg[j], "dCAGR_p05": np.quantile(d_cg, 0.05),
                        "dCAGR_p95": np.quantile(d_cg, 0.95), "P(dCAGR>0)": (d_cg > 0).mean()})
    P = pd.DataFrame(out)
    P.to_csv(os.path.join(TAB, "pairwise_bootstrap.csv"), index=False)
    RES["pairwise"] = P.round(4).to_dict("records")
    return P


CANDIDATES = {
    "VT100": [1.0, 0, 0, 0],
    "A: VT90/G10": [0.9, 0.1, 0, 0],
    "B: VT80/G10/VGIT10": [0.8, 0.1, 0.1, 0],
    "C: VT80/G10/SGOV10": [0.8, 0.1, 0, 0.1],
    "VT80/VGIT20": [0.8, 0, 0.2, 0],
    "VT95/G5": [0.95, 0.05, 0, 0],
    "VT85/G15": [0.85, 0.15, 0, 0],
    "VT80/G20": [0.8, 0.2, 0, 0],
    "VT75/G15/VGIT10": [0.75, 0.15, 0.10, 0],
    "VT70/G15/VGIT15": [0.70, 0.15, 0.15, 0],
    "VT80/G10/VGIT5/SGOV5": [0.8, 0.1, 0.05, 0.05],
    "VT60/G30/VGIT10 (in-sample robust)": [0.6, 0.3, 0.1, 0],
}


def main2():
    R, _ = data.build_panel(verbose=False)
    global RES
    RES = json.load(open(os.path.join(OUT, "results_partial.json")))
    sec_gold(R)
    sec_pairwise(R)
    sec_regimes(R, extra={k: v for k, v in CANDIDATES.items() if k not in NAMED})
    sec_montecarlo(R, CANDIDATES)
    sec_stress(CANDIDATES)
    sec_2026(CANDIDATES)
    json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "part2":
        main2()
    else:
        main()
        main2()
