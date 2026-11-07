"""Vectorized portfolio backtest engine with rebalancing policies and risk metrics."""
import numpy as np
import pandas as pd

ASSETS = ["VT", "GOLD", "VGIT", "TBILL"]
TC_BPS = 5.0  # assumed one-way trading cost (bid/ask + slippage) in bps of traded notional


def simulate(R, W, policy="A", tc_bps=TC_BPS, return_paths=False):
    """Simulate P portfolios over T months.

    R : (T, N) array of simple monthly returns
    W : (P, N) target weights
    policy : 'none' | 'M' | 'Q' | 'S' | 'A' | 'A_band5' | 'abs5' | 'rel25'
    Rebalancing happens at month end *after* that month's returns are realized, using only
    information available at that time. Calendar rebalances align with Dec (A), Jun/Dec (S),
    quarter ends (Q).
    Returns dict with portfolio monthly returns (T, P), turnover per month (T, P), n_trades.
    """
    R = np.asarray(R, float)
    W = np.atleast_2d(np.asarray(W, float))
    T, N = R.shape
    P = W.shape[0]
    months = None
    if isinstance(R, pd.DataFrame):
        months = R.index.month
    h = W.copy()  # holdings in dollars, start with $1 total
    rp = np.empty((T, P))
    turn = np.zeros((T, P))
    trades = np.zeros(P, dtype=int)
    tc = tc_bps / 1e4
    for t in range(T):
        v0 = h.sum(1)
        h = h * (1 + R[t])
        v1 = h.sum(1)
        w = h / v1[:, None]
        do = _rebalance_flag(policy, t, w, W, simulate.months)
        if do.any():
            dw = np.abs(w - W).sum(1) * do  # two-way sum |dw|
            cost = tc * dw * v1
            v1 = v1 - cost
            h = np.where(do[:, None], W * v1[:, None], h)
            turn[t] = 0.5 * dw
            trades += (do & (dw > 1e-9)) * ((np.abs(w - W) > 1e-9).sum(1))
        rp[t] = v1 / v0 - 1
    out = {"ret": rp, "turnover": turn, "trades": trades}
    return out


simulate.months = None  # set externally to an int array of calendar months (1..12)


def _rebalance_flag(policy, t, w, W, months):
    P = w.shape[0]
    if policy == "none":
        return np.zeros(P, bool)
    if policy in ("M", "Q", "S", "A"):
        m = months[t]
        ok = {"M": True, "Q": m % 3 == 0, "S": m % 6 == 0, "A": m == 12}[policy]
        return np.full(P, ok)
    if policy == "A_band5":  # check once a year (Dec); trade only if a sleeve is >5pp off
        if months[t] != 12:
            return np.zeros(P, bool)
        return (np.abs(w - W) > 0.05).any(1)
    if policy == "abs5":
        return (np.abs(w - W) > 0.05).any(1)
    if policy == "rel25":
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(W > 0, np.abs(w / W - 1), 0.0)
        return (rel > 0.25).any(1)
    raise ValueError(policy)


def run(df, W, policy="A", tc_bps=TC_BPS):
    """Convenience wrapper on a DataFrame slice with ASSETS columns."""
    simulate.months = df.index.month.values
    o = simulate(df[ASSETS].values, W, policy, tc_bps)
    return o


def metrics(r, rf=None, cpi=None, init=80_000.0):
    """r : (T,) or (T,P) monthly portfolio returns. Returns DataFrame of metrics (P rows)."""
    r = np.asarray(r, float)
    if r.ndim == 1:
        r = r[:, None]
    T, P = r.shape
    yrs = T / 12
    wealth = np.cumprod(1 + r, 0)
    cagr = wealth[-1] ** (1 / yrs) - 1
    vol = r.std(0, ddof=1) * np.sqrt(12)
    rfv = np.zeros(T) if rf is None else np.asarray(rf, float)
    ex = r - rfv[:, None]
    sharpe = ex.mean(0) * 12 / (ex.std(0, ddof=1) * np.sqrt(12))
    dd_dev = np.sqrt((np.minimum(ex, 0) ** 2).mean(0)) * np.sqrt(12)
    sortino = ex.mean(0) * 12 / dd_dev
    peak = np.maximum.accumulate(np.vstack([np.ones(P), wealth]), 0)[1:]
    dd = wealth / peak - 1
    mdd = dd.min(0)
    calmar = cagr / np.abs(mdd)
    k = max(1, int(np.ceil(0.05 * T)))
    cvar = np.sort(r, 0)[:k].mean(0)  # monthly 95% CVaR (expected shortfall)
    lw = np.log1p(r)
    cs = np.vstack([np.zeros(P), np.cumsum(lw, 0)])
    r12 = np.expm1(cs[12:] - cs[:-12]).min(0) if T >= 12 else np.full(P, np.nan)
    r36 = (np.exp((cs[36:] - cs[:-36]) / 3) - 1).min(0) if T >= 36 else np.full(P, np.nan)
    rec_mdd, longest_uw = _recovery(wealth)
    out = pd.DataFrame({
        "CAGR": cagr, "Vol": vol, "MaxDD": mdd, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar, "CVaR95_m": cvar, "Worst12m": r12, "Worst36m_ann": r36,
        "MaxDD_recovery_m": rec_mdd, "Longest_underwater_m": longest_uw,
        "Final_$": init * wealth[-1],
    })
    if cpi is not None:
        infl = np.prod(1 + np.asarray(cpi, float))
        out["Real_final_$"] = out["Final_$"] / infl
        out["Real_CAGR"] = (1 + cagr) / infl ** (1 / yrs) - 1
    return out


def _recovery(wealth):
    """Months from MaxDD trough back to prior peak (nan if not recovered) and longest
    peak-to-recovery underwater spell (open spells counted to end)."""
    T, P = wealth.shape
    rec = np.full(P, np.nan)
    longest = np.zeros(P)
    for p in range(P):
        w = np.concatenate([[1.0], wealth[:, p]])
        pk = np.maximum.accumulate(w)
        dd = w / pk - 1
        tr = int(np.argmin(dd))
        after = np.where(w[tr:] >= pk[tr] - 1e-12)[0]
        rec[p] = after[0] if len(after) else np.nan
        uw = dd < -1e-9
        run_, best = 0, 0
        for u in uw:
            run_ = run_ + 1 if u else 0
            best = max(best, run_)
        longest[p] = best
    return rec, longest


def drawdown_series(r):
    w = np.cumprod(1 + np.asarray(r))
    return w / np.maximum.accumulate(np.maximum(w, 1.0)) - 1


def weight_grid(step=0.05, bounds=None, include_both_bonds=True):
    """All weight vectors on a simplex grid with box bounds (VT, GOLD, VGIT, TBILL)."""
    if bounds is None:
        bounds = {"VT": (0.60, 1.00), "GOLD": (0.0, 0.30), "VGIT": (0.0, 0.30), "TBILL": (0.0, 0.30)}
    n = int(round(1 / step))
    rows = []
    for g in range(0, n + 1):
        for b in range(0, n + 1 - g):
            for c in range(0, n + 1 - g - b):
                e = n - g - b - c
                w = np.array([e, g, b, c]) / n
                if all(bounds[a][0] - 1e-9 <= w[i] <= bounds[a][1] + 1e-9 for i, a in enumerate(ASSETS)):
                    if not include_both_bonds and w[2] > 0 and w[3] > 0:
                        continue
                    rows.append(w)
    return np.array(rows)


def stationary_bootstrap_idx(T, n_out, mean_block, rng, n_paths):
    """Politis-Romano stationary bootstrap indices, shape (n_paths, n_out)."""
    p = 1.0 / mean_block
    idx = np.empty((n_paths, n_out), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_paths)
    new_block = rng.random((n_paths, n_out)) < p
    starts = rng.integers(0, T, (n_paths, n_out))
    for t in range(1, n_out):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def wlabel(w):
    return "/".join(f"{a}{int(round(x*100))}" for a, x in zip(["VT", "G", "IT", "TB"], w) if x > 1e-9)
