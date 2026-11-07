"""Build a monthly USD total-return panel for VT / Gold / VGIT / T-bills, 1972-01 onward.

Splicing rule (no look-ahead): for each asset, use a historical proxy until the first full
month in which the investable ETF has an adjusted-close history, then switch to the ETF.
Proxy returns are net of the target ETF's current expense ratio; ETF adjusted closes are
already net of fees (NAV-based fee drag is embedded in price).

Sources (all downloaded raw into ../data):
  - Ken French Data Library: F-F_Research_Data_Factors (US Mkt-RF, RF), Developed_3_Factors
  - Yahoo Finance chart API: daily adjclose (dividends reinvested) for VT, GLD, GLDM, VGIT,
    SGOV, BIL, IEI, VTI and yields ^FVX (5y), ^IRX (13w)
  - datahub.io / GitHub datasets/gold-prices monthly.csv (monthly gold USD/oz)
  - datahub.io / GitHub datasets/cpi-us cpiai.csv (BLS CPI-U NSA, monthly)
  - Damodaran histretSP.xls (annual gold, inflation; used for cross-validation)
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data")

# Current expense ratios (annual) used to haircut proxy returns
ER = {"VT": 0.0006, "GOLD": 0.0010, "VGIT": 0.0003, "TBILL": 0.0009}


def _yahoo_monthly(ticker, field="adjclose"):
    with open(os.path.join(RAW, f"yh_{ticker}.json")) as f:
        r = json.load(f)["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    if field == "adjclose":
        px = r["indicators"]["adjclose"][0]["adjclose"]
    else:
        px = r["indicators"]["quote"][0]["close"]
    s = pd.Series(px, index=ts, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    m = s.resample("ME").last()
    # drop first (partial) month so the first monthly return is a full month
    first_full = s.index[0] + pd.offsets.MonthEnd(0)
    if s.index[0].day > 5:
        m = m[m.index > first_full]
    return m


def _splice(base, new, idx):
    """Replace/extend `base` with `new` on index `idx` (union of indexes)."""
    new = new.reindex(idx) if hasattr(new, "reindex") else pd.Series(new, index=idx)
    out = base.reindex(base.index.union(idx))
    out.loc[idx] = new
    return out


def _french(path, start_marker=None):
    rows = []
    with open(path) as f:
        lines = f.readlines()
    started = False
    for ln in lines:
        parts = [p.strip() for p in ln.strip().split(",")]
        if len(parts) >= 2 and parts[0] == "" and parts[1].startswith("Mkt-RF"):
            if started:
                break  # second header -> annual block
            started = True
            cols = parts[1:]
            continue
        if started:
            if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 6:
                if rows:
                    break
                continue
            rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["ym"] + cols)
    df.index = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.drop(columns="ym") / 100.0


def load_cpi():
    c = pd.read_csv(os.path.join(RAW, "cpi_us.csv"), parse_dates=["Date"])
    s = c.set_index("Date")["Index"]
    s.index = s.index + pd.offsets.MonthEnd(0)
    # BLS did not publish October 2025 CPI (federal shutdown); fill gaps log-linearly
    full = pd.date_range(s.index[0], s.index[-1], freq="ME")
    s = np.exp(np.log(s.reindex(full)).interpolate())
    return s


def build_panel(start="1972-01-31", end="2025-12-31", verbose=True, gold_unsmooth=True):
    us = _french(os.path.join(RAW, "F-F_Research_Data_Factors.csv"))
    dev = _french(os.path.join(RAW, "Developed_3_Factors.csv"))
    rf = us["RF"]

    # ---- Equity: US market (1972-1990/06) -> French Developed (1990/07-2008/06) -> VT ETF
    eq_us = us["Mkt-RF"] + us["RF"]
    eq_dev = dev["Mkt-RF"] + dev["RF"]  # Developed factors RF is the same US 1m T-bill
    vt = _yahoo_monthly("VT").pct_change().dropna()
    eq = eq_us.copy()
    eq = _splice(eq, eq_dev, eq_dev.index)
    eq = eq - ER["VT"] / 12
    eq_src = pd.Series("US mkt (French)", index=eq.index)
    eq_src = _splice(eq_src, "Developed mkt (French)", eq_dev.index)
    eq = _splice(eq, vt, vt.index)
    eq_src = _splice(eq_src, "VT ETF", vt.index)

    # ---- Gold: monthly gold price series -> GLD ETF (2004/12) -> GLDM ETF (2018/07)
    g = pd.read_csv(os.path.join(RAW, "gold_monthly.csv"))
    g.index = pd.to_datetime(g["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)
    gold_px = g["Price"].astype(float)
    gold = gold_px.pct_change().dropna()
    if gold_unsmooth:
        # The pre-ETF series is a monthly *average* price. Averaging a random walk induces
        # first-order autocorrelation of ~0.25 and cuts variance to ~2/3 (Working, 1960).
        # Invert with a Geltner AR(1) filter at rho=0.25 (approximate; sensitivity tested).
        rho = 0.25
        # Done in log space so that cumulative (geometric) return is preserved.
        lg = np.log1p(gold)
        gold = np.expm1((lg - rho * lg.shift(1)) / (1 - rho)).dropna()
    gold = gold - ER["GOLD"] / 12
    gld = _yahoo_monthly("GLD").pct_change().dropna()
    gldm = _yahoo_monthly("GLDM").pct_change().dropna()
    gold_src = pd.Series("Gold spot (monthly)", index=gold.index)
    # GLD charges 0.40%; add back the ER differential so the series reflects GLDM's 0.10%
    gld_adj = gld + (0.0040 - ER["GOLD"]) / 12
    gold = _splice(gold, gld_adj, gld_adj.index)
    gold_src = _splice(gold_src, "GLD ETF (+0.30% ER adj)", gld_adj.index)
    gold = _splice(gold, gldm, gldm.index)
    gold_src = _splice(gold_src, "GLDM ETF", gldm.index)

    # ---- Intermediate Treasuries: 5y CMT par-bond model -> IEI (2007/02) -> VGIT (2009/12)
    y5 = _yahoo_monthly("FVX", field="close") / 100.0
    it = par_bond_return(y5, maturity=5.0) - ER["VGIT"] / 12
    iei = _yahoo_monthly("IEI").pct_change().dropna()
    vgit = _yahoo_monthly("VGIT").pct_change().dropna()
    it_src = pd.Series("5y CMT par-bond model", index=it.index)
    it = _splice(it, iei + (0.0015 - ER["VGIT"]) / 12, iei.index)
    it_src = _splice(it_src, "IEI ETF (+0.12% ER adj)", iei.index)
    it = _splice(it, vgit, vgit.index)
    it_src = _splice(it_src, "VGIT ETF", vgit.index)

    # ---- T-bills: French RF (1m T-bill) -> BIL (2007/06) -> SGOV (2020/06)
    tb = rf - ER["TBILL"] / 12
    bil = _yahoo_monthly("BIL").pct_change().dropna()
    sgov = _yahoo_monthly("SGOV").pct_change().dropna()
    tb_src = pd.Series("1m T-bill (French RF)", index=tb.index)
    tb = _splice(tb, bil + (0.00136 - ER["TBILL"]) / 12, bil.index)
    tb_src = _splice(tb_src, "BIL ETF (+0.05% ER adj)", bil.index)
    tb = _splice(tb, sgov, sgov.index)
    tb_src = _splice(tb_src, "SGOV ETF", sgov.index)

    cpi = load_cpi()
    infl = cpi.pct_change()

    R = pd.concat({"VT": eq, "GOLD": gold, "VGIT": it, "TBILL": tb}, axis=1, sort=True)
    S = pd.concat({"VT": eq_src, "GOLD": gold_src, "VGIT": it_src, "TBILL": tb_src}, axis=1, sort=True)
    R["CPI"] = infl
    R["RF"] = rf  # gross 1m T-bill, used as Sharpe risk-free rate
    R = R.loc[start:end].dropna()
    S = S.loc[R.index]
    if verbose:
        print("Panel:", R.index[0].date(), "->", R.index[-1].date(), len(R), "months")
        for c in S.columns:
            seg = S[c].groupby((S[c] != S[c].shift()).cumsum()).agg(["first", "size"])
            idx = S[c].groupby((S[c] != S[c].shift()).cumsum()).apply(lambda x: f"{x.index[0]:%Y-%m}..{x.index[-1]:%Y-%m}")
            print(f"  {c}: " + "; ".join(f"{a} [{b}]" for a, b in zip(seg['first'], idx)))
    return R, S


def par_bond_return(y, maturity=5.0):
    """Monthly total return of a constant-maturity par bond priced off month-end yields.

    Bond bought at par with coupon y_{t-1} (semiannual), repriced one month later at y_t with
    maturity (maturity - 1/12). Same construction Damodaran uses (annual, 10y)."""
    y_prev = y.shift(1)
    n = maturity - 1 / 12
    c = y_prev / 2
    r = y / 2
    periods = 2 * n
    pv_coupons = c * (1 - (1 + r) ** (-periods)) / r
    pv_face = (1 + r) ** (-periods)
    price = pv_coupons + pv_face
    ret = price - 1 + y_prev / 12
    return ret.dropna()


def validate(R_full=None):
    """Cross-check proxies vs. investable ETFs and Damodaran annual data on overlaps."""
    out = []
    us = _french(os.path.join(RAW, "F-F_Research_Data_Factors.csv"))
    dev = _french(os.path.join(RAW, "Developed_3_Factors.csv"))
    vt = _yahoo_monthly("VT").pct_change().dropna()
    eq_dev = dev["Mkt-RF"] + dev["RF"]
    j = pd.concat([vt, eq_dev], axis=1, sort=True, keys=["VT", "Dev"]).dropna()
    out.append(("VT vs French Developed", j.index[0], j.index[-1], j.corr().iloc[0, 1],
                (j["VT"] - j["Dev"]).std() * np.sqrt(12),
                ((1 + j).prod() ** (12 / len(j)) - 1).diff().iloc[-1]))
    g = pd.read_csv(os.path.join(RAW, "gold_monthly.csv"))
    g.index = pd.to_datetime(g["Date"], format="%Y-%m") + pd.offsets.MonthEnd(0)
    gs = g["Price"].astype(float).pct_change()
    gld = _yahoo_monthly("GLD").pct_change().dropna()
    j = pd.concat([gld, gs], axis=1, sort=True, keys=["GLD", "Spot"]).dropna()
    out.append(("GLD vs monthly gold series", j.index[0], j.index[-1], j.corr().iloc[0, 1],
                (j["GLD"] - j["Spot"]).std() * np.sqrt(12),
                ((1 + j).prod() ** (12 / len(j)) - 1).diff().iloc[-1]))
    vol_ratio = j["Spot"].std() / j["GLD"].std()
    y5 = _yahoo_monthly("FVX", field="close") / 100.0
    model = par_bond_return(y5)
    vgit = _yahoo_monthly("VGIT").pct_change().dropna()
    j = pd.concat([vgit, model], axis=1, sort=True, keys=["VGIT", "Model"]).dropna()
    out.append(("VGIT vs 5y par-bond model", j.index[0], j.index[-1], j.corr().iloc[0, 1],
                (j["VGIT"] - j["Model"]).std() * np.sqrt(12),
                ((1 + j).prod() ** (12 / len(j)) - 1).diff().iloc[-1]))
    sgov = _yahoo_monthly("SGOV").pct_change().dropna()
    j = pd.concat([sgov, us["RF"]], axis=1, sort=True, keys=["SGOV", "RF"]).dropna()
    out.append(("SGOV vs French RF", j.index[0], j.index[-1], j.corr().iloc[0, 1],
                (j["SGOV"] - j["RF"]).std() * np.sqrt(12),
                ((1 + j).prod() ** (12 / len(j)) - 1).diff().iloc[-1]))
    v = pd.DataFrame(out, columns=["check", "start", "end", "corr", "tracking_err_ann", "CAGR_diff(2nd-1st)"])
    # Damodaran annual gold & CPI vs our series
    x = pd.ExcelFile(os.path.join(RAW, "histretSP.xls"))
    d = pd.read_excel(x, "Returns by year", header=None)
    hdr_row = d.index[d.iloc[:, 0].astype(str).str.strip() == "Year"][0]
    d.columns = [str(c).strip() for c in d.iloc[hdr_row]]
    d = d.iloc[hdr_row + 1:]
    d = d[pd.to_numeric(d["Year"], errors="coerce").notna()].copy()
    d["Year"] = d["Year"].astype(int)
    d = d.set_index("Year")
    gcol = [c for c in d.columns if "Gold" in c][0]
    gs_ann = g["Price"].astype(float).resample("YE").last().pct_change()
    gs_ann.index = gs_ann.index.year
    cpi = load_cpi()
    cpi_dec = cpi[cpi.index.month == 12].pct_change()
    cpi_dec.index = cpi_dec.index.year
    yrs = range(1972, 2025)
    dd = pd.DataFrame({"dam_gold": pd.to_numeric(d[gcol], errors="coerce").reindex(yrs),
                       "our_gold": gs_ann.reindex(yrs),
                       "our_cpi_dec": cpi_dec.reindex(yrs)})
    return v, vol_ratio, dd


if __name__ == "__main__":
    R, S = build_panel()
    v, vr, dd = validate()
    pd.set_option("display.width", 200)
    print(v)
    print("Gold monthly-series vol / GLD vol:", round(vr, 3))
    print(dd.corr())
    print(dd.tail(8))
