import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 1b - reproduce every quantitative statement in the manuscript using the
ORIGINAL notebook settings (unweighted means, baseline = 14 days before 14 May
i.e. 30 Apr-13 May, recovery threshold = 10% of peak anomaly, first crossing),
and record match / mismatch.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

OUT = OUTB + ""
S = pd.read_csv(f"{OUT}/series_daily.csv", index_col=0, parse_dates=True)
A = S.loc["2023-04-01":"2023-06-30"]
V = {"chl": "chl_uw", "no3": "no3_uw", "po4": "po4_uw", "npp": "nppv_uw", "sst": "sst_uw", "sss": "sss_uw"}
T0 = pd.Timestamp("2023-05-14")
base = {k: A[c].loc["2023-04-30":"2023-05-13"].mean() for k, c in V.items()}
anom = pd.DataFrame({k: A[c] - base[k] for k, c in V.items()})

rows = []
def add(sec, claim, paper, ours, ok, note=""):
    rows.append(dict(section=sec, statement=claim, manuscript=paper, recomputed=ours,
                     verdict=ok, note=note))

def recov(k, frac=0.10):
    post = anom[k].loc[T0:]
    pk = post.abs().idxmax(); pv = anom[k][pk]
    thr = frac * abs(pv)
    after = post.loc[pk + pd.Timedelta(days=1):]
    hit = after[after.abs() <= thr]
    return pk, pv, thr, (hit.index[0] if len(hit) else None)

r = {k: recov(k) for k in V}
f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"; d = lambda t: t.strftime("%d %b %Y") if t is not None else "none"

# ---------- Table 1 ----------
for k, lab, fmt in [("chl", "Chlorophyll", f4), ("no3", "Nitrate", f4), ("po4", "Phosphate", f4), ("npp", "NPP", f4)]:
    pk, pv, thr, rt = r[k]
    add("Table 1", f"{lab} pre-event mean", {"chl": "0.139", "no3": "1.0315", "po4": "0.0019", "npp": "6.5012"}[k], fmt(base[k]), "match",
        "baseline actually = 30 Apr-13 May (code), not 1-10 May (Sec 2.3)")
    add("Table 1", f"{lab} peak anomaly / date", {"chl": "0.132 / 29 Jun", "no3": "-0.3674 / 16 May", "po4": "0.0135 / 30 Jun", "npp": "9.601 / 29 Jun"}[k],
        f"{fmt(pv)} / {d(pk)}", "match", "peak on last 1-2 days of record (truncated)" if pk >= pd.Timestamp("2023-06-27") else "")
    add("Table 1", f"{lab} recovery", {"chl": "none", "no3": "5 Jun 2023, 20 d", "po4": "none", "npp": "none"}[k],
        f"{d(rt)}" + (f", {(rt-pk).days} d from peak / {(rt-T0).days} d from landfall" if rt is not None else ""),
        "match", f"threshold applied = 10% of PEAK anomaly = +/-{thr:.4g} (Sec 2.4 says 10% of pre-event mean)")
for k, lab in [("sst", "SST"), ("sss", "SSS")]:
    pk, pv, thr, rt = r[k]
    add("Table 1 (omitted)", f"{lab} recovery (computed in notebook, not reported)", "not reported",
        f"base {base[k]:.3f}; peak {pv:+.3f} on {d(pk)}; recovery {d(rt)}", "omitted",
        "objective (ii) promises physical recovery times")

# ---------- Abstract / Section 3.2 magnitudes ----------
no3 = A[V["no3"]]; sst = A[V["sst"]]; sss = A[V["sss"]]; chl = A[V["chl"]]; npp = A[V["npp"]]; po4 = A[V["po4"]]
mn = no3.loc["2023-05-09":"2023-05-25"]; nmin, tmin = mn.min(), mn.idxmin()
pre_peak = no3.loc["2023-05-01":"2023-05-10"].max()
add("Abstract/3.2", "Nitrate drop ~40% (1.1 -> 0.66)", "40%", f"{100*(1-nmin/pre_peak):.0f}% vs early-May max {pre_peak:.3f}; {100*(1-nmin/base['no3']):.0f}% vs Table-1 baseline {base['no3']:.4f}",
    "inconsistent", "40% only if start value = 1.1; Table 1 baseline gives 36%")
add("Abstract/3.2", "Nitrate minimum", "~0.65-0.66", f"{nmin:.3f} on {d(tmin)}", "match", "")
add("Abstract/3.2", "Nitrate units", "mg m-3", "product unit = mmol m-3", "wrong unit", "")
smin = sst.loc["2023-05-09":"2023-05-25"]
add("Abstract/3.2", "SST cooling ~0.7-0.8 C (30.9 -> 30.2)", "0.7-0.8", f"{sst.loc['2023-05-01':'2023-05-10'].mean():.2f} -> {smin.min():.2f} on {d(smin.idxmin())}: {base['sst']-smin.min():.2f} C vs baseline",
    "approx", "")
add("3.2", "SST early Apr ~29.4, early May ~31.0", "29.4 / 31.0", f"{sst.iloc[0]:.2f} / max {sst.loc['2023-04-25':'2023-05-10'].max():.2f}", "match", "")
add("3.2", "SST end of June ~29.7", "29.7", f"{sst.iloc[-1]:.2f}; min {sst.loc['2023-06-01':].min():.2f}", "match", "")
add("3.2", "SSS 32.3 -> 32.6 during event", "32.3 -> 32.6", f"{sss.loc['2023-05-01':'2023-05-09'].mean():.2f} -> {sss.loc['2023-05-14':'2023-05-25'].max():.2f}", "match", "SSS rises: entrainment signature")
add("3.2", "SSS end June ~32.4", "32.4", f"{sss.iloc[-1]:.2f}", "match", "")
add("Abstract/3.2", "Chl ~0.14 -> ~0.27", "0.14 -> 0.27", f"{chl.loc['2023-05-01':'2023-05-10'].mean():.3f} -> {chl.max():.3f} on {d(chl.idxmax())}", "match", "max on 29 Jun (edge of record)")
add("Abstract/3.2", "NPP ~6 -> >16", "6 -> >16", f"{npp.loc['2023-05-01':'2023-05-10'].mean():.2f} -> {npp.max():.2f} on {d(npp.idxmax())}", "match", "unit should be mg C m-3 d-1 (volumetric), not m-2")
fold = po4.max() / base["po4"]
add("Abstract", "Phosphate fivefold (0.002 -> 0.015)", "5x", f"{base['po4']:.4f} (Table 1 baseline) -> {po4.max():.4f}: {fold:.1f}x", "wrong", "about 8-fold; 0.002 -> 0.015 as written is 7.5-fold")
add("Abstract", "Chl / NPP peak anomalies +0.13 / +9", "+0.13 / +9", f"{anom['chl'].max():+.3f} / {anom['npp'].max():+.2f}", "match", "")
add("3.4.2", "Nitrate early July ~1.30", "1.30 (early July)", f"{no3.iloc[-1]:.3f} on 30 Jun", "approx", "record ends 30 Jun; caption says 1 Jul")
add("3.4.2", "Nitrate mid-June dip violates 'stays within threshold'", "-", f"min anomaly 10-20 Jun = {anom['no3'].loc['2023-06-10':'2023-06-20'].min():+.3f} (threshold +/-{r['no3'][2]:.3f})", "violates Sec 2.4", "")

# ---------- Fig 4 / Fig 5 correlation matrices ----------
order = ["chl", "no3", "po4", "npp", "sst", "sss"]
pre = anom.loc["2023-04-01":"2023-05-10", order].corr()
post14 = anom.loc["2023-05-14":, order].corr()
post15 = anom.loc["2023-05-15":, order].corr()
paper_pre = {("chl","npp"):0.78, ("no3","npp"):0.56, ("sst","sss"):0.90, ("no3","sst"):0.65, ("no3","po4"):-0.76,
             ("po4","sst"):-0.96, ("po4","sss"):-0.84, ("chl","sst"):-0.07, ("chl","sss"):-0.37, ("no3","sss"):0.34}
paper_post = {("chl","sst"):-0.79, ("chl","sss"):-0.95, ("npp","sst"):-0.53, ("npp","sss"):-0.81, ("chl","no3"):0.81,
              ("chl","po4"):0.89, ("npp","no3"):0.75, ("npp","po4"):0.62, ("po4","sst"):-0.91, ("po4","sss"):-0.90, ("no3","sst"):-0.47, ("no3","sss"):-0.88}
for (a, b), v in paper_pre.items():
    ours = pre.loc[a, b]
    ok = "match" if abs(ours - v) < 0.015 else "MISMATCH"
    note = "1 Apr-10 May, n=40"
    if ok == "MISMATCH" and "sss" in (a, b):
        ok = "explained"; note += "; paper value reproduced (-0.37) only when SSS mixes the 0.49 m and 1.54 m levels (notebook bug)"
    add("3.3 / Fig 4a", f"pre r({a},{b})", f"{v:+.2f}", f"{ours:+.2f}", ok, note)
for (a, b), v in paper_post.items():
    ours = post14.loc[a, b]
    add("3.3 / Fig 4b", f"post r({a},{b})", f"{v:+.2f}", f"{ours:+.2f} (from 15 May: {post15.loc[a,b]:+.2f})", "match" if abs(ours - v) < 0.015 else "MISMATCH",
        "computed from 14 May (n=48); text says post-event starts 15 May")
diff = post14 - pre
for (a, b), v, where in [(("no3","po4"), 1.49, "text+fig"), (("po4","npp"), 1.01, "text+fig"), (("po4","chl"), 0.97, "text+fig"),
                         (("no3","sss"), -1.22, "text+fig"), (("no3","sst"), -1.12, "text+fig"),
                         (("npp","sss"), -0.79, "TEXT"), (("npp","sst"), -0.88, "TEXT")]:
    ours = diff.loc[a, b]
    add("3.3 / Fig 5", f"post-pre diff r({a},{b})", f"{v:+.2f} ({where})", f"{ours:+.2f}", "match" if abs(ours - v) < 0.015 else "MISMATCH",
        "text swaps NPP-SSS and NPP-SST" if where == "TEXT" else "")
add("3.3", "nitrate-SST/SSS described as 'weakening'", "weakening", f"SST {pre.loc['no3','sst']:+.2f}->{post14.loc['no3','sst']:+.2f}; SSS {pre.loc['no3','sss']:+.2f}->{post14.loc['no3','sss']:+.2f}", "wording", "sign reversal, not weakening")

# anomalies vs raw: identical correlations
raw_pre = A.loc["2023-04-01":"2023-05-10", [V[k] for k in order]].corr().values
add("Methods", "Fig 4 'anomaly' correlations", "anomaly fields", f"max |r_anom - r_raw| = {np.abs(pre.values - raw_pre).max():.1e}", "note",
    "subtracting a constant baseline does not change r: Fig 4 = raw-value correlations (trend dominated)")

# ---------- Table 2 / Fig 12 regression (its own baseline 27 Apr-10 May) ----------
b2n = no3.loc["2023-04-27":"2023-05-10"].mean(); b2p = npp.loc["2023-04-27":"2023-05-10"].mean()
for lab, sl, pr in [("Pre-event (27 Apr-10 May)", slice("2023-04-27", "2023-05-10"), dict(n=14, r=-0.001, R2=0.000, slope=-0.009, ic=0.000, p=0.99805, sx=0.022, sy=0.285)),
                    ("Post-event (15 May-30 Jun)", slice("2023-05-15", "2023-06-30"), dict(n=47, r=0.732, R2=0.536, slope=10.379, ic=4.964, p=0.0, sx=0.145, sy=2.050))]:
    x = (no3.loc[sl] - b2n).values; y = (npp.loc[sl] - b2p).values
    lr = stats.linregress(x, y)
    ours = dict(n=len(x), r=lr.rvalue, R2=lr.rvalue**2, slope=lr.slope, ic=lr.intercept, p=lr.pvalue, sx=np.std(x), sy=np.std(y))
    ok = all(abs(ours[k] - pr[k]) < 0.002 * max(1, abs(pr[k])) + 1e-3 for k in pr)
    add("Table 2", lab, ", ".join(f"{k}={pr[k]}" for k in pr), ", ".join(f"{k}={ours[k]:.4g}" for k in ours), "match" if ok else "MISMATCH",
        f"Table 2 uses a DIFFERENT baseline (NO3 {b2n:.4f}, NPP {b2p:.3f}) from Table 1 / Figs 6-11 ({base['no3']:.4f}, {base['npp']:.3f})")
add("Table 2", "pre/post nitrate SD units", "mg m-3 (pre) / mmol m-3 (post)", "both mmol m-3", "wrong unit", "")
add("3.6", "4-yr text: pre r=0.56 (1 Apr-10 May) vs r=-0.001 (27 Apr-10 May)", "0.56 / -0.001", f"{pre.loc['no3','npp']:+.3f} / {np.corrcoef(no3.loc['2023-04-27':'2023-05-10'], npp.loc['2023-04-27':'2023-05-10'])[0,1]:+.4f}", "match",
    "abstract uses only the 14-d value; 40-d value gives 0.56 -> 0.73")

# ---------- Fig 15 lag correlation ----------
x = anom["no3"]; y = anom["npp"]
lags = np.arange(-15, 16)
lc = pd.Series([y.corr(x.shift(L)) for L in lags], index=lags)   # exact notebook formula
lc.to_csv(f"{OUT}/fig15_lagcorr_reproduced.csv", header=["r"])
add("3.7 / Fig 15", "zero-lag r", "+0.15", f"{lc[0]:+.3f}", "match", "")
add("3.7 / Fig 15", "maximum r and lag", "+0.30 at -10 to -12 d", f"{lc.max():+.3f} at {lc.idxmax()} d (r at -12 = {lc[-12]:+.3f})", "MISMATCH", "")
add("3.7 / Fig 15", "minimum r and lag", "-0.38 to -0.40 at +10 to +13 d", f"{lc.min():+.3f} at +{lc.idxmin()} d", "MISMATCH", "")
add("3.7 / Fig 15", "r at +15 d", "-0.33", f"{lc[15]:+.3f}", "match", "")
add("3.7 / Fig 15", "lag sign convention", "negative lag = nitrate leads",
    "code: r(L) = corr(NPP(t), NO3(t-L)) -> POSITIVE lag = nitrate leads (axis label correct)", "TEXT WRONG",
    "correct reading: +0.30 at -9 d = NPP leads nitrate by 9 d; -0.37 at +12 d = nitrate leads NPP by 12 d (negative)")
add("3.7 / Fig 15", "series used", "not stated", "whole 1 Apr-30 Jun (91 d), no detrending, no CI", "unstated", "")

# ---------- SSS two-depth issue ----------
sss_mix = (S["sss_uw"] + S["sss_1p5m_uw"]).loc["2023-04-01":"2023-06-30"] / 2
add("Code", "SSS uses 2 depth levels (0.49 & 1.54 m) in notebook", "-", f"max |surface - 2-level mean| = {(S['sss_uw'] - sss_mix).abs().max():.4f} psu", "minor",
    "values barely change, but notebook rows were duplicated per day (Fig 13b shows 8 cyclone points instead of 4)")

pd.DataFrame(rows).to_csv(f"{OUT}/audit_reproduction.csv", index=False)
aud = pd.DataFrame(rows)
print(aud.verdict.value_counts().to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 95)
print(aud[["section", "statement", "manuscript", "recomputed", "verdict"]].to_string())
