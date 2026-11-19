import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 4a/4b - autocorrelation-aware statistics.

Series: daily 2023 anomalies vs the 1993-2022 daily climatology (step 2), wake-core region mean
(and, for the response letter, the original basin-mean raw series as published).
Windows: pre = 1 Apr - 8 May (38 d), post = 15 May - 30 Jun (47 d); event days 9-14 May excluded.
Each series is linearly detrended within the window before correlating.

Effective sample size for a correlation (Bretherton et al. 1999):
    n_eff = n (1 - r1x r1y) / (1 + r1x r1y),  r1 = lag-1 autocorrelation of the detrended series
p-value: t = r sqrt((n_eff - 2) / (1 - r^2)), Student t with n_eff - 2 dof (two-sided)
95% CI: Fisher z +- 1.96 / sqrt(n_eff - 3)
Pre vs post difference: z = (z_post - z_pre) / sqrt(1/(n_eff_post - 3) + 1/(n_eff_pre - 3))
Lag correlation: r(L) = corr(NPP(t), NO3(t - L)); L > 0 means nitrate LEADS NPP by L days.
Significance from 2000 phase-randomised surrogates of the nitrate series (Ebisuzaki 1997):
pointwise 95% bounds per lag and a global bound for the maximum |r| over all lags.
"""
import numpy as np, pandas as pd, warnings
from scipy import stats, signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
O2 = OUTB + "/step2"; O = OUTB + "/step4"
import os; os.makedirs(O, exist_ok=True)
dd = pd.read_pickle(f"{O2}/daily_region_series.pkl")
VARS = ["sst", "sss", "no3", "po4", "chl", "npp"]
LAB = {"sst": "SST", "sss": "SSS", "no3": "NO3", "po4": "PO4", "chl": "Chl", "npp": "NPP"}
WIN = {"pre": ("2023-04-01", "2023-05-08"), "post": ("2023-05-15", "2023-06-30")}
rng = np.random.default_rng(42)


def detr(x):
    x = np.asarray(x, float); ok = np.isfinite(x); out = np.full_like(x, np.nan)
    t = np.arange(len(x)); p = np.polyfit(t[ok], x[ok], 1); out[ok] = x[ok] - np.polyval(p, t[ok]); return out


def r1(x):
    x = x[np.isfinite(x)]; x = x - x.mean(); return np.sum(x[1:] * x[:-1]) / np.sum(x * x)


def corr_stats(x, y):
    x = detr(x); y = detr(y); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
    r = np.corrcoef(x, y)[0, 1]; a, b = r1(x), r1(y)
    ne = n * (1 - a * b) / (1 + a * b); ne = float(np.clip(ne, 3.01, n))
    t = r * np.sqrt(max(ne - 2, 1e-6) / max(1 - r * r, 1e-12)); p = 2 * stats.t.sf(abs(t), max(ne - 2, 1e-6))
    z = np.arctanh(np.clip(r, -0.9999, 0.9999)); se = 1 / np.sqrt(max(ne - 3, 1e-6))
    return dict(r=r, n=n, r1_x=a, r1_y=b, n_eff=ne, p=p, ci_lo=np.tanh(z - 1.96 * se), ci_hi=np.tanh(z + 1.96 * se))


def series(source, v, w):
    a, z = WIN[w]
    if source == "wake":
        return dd[(2023, v, "wake", "anom")].loc[a:z].values
    S = pd.read_csv(OUTB + "/series_daily.csv", index_col=0, parse_dates=True)
    col = {"sst": "sst_uw", "sss": "sss_uw", "no3": "no3_uw", "po4": "po4_uw", "chl": "chl_uw", "npp": "nppv_uw"}[v]
    return S[col].loc[a:z].values


# ---------- 4a correlation matrices ----------
rows = []
for source in ["wake", "basin_raw_as_published"]:
    for w in WIN:
        for i, a in enumerate(VARS):
            for b in VARS[i + 1:]:
                cs = corr_stats(series(source, a, w), series(source, b, w))
                # undetrended, nominal-n version (what the original paper did) for comparison
                xa, xb = series(source, a, w), series(source, b, w); ok = np.isfinite(xa) & np.isfinite(xb)
                r_nom = np.corrcoef(xa[ok], xb[ok])[0, 1]; p_nom = stats.pearsonr(xa[ok], xb[ok])[1]
                rows.append(dict(source=source, window=w, pair=f"{LAB[a]}-{LAB[b]}", r_nominal_no_detrend=r_nom, p_nominal=p_nom, **cs))
C = pd.DataFrame(rows)
# pre vs post difference test
diff = []
for source in C.source.unique():
    for pair in C.pair.unique():
        q = C[(C.source == source) & (C.pair == pair)].set_index("window")
        z1, z0 = np.arctanh(np.clip(q.loc["post", "r"], -.9999, .9999)), np.arctanh(np.clip(q.loc["pre", "r"], -.9999, .9999))
        se = np.sqrt(1 / max(q.loc["post", "n_eff"] - 3, .5) + 1 / max(q.loc["pre", "n_eff"] - 3, .5))
        zz = (z1 - z0) / se
        diff.append(dict(source=source, pair=pair, r_pre=q.loc["pre", "r"], r_post=q.loc["post", "r"], diff=q.loc["post", "r"] - q.loc["pre", "r"],
                         z=zz, p_diff=2 * stats.norm.sf(abs(zz)), nominal_diff_no_detrend=q.loc["post", "r_nominal_no_detrend"] - q.loc["pre", "r_nominal_no_detrend"]))
Dd = pd.DataFrame(diff)
C.to_csv(f"{O}/correlations_effective_n.csv", index=False); Dd.to_csv(f"{O}/correlation_differences.csv", index=False)
pd.set_option("display.width", 250)
print("WAKE CORE, detrended anomalies:"); print(C[C.source == "wake"][["window", "pair", "r", "n", "n_eff", "p", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
print("\nsignificant (p<0.05) pairs: wake", int((C[C.source == "wake"].p < 0.05).sum()), "of", int((C.source == "wake").sum()),
      "| basin as published, detrended+n_eff:", int((C[C.source != "wake"].p < 0.05).sum()), "of", int((C.source != "wake").sum()),
      "| basin as published, nominal:", int((C[C.source != "wake"].p_nominal < 0.05).sum()))
print("\npre->post differences significant at p<0.05:"); print(Dd[Dd.p_diff < 0.05].round(3).to_string(index=False))

# ---------- 4b lag correlation with surrogates ----------
def phase_surrogate(x):
    x = x - x.mean(); n = len(x); F = np.fft.rfft(x); ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0
    if n % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n)


def lagcorr(npp, no3, lags):
    a = pd.Series(npp); b = pd.Series(no3)
    return np.array([a.corr(b.shift(L)) for L in lags])


lags = np.arange(-15, 16); lag_rows = []; env = {}
cases = {"wake, post-event 15 May-30 Jun (detrended)": ("wake", "2023-05-15", "2023-06-30", True),
         "wake, whole Apr-Jun (detrended)": ("wake", "2023-04-01", "2023-06-30", True),
         "basin raw, whole Apr-Jun (as published, sign corrected)": ("basin", "2023-04-01", "2023-06-30", False)}
S0 = pd.read_csv(OUTB + "/series_daily.csv", index_col=0, parse_dates=True)
for name, (src, a, z, dt) in cases.items():
    if src == "wake":
        npp = dd[(2023, "npp", "wake", "anom")].loc[a:z].values; no3 = dd[(2023, "no3", "wake", "anom")].loc[a:z].values
    else:
        npp = S0["nppv_uw"].loc[a:z].values; no3 = S0["no3_uw"].loc[a:z].values
    if dt: npp, no3 = detr(npp), detr(no3)
    r = lagcorr(npp, no3, lags)
    sur = np.array([lagcorr(npp, phase_surrogate(no3), lags) for _ in range(2000)])
    lo, hi = np.percentile(sur, 2.5, 0), np.percentile(sur, 97.5, 0); gmax = np.percentile(np.max(np.abs(sur), 1), 95)
    env[name] = (r, lo, hi, gmax)
    k1, k2 = np.argmax(r), np.argmin(r)
    lag_rows.append(dict(case=name, n_days=len(npp), r_at_0=r[lags == 0][0], max_r=r[k1], lag_of_max=lags[k1],
                         meaning_max=("nitrate leads NPP" if lags[k1] > 0 else "NPP leads nitrate" if lags[k1] < 0 else "simultaneous"),
                         min_r=r[k2], lag_of_min=lags[k2], meaning_min=("nitrate leads NPP" if lags[k2] > 0 else "NPP leads nitrate" if lags[k2] < 0 else "simultaneous"),
                         pointwise_95_at_max=f"[{lo[k1]:.2f}, {hi[k1]:.2f}]", pointwise_95_at_min=f"[{lo[k2]:.2f}, {hi[k2]:.2f}]",
                         global_95_bound_abs_r=gmax, any_lag_globally_significant=bool(np.max(np.abs(r)) > gmax)))
    for L, rv, l_, h_ in zip(lags, r, lo, hi): pass
Lg = pd.DataFrame(lag_rows); Lg.to_csv(f"{O}/lag_correlation_summary.csv", index=False)
pd.DataFrame({f"{n}|{k}": v for n, (r, lo, hi, g) in env.items() for k, v in [("r", r), ("lo95", lo), ("hi95", hi)]}, index=lags).to_csv(f"{O}/lag_correlation_curves.csv")
print("\nLAG CORRELATION (r(L) = corr(NPP(t), NO3(t-L)); L>0: nitrate leads):"); print(Lg.round(3).to_string(index=False))

# ---------- 4b regression (appendix) ----------
reg = []
for w in WIN:
    x = detr(series("wake", "no3", w)); y = detr(series("wake", "npp", w)); ok = np.isfinite(x) & np.isfinite(y)
    lr = stats.linregress(x[ok], y[ok]); cs = corr_stats(series("wake", "no3", w), series("wake", "npp", w))
    se_adj = lr.stderr * np.sqrt((cs["n"] - 2) / max(cs["n_eff"] - 2, 1e-6))
    reg.append(dict(window=w, n=cs["n"], n_eff=cs["n_eff"], slope_mgC_per_mmolN=lr.slope, slope_se_adjusted=se_adj, intercept=lr.intercept,
                    r=lr.rvalue, R2=lr.rvalue ** 2, p_nominal=lr.pvalue, p_effective=cs["p"]))
Rg = pd.DataFrame(reg); Rg.to_csv(f"{O}/regression_wake_detrended.csv", index=False)
print("\nREGRESSION NPP anomaly on NO3 anomaly, wake, detrended:"); print(Rg.round(4).to_string(index=False))

# ---------- figures ----------
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, w in zip(axs[:2], ["pre", "post"]):
    M = np.full((6, 6), np.nan); P = np.full((6, 6), np.nan)
    for _, rr in C[(C.source == "wake") & (C.window == w)].iterrows():
        a, b = rr.pair.split("-"); i, j = [list(LAB.values()).index(a), list(LAB.values()).index(b)]
        M[i, j] = M[j, i] = rr.r; P[i, j] = P[j, i] = rr.p
    np.fill_diagonal(M, 1)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(6):
        for j in range(6):
            if i != j: ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center", fontsize=8,
                               color="k" if abs(M[i, j]) < 0.6 else "w", fontweight="bold" if P[i, j] < 0.05 else "normal")
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(LAB.values()); ax.set_yticklabels(LAB.values())
    ax.set_title(f"Wake core, {w}-event ({WIN[w][0][5:]} to {WIN[w][1][5:]})\ndetrended anomalies; * p<0.05 with n$_{{eff}}$", fontsize=9)
fig.colorbar(im, ax=axs[:2], shrink=0.8, label="Pearson r (same scale both panels)")
ax = axs[2]
cols = ["#d7301f", "#fc8d59", "0.4"]
for (name, (r, lo, hi, g)), c in zip(env.items(), cols):
    ax.plot(lags, r, "-o", ms=3, color=c, label=name)
    if name.startswith("wake, post"): ax.fill_between(lags, lo, hi, color=c, alpha=0.12, label="95% bounds, wake post-event")
    if name.startswith("basin"): ax.fill_between(lags, lo, hi, color="0.5", alpha=0.18, label="95% bounds, basin as published")
    if name.startswith("wake, whole"): ax.axhline(-g, color=c, ls="--", lw=0.9, label=f"global 95% |r| bound, wake whole record ({g:.2f})")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.4, ls=":")
ax.set_xlabel("lag L (days)   ←  NPP leads nitrate   |   nitrate leads NPP  →"); ax.set_ylabel("r(L) = corr(NPP(t), NO$_3$(t−L))")
ax.legend(fontsize=6.5, loc="lower left"); ax.grid(alpha=0.3); ax.set_title("Lag correlation, explicit convention", fontsize=9)
fig.savefig(f"{O}/fig_s4_correlations_lag.png", dpi=150, bbox_inches="tight")
