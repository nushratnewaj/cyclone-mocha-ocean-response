import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 1c - corrected headline numbers with a clean, consistent set of choices:
  * cos(lat) area-weighted domain means, surface level only
  * baseline  = 25 Apr - 8 May 2023 (14 d ending the day before the 9 May depression)
  * event     = 9 - 14 May 2023 (depression -> landfall); post-event from 15 May
  * recovery  = evaluated under four threshold definitions (sensitivity)
Also: an early look at the extended BGC record (to Aug 2023) and at the same
season in 2024 / 2025, which the reviewer's 'peaks at edge of record' and
'seasonal cycle' comments hinge on.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUT = OUTB + ""
S = pd.read_csv(f"{OUT}/series_daily.csv", index_col=0, parse_dates=True)
VW = {"chl": "chl_w", "no3": "no3_w", "po4": "po4_w", "npp": "nppv_w", "sst": "sst_w", "sss": "sss_w"}
UNIT = {"chl": "mg m-3", "no3": "mmol m-3", "po4": "mmol m-3", "npp": "mg C m-3 d-1", "sst": "degC", "sss": "psu"}
B0, B1 = "2023-04-25", "2023-05-08"; EV0, EV1 = "2023-05-09", "2023-05-14"; P0 = pd.Timestamp("2023-05-15")
LANDFALL = pd.Timestamp("2023-05-14")

rows = []; rec_rows = []
for k, c in VW.items():
    s = S[c].dropna()
    s_jun = s.loc["2023-04-01":"2023-06-30"]
    b = s.loc[B0:B1]; mu = b.mean()
    sd_det = signal.detrend(b.values).std(ddof=1)
    a = s - mu
    post = a.loc[P0:"2023-06-30"]
    # largest excursion in the 15 May - 30 Jun window, as in the paper, plus its sign
    pk = post.abs().idxmax(); pv = a[pk]
    # event-window extreme (9 May - 20 May): the part most plausibly attributable to the storm
    ev = a.loc[EV0:"2023-05-20"]; evk = ev.abs().idxmax()
    rows.append(dict(variable=k, unit=UNIT[k], baseline_mean=mu, baseline_sd_detrended=sd_det,
                     storm_window_extreme=a[evk], storm_window_date=evk.date(), storm_window_pct=100 * a[evk] / mu,
                     max_excursion_to_30Jun=pv, max_excursion_date=pk.date(),
                     max_on_last3days=pk >= pd.Timestamp("2023-06-28")))
    # recovery sensitivity: measured from the storm-window extreme
    thr = {"10% of peak anomaly (as applied)": 0.10 * abs(a[evk]),
           "10% of baseline mean (as stated)": 0.10 * abs(mu),
           "1 SD of detrended baseline": sd_det,
           "2 SD of detrended baseline": 2 * sd_det}
    after = a.loc[evk + pd.Timedelta(days=1):"2023-06-30"]
    for name, th in thr.items():
        inside = after.abs() <= th
        first = inside[inside].index[0] if inside.any() else None
        # sustained: first day after which the anomaly stays inside for >= 7 consecutive days
        run = inside.astype(int).groupby((~inside).cumsum()).cumsum()
        sus = run[run >= 7].index
        sustained = (sus[0] - pd.Timedelta(days=6)) if len(sus) else None
        rec_rows.append(dict(variable=k, threshold=name, threshold_value=th,
                             first_return=first.date() if first is not None else "none by 30 Jun",
                             days_from_landfall=(first - LANDFALL).days if first is not None else None,
                             sustained_7d_return=sustained.date() if sustained is not None else "none by 30 Jun"))

H = pd.DataFrame(rows); R = pd.DataFrame(rec_rows)

# nitrate: percentage drop and quick stoichiometric check (reviewer's General Comment 3)
no3 = S[VW["no3"]]; chl = S[VW["chl"]]; po4 = S[VW["po4"]]
d0, d1 = "2023-05-09", "2023-05-16"
dN = no3[d1] - no3[d0]; dChl = chl[d1] - chl[d0]; dP = po4[d1] - po4[d0]
C_from_N = -dN * 106 / 16 * 12.011          # mg C m-3 if all removed N became organic matter (Redfield)
stoich = dict(window=f"{d0} -> {d1}", dNO3_mmol=dN, dPO4_mmol=dP, dChl_mg=dChl,
              C_equiv_mgC=C_from_N, Chl_equiv_at_CChl50=C_from_N / 50, Chl_equiv_at_CChl100=C_from_N / 100)

# lag correlation on area-weighted series, whole Apr-Jun record (same as paper), with explicit convention
A = S.loc["2023-04-01":"2023-06-30"]
x = A[VW["no3"]]; y = A[VW["npp"]]
lags = np.arange(-15, 16)
lc = pd.Series([y.corr(x.shift(L)) for L in lags], index=lags)  # r(L)=corr(NPP(t), NO3(t-L)); L>0: NO3 leads

# extended record + other years (BGC only)
ext = {}
for yr in [2023, 2024, 2025]:
    seg = S.loc[f"{yr}-04-01":f"{yr}-08-31", [VW[v] for v in ["chl", "no3", "po4", "npp"]]]
    seg.index = seg.index.dayofyear; ext[yr] = seg
pk_ext = {v: S[VW[v]].loc["2023-04-01":"2023-08-31"].idxmax().date() for v in ["chl", "npp", "po4"]}

with pd.ExcelWriter(f"{OUT}/corrected_headlines.xlsx") as w:
    H.to_excel(w, "headline_numbers", index=False)
    R.to_excel(w, "recovery_sensitivity", index=False)
    pd.DataFrame([stoich]).to_excel(w, "nitrate_stoich_check", index=False)
    lc.rename("r").to_frame().to_excel(w, "lagcorr_weighted")

pd.set_option("display.width", 220)
print(H.round(4).to_string()); print(); print(R.to_string()); print(); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stoich.items()})
print("\nlag corr (weighted): max", round(lc.max(), 3), "at", lc.idxmax(), "| min", round(lc.min(), 3), "at", lc.idxmin(), "| r(0)", round(lc[0], 3))
print("peaks when record extended to 31 Aug 2023:", pk_ext)

# ---------------- figures ----------------
fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
old_base = S.loc["2023-04-30":"2023-05-13"]
for ax, k in zip(axs.flat, ["sst", "sss", "chl", "no3", "po4", "npp"]):
    s = S[VW[k]].loc["2023-04-01":"2023-06-30"]
    ax.plot(s.index, s, color="0.15", lw=1.4)
    ax.axvspan(pd.Timestamp(B0), pd.Timestamp(B1) + pd.Timedelta(hours=23), color="tab:blue", alpha=0.12, label="new baseline 25 Apr-8 May")
    ax.axvspan(pd.Timestamp("2023-04-30"), pd.Timestamp("2023-05-13") + pd.Timedelta(hours=23), ymin=0, ymax=0.06, color="tab:red", alpha=0.6, label="original baseline 30 Apr-13 May")
    ax.axvspan(pd.Timestamp(EV0), pd.Timestamp(EV1) + pd.Timedelta(hours=23), color="0.6", alpha=0.35, label="event 9-14 May")
    mu = S[VW[k]].loc[B0:B1].mean(); ax.axhline(mu, color="tab:blue", lw=0.8, ls="--")
    ax.set_title(f"{k.upper()} ({UNIT[k]}) - area-weighted, surface", fontsize=10); ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7, loc="lower right"); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(f"{OUT}/check_series_baselines.png", dpi=130)

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
cols = {2023: "tab:red", 2024: "tab:orange", 2025: "tab:blue"}
for ax, v in zip(axs.flat, ["chl", "npp", "no3", "po4"]):
    for yr, seg in ext.items():
        lab = f"{yr}" + (" (Mocha, 9-14 May)" if yr == 2023 else " (Remal, 23-28 May)" if yr == 2024 else " (no BoB cyclone Apr-Jun in IBTrACS)")
        ax.plot(seg.index, seg[VW[v]], color=cols[yr], lw=1.5, label=lab)
    ax.axvspan(129, 134, color="tab:red", alpha=0.12)
    ax.axvline(181, color="k", lw=0.8, ls=":")
    ax.text(182, ax.get_ylim()[1], " end of paper's record (30 Jun)", fontsize=7, va="top")
    ax.set_title(f"{v.upper()} ({UNIT[v]}) - area-weighted domain mean", fontsize=10); ax.grid(alpha=0.3)
    ax.set_xticks([91, 121, 152, 182, 213, 244]); ax.set_xticklabels(["1 Apr", "1 May", "1 Jun", "1 Jul", "1 Aug", "1 Sep"])
axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/check_years_2023_2025.png", dpi=130)
