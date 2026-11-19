import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""Step 2 figures: (A) daily 2023 anomalies by region with interannual envelope, (B) maps of the
storm-period anomaly change, (C) 2023 vs 33 seasons. Also writes recovery times vs climatology."""
import numpy as np, pandas as pd, scipy.io as sio, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
O = OUTB + "/step2"
dd = pd.read_pickle(f"{O}/daily_region_series.pkl"); P = pd.read_pickle(f"{O}/pentad_region_anoms.pkl")
G = np.load(f"{O}/geometry.npz"); LAT, LON = G["lat"], G["lon"]
tr = pd.read_csv(f"{O}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True)
QUIET = [1993, 1995, 2000, 2001, 2005, 2012, 2014, 2015, 2025]
storms = pd.read_csv(f"{O}/bob_storms_AprSep_1993_2025.csv"); STORMY = sorted(storms[storms.AprJun].year.unique())
VARS = [("sst", "SST anomaly (°C)"), ("sss", "SSS anomaly (psu)"), ("no3", "NO$_3$ anomaly (mmol m$^{-3}$)"),
        ("po4", "PO$_4$ anomaly (mmol m$^{-3}$)"), ("chl", "Chl anomaly (mg m$^{-3}$)"), ("npp", "NPP anomaly (mg C m$^{-3}$ d$^{-1}$)")]
COL = {"wake": "#d7301f", "outer": "#fc8d59", "control": "#4575b4", "shelf": "#74add1"}
LAB = {"wake": "wake core 0–200 km", "outer": "outer wake 200–400 km", "control": "control (calm, W Bay)", "shelf": "shelf <200 m"}
days = pd.date_range("2023-04-01", "2023-09-30")


def sigma_daily(var, region):
    s = P[(var, region)].loc[1993:2022].std(ddof=1).values      # per pentad
    idx = np.minimum((days - pd.Timestamp("2023-04-01")).days // 5, 35)
    return pd.Series(s[idx], index=days)


# ---------- (A) daily anomaly series ----------
fig, axs = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True)
rec = []
for ax, (v, lab) in zip(axs.flat, VARS):
    sd = sigma_daily(v, "wake")
    ax.fill_between(days, -2 * sd, 2 * sd, color="0.9", lw=0, label="±2σ interannual (wake, 1993–2022)")
    ax.fill_between(days, -sd, sd, color="0.8", lw=0, label="±1σ")
    for r in ["control", "shelf", "outer", "wake"]:
        s = dd[(2023, v, r, "anom")].dropna()
        ax.plot(s.index, s.values, color=COL[r], lw=2.0 if r == "wake" else 1.2, ls="--" if r == "shelf" else "-", label=LAB[r])
    if v in ("sst", "sss"):   # pentad continuation for Jul-Sep
        pw = P[(v, "wake")].loc[2023].values; pdays = pd.Timestamp("2023-04-01") + pd.to_timedelta(np.arange(36) * 5 + 2, "D")
        ax.plot(pdays[18:], pw[18:], "o-", color=COL["wake"], ms=3, lw=1, alpha=0.7, label="wake, 5-day means (Jul–Sep)")
    ax.axvspan(pd.Timestamp("2023-05-09"), pd.Timestamp("2023-05-14 23:00"), color="k", alpha=0.08)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(lab, fontsize=9); ax.grid(alpha=0.25)
    # recovery in wake: after the extreme (9 May - 31 May), first day back inside +-1 sigma, sustained 7 d
    s = dd[(2023, v, "wake", "anom")].dropna(); pre = s.loc["2023-05-01":"2023-05-07"].mean()
    ev = s.loc["2023-05-09":"2023-05-31"]; ke = (ev - pre).abs().idxmax()
    for name, ok in [("within 1σ of climatology", s.loc[ke:].abs() <= sd.loc[s.loc[ke:].index]),
                     ("back to pre-storm anomaly ±0.5σ", (s.loc[ke:] - pre).abs() <= 0.5 * sd.loc[s.loc[ke:].index])]:
        run = ok.astype(int).groupby((~ok).cumsum()).cumsum(); hit = run[run >= 7].index
        d0 = (hit[0] - pd.Timedelta(days=6)) if len(hit) else None
        rec.append(dict(variable=v, extreme_date=ke.date(), extreme_anom=s[ke], pre_storm_anom_1_7May=pre, storm_change=s[ke] - pre, storm_change_in_sigma=(s[ke] - pre) / sd[ke],
                        criterion=name, recovery_date=d0.date() if d0 is not None else f"not by {s.index[-1].date()}",
                        days_after_landfall=(d0 - pd.Timestamp("2023-05-14")).days if d0 is not None else None))
axs[0, 0].legend(fontsize=7, loc="lower left", ncol=2)
for ax in axs[-1]: ax.set_xlim(days[0], days[-1])
fig.suptitle("2023 anomalies relative to the 1993–2022 daily climatology (area-weighted region means; grey bar = 9–14 May)", fontsize=11)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"{O}/fig_s2_anomaly_series.png", dpi=150)
R = pd.DataFrame(rec); R.to_csv(f"{O}/step2_recovery_vs_climatology.csv", index=False)
print(R.round(3).to_string(index=False))

# ---------- (B) maps: anomaly change 15-19 May minus 1-7 May ----------
g = sio.netcdf_file(DATA + "/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::10].copy(); go = g.variables["lon"][::10].copy(); ge = g.variables["elevation"][::10, ::10].astype(float)
tb = pd.to_datetime(np.load(f"{O}/time2023_bgc.npy")); tp = pd.to_datetime(np.load(f"{O}/time2023_phy.npy"))
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
lims = {"sst": 2.0, "sss": 1.0, "no3": 1.5, "po4": 0.03, "chl": 0.15, "npp": 6}
for ax, (v, lab) in zip(axs.flat, VARS):
    a = np.load(f"{O}/anom2023_{v}.npy"); t = tp if v in ("sst", "sss") else tb
    d = np.nanmean(a[(t >= "2023-05-15") & (t <= "2023-05-19")], 0) - np.nanmean(a[(t >= "2023-05-01") & (t <= "2023-05-07")], 0)
    L = lims[v]; cmap = "RdBu_r" if v in ("sst",) else "BrBG_r" if v == "sss" else "PuOr_r" if v in ("no3", "po4") else "PiYG"
    if v == "sst": cmap = "RdBu_r"
    pc = ax.pcolormesh(LON, LAT, d, cmap=cmap, vmin=-L, vmax=L, shading="nearest")
    ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"]); ax.contour(go, gl, ge, levels=[0], colors="0.3", linewidths=0.4)
    ax.contour(LON, LAT, G["region"] == 1, levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
    ax.contour(LON, LAT, G["region"] == 3, levels=[0.5], colors="#4575b4", linewidths=0.8)
    oc = tr.loc["2023-05-09":"2023-05-14 06:00"]; ax.plot(oc.lon, oc.lat, "k-", lw=1.6)
    ax.set_xlim(78, 100); ax.set_ylim(5, 25); ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    fig.colorbar(pc, ax=ax, shrink=0.8, extend="both"); ax.set_title(lab.replace("anomaly", "Δanomaly"), fontsize=10)
fig.suptitle("Change in anomaly: mean 15–19 May minus mean 1–7 May 2023 (seasonal cycle removed). Dashed = wake core, blue = control.", fontsize=11)
fig.tight_layout(); fig.savefig(f"{O}/fig_s2_change_maps.png", dpi=150)

# ---------- (C) 2023 among 33 seasons ----------
fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
items = [("sst", "wake excess ΔSST (°C)"), ("sss", "wake excess ΔSSS (psu)"), ("no3", "wake excess ΔNO$_3$"), ("po4", "wake excess ΔPO$_4$"),
         ("chl", "wake excess ΔChl"), ("npp", "wake excess ΔNPP"), ("chl", "late-June Chl anomaly, open ocean"), ("npp", "late-June NPP anomaly, open ocean")]
for k, (ax, (v, lab)) in enumerate(zip(axs.flat, items)):
    if k < 6:
        W, C = P[(v, "wake")], P[(v, "control")]; s = (W[9] - W[6]) - (C[9] - C[6])
    else:
        s = P[(v, "open_all")][[17, 18]].mean(axis=1)
    x = np.zeros(len(s)) + np.random.default_rng(1).uniform(-0.15, 0.15, len(s))
    for i, (y, val) in enumerate(s.items()):
        c, mk, sz = ("#d7301f", "*", 220) if y == 2023 else ("#2c7fb8", "o", 40) if y in QUIET else ("#fdae61", "o", 40)
        ax.scatter(x[i], val, c=c, marker=mk, s=sz, edgecolor="k", lw=0.4, zorder=3 if y == 2023 else 2)
        if y in (2023, 2024, 2025) or abs(val) >= np.sort(abs(s.values))[-3]:
            ax.annotate(str(y), (x[i], val), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
    ref = s.loc[1993:2022]; ax.axhspan(ref.mean() - ref.std(), ref.mean() + ref.std(), color="0.85", zorder=0)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_title(lab, fontsize=9)
    z = (s.loc[2023] - ref.mean()) / ref.std(); ax.text(0.02, 0.02, f"2023: z = {z:+.1f}", transform=ax.transAxes, fontsize=8)
axs[0, 0].scatter([], [], c="#2c7fb8", label="no ≥34 kt storm Apr–Jun"); axs[0, 0].scatter([], [], c="#fdae61", label="storm year"); axs[0, 0].scatter([], [], c="#d7301f", marker="*", s=120, label="2023 (Mocha)")
axs[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("2023 among 33 seasons (1993–2025). Wake excess = (16–20 May minus 1–5 May anomaly change) in wake core minus the same in the control. Grey = ±1σ of 1993–2022.", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig_s2_interannual.png", dpi=150)
