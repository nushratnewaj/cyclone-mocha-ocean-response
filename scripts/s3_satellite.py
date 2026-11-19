import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 3f - does satellite chlorophyll show the same wake pulse as the model?
Satellite: cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D (version 202603), CHL, 4 km
           -> 6x6 block mean -> 0.25 deg grid (Colab block K). Gap-free L4: cloudy days are filled by
           interpolation, so values during the storm itself (9-14 May) are partly interpolated.
Model:     GLOBAL_MULTIYEAR_BGC_001_029 surface chl (0.5 m).
Both on the common ocean mask; region means cos-lat weighted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h5lite import File
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
D = DATA + ""; O2 = OUTB + "/step2"; O = OUTB + "/step3"
G = np.load(f"{O2}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; REG = G["region"]; SIDE = G["side"]; OCEAN = G["ocean"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
R = {"wake": REG == 1, "wake_right": (REG == 1) & (SIDE == 1), "wake_left": (REG == 1) & (SIDE == -1),
     "outer": REG == 2, "control": REG == 3, "shelf": REG == 4}


def rmean(a, m):
    w = WT * m; ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


f = File(f"{D}/dl/satellite_chl_2023_0p25.nc")
ts_ = pd.Timestamp("1900-01-01") + pd.to_timedelta(f.read("time"), unit="D"); ts_ = ts_.floor("D")
sat = f.read("chl_sat")
fb = File(f"{D}/nutrients_april-june-2023.nc"); tb = pd.to_datetime(fb.read("time"), unit="s")
kb = np.where((tb >= ts_[0]) & (tb <= ts_[-1]))[0]
mod = fb.read("chl", sel=((kb[0], len(kb)), (0, 1), (0, 81), (0, 89)))[:, 0]; tm = tb[kb]
assert (tm == ts_).all()
both = OCEAN[None] & np.isfinite(sat) & np.isfinite(mod)
sat = np.where(both, sat, np.nan); mod = np.where(both, mod, np.nan)

S = pd.DataFrame({(src, rn): rmean(a, rm) for src, a in [("satellite", sat), ("model", mod)] for rn, rm in R.items()}, index=ts_)
S.to_csv(f"{O}/satellite_vs_model_chl_regions.csv")

rows = []
for src in ["satellite", "model"]:
    for rn in R:
        s = S[(src, rn)]; pre = s.loc["2023-05-01":"2023-05-07"].mean()
        post = s.loc["2023-05-15":"2023-05-19"].mean(); post2 = s.loc["2023-05-20":"2023-05-26"].mean()
        pk = s.loc["2023-05-09":"2023-05-31"].idxmax()
        rows.append(dict(source=src, region=rn, pre_1_7May=pre, post_15_19May=post, post_20_26May=post2,
                         change_15_19=post - pre, pct_change_15_19=100 * (post - pre) / pre, change_20_26=post2 - pre,
                         peak_date_9_31May=pk.date(), peak_value=s[pk], peak_pct_above_pre=100 * (s[pk] - pre) / pre))
T = pd.DataFrame(rows)
for src in ["satellite", "model"]:
    w = T[(T.source == src) & (T.region == "wake")].iloc[0]; c = T[(T.source == src) & (T.region == "control")].iloc[0]
    rows.append(dict(source=src, region="wake minus control", change_15_19=w.change_15_19 - c.change_15_19,
                     pct_change_15_19=w.pct_change_15_19 - c.pct_change_15_19, change_20_26=w.change_20_26 - c.change_20_26))
T = pd.DataFrame(rows); T.to_csv(f"{O}/satellite_vs_model_chl_summary.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))

# spatial agreement of the change pattern (open ocean, not shelf)
opn = OCEAN & ~G["shelf"]
def change(a, t, w0=("2023-05-01", "2023-05-07"), w1=("2023-05-15", "2023-05-19")):
    return np.nanmean(a[(t >= w1[0]) & (t <= w1[1])], 0) - np.nanmean(a[(t >= w0[0]) & (t <= w0[1])], 0)
dS = change(sat, ts_); dM = change(mod, tm)
m = opn & np.isfinite(dS) & np.isfinite(dM)
r_all = np.corrcoef(dS[m], dM[m])[0, 1]
near = m & (G["dist_all_km"] < 600)
r_near = np.corrcoef(dS[near], dM[near])[0, 1]
lr = np.corrcoef(np.log10(np.nanmean(sat[(ts_ >= "2023-05-01") & (ts_ <= "2023-05-07")], 0)[m]),
                 np.log10(np.nanmean(mod[(tm >= "2023-05-01") & (tm <= "2023-05-07")], 0)[m]))[0, 1]
print(f"\nspatial r of change maps (15-19 May minus 1-7 May): open ocean {r_all:.2f} (n={m.sum()}), within 600 km of track {r_near:.2f} (n={near.sum()})")
print(f"pre-storm (1-7 May) log10 chl spatial r, satellite vs model: {lr:.2f}")
print("pre-storm open-ocean median chl: satellite", np.nanmedian(np.nanmean(sat[(ts_ >= '2023-05-01') & (ts_ <= '2023-05-07')], 0)[m]).round(3),
      "| model", np.nanmedian(np.nanmean(mod[(tm >= '2023-05-01') & (tm <= '2023-05-07')], 0)[m]).round(3))
pd.DataFrame([dict(spatial_r_open_ocean=r_all, n_open=int(m.sum()), spatial_r_within_600km=r_near, n_near=int(near.sum()), pre_storm_log_chl_r=lr)]).to_csv(f"{O}/satellite_vs_model_spatial.csv", index=False)

# ---------- figure ----------
g = sio.netcdf_file(f"{D}/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::10].copy(); go = g.variables["lon"][::10].copy(); ge = g.variables["elevation"][::10, ::10].astype(float)
tr = pd.read_csv(f"{O2}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True).loc["2023-05-09":"2023-05-14 06:00"]
fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
for j, (lab, d) in enumerate([("Satellite (OC L4 gap-free)", dS), ("Model (BGC 001_029, 0.5 m)", dM)]):
    ax = fig.add_subplot(gs[0, j]); pc = ax.pcolormesh(LON, LAT, np.where(OCEAN, d, np.nan), cmap="PiYG", vmin=-0.15, vmax=0.15, shading="nearest")
    ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"]); ax.contour(go, gl, ge, levels=[0], colors="0.3", linewidths=0.4)
    ax.plot(tr.lon, tr.lat, "k-", lw=1.5); ax.contour(LON, LAT, REG == 1, levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
    ax.contour(LON, LAT, REG == 3, levels=[0.5], colors="#2171b5", linewidths=0.8)
    ax.set_xlim(78, 100); ax.set_ylim(5, 25); ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    ax.set_title(f"{lab}\nchl change, 15–19 May minus 1–7 May (mg m$^{{-3}}$)", fontsize=9); fig.colorbar(pc, ax=ax, shrink=0.8, extend="both")
ax = fig.add_subplot(gs[0, 2])
ax.scatter(dS[m & ~near], dM[m & ~near], s=5, c="0.7", label="open ocean >600 km")
ax.scatter(dS[near], dM[near], s=7, c="#d7301f", label="within 600 km of track")
ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5); ax.plot([-0.3, 0.6], [-0.3, 0.6], "k--", lw=0.7)
ax.set_xlim(-0.2, 0.5); ax.set_ylim(-0.2, 0.5); ax.set_xlabel("satellite change (mg m$^{-3}$)"); ax.set_ylabel("model change (mg m$^{-3}$)")
ax.set_title(f"cell-by-cell change, r = {r_all:.2f} (open ocean), {r_near:.2f} (≤600 km)", fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
for j, (rn, lab) in enumerate([("wake_right", "wake core, right of track"), ("wake_left", "wake core, left of track"), ("control", "control (calm W Bay)")]):
    ax = fig.add_subplot(gs[1, j])
    ax.plot(S.index, S[("satellite", rn)], color="#1b7837", lw=2, label="satellite")
    ax.plot(S.index, S[("model", rn)], color="#762a83", lw=1.6, ls="--", label="model")
    ax.axvspan(pd.Timestamp("2023-05-09"), pd.Timestamp("2023-05-14 23:00"), color="k", alpha=0.1)
    ax.set_title(lab, fontsize=10); ax.set_ylabel("surface chl (mg m$^{-3}$)"); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=8)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"{O}/fig_s3_satellite_chl.png", dpi=150)
