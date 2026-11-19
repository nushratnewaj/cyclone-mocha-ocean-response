import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 3d - (1) model surface/upper-ocean NO3 and PO4 vs World Ocean Atlas 2023 (1 deg, Apr-Jun,
objectively analysed means), and (2) Ekman pumping from ERA5 wind stress.
WOA units umol/kg are converted to mmol m-3 with rho = 1.022 kg L-1 (warm, fresh surface water).
Model: 1993-2022 daily climatology (surface) averaged by month, block-averaged to the WOA 1 deg cells;
profiles: 2023 monthly means from the 0-200 m BGC profile file (2023 only).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h5lite import File
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
D = DATA + ""; DL = f"{D}/dl"; O2 = OUTB + "/step2"; O = OUTB + "/step3"
G = np.load(f"{O2}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; OCEAN = G["ocean"]; SHELF = G["shelf"]; REG = G["region"]; SIDE = G["side"]
RHO = 1.022

w = File(f"{DL}/woa23_no3_po4_AprJun_0_100m.nc")
wl = w.read("lat"); wo = w.read("lon"); wz = w.read("depth"); months = w.read("month").astype(int)
WN = w.read("n_an") * RHO; WP = w.read("p_an") * RHO; WNd = w.read("n_dd", raw=True); WPd = w.read("p_dd", raw=True)
WN[WN > 1e30] = np.nan; WP[WP > 1e30] = np.nan

cb = File(f"{DL}/bgc_surface_clim_doy.nc"); doy = cb.read("doy")
mdoy = {4: (91, 120), 5: (121, 151), 6: (152, 181)}
rows = []; pairs = {"no3": [], "po4": []}
for mi, m in enumerate(months):
    a, z = mdoy[m]; k = (doy >= a) & (doy <= z)
    for v, Wv, Wd in [("no3", WN, WNd), ("po4", WP, WPd)]:
        mod = np.nanmean(cb.read(f"{v}_mean")[k], 0)
        mod = np.where(OCEAN & ~SHELF, mod, np.nan)               # open ocean only
        for i, la in enumerate(wl):
            for j, lo in enumerate(wo):
                mm = (np.abs(LAT - la) < 0.5)[:, None] & (np.abs(LON - lo) < 0.5)[None, :]
                mv = np.nanmean(np.where(mm, mod, np.nan))
                ov = Wv[mi, 0, i, j]
                if np.isfinite(mv) and np.isfinite(ov):
                    pairs[v].append(dict(month=m, lat=la, lon=lo, model=mv, woa=ov, n_obs=int(Wd[mi, 0, i, j]) if Wd[mi, 0, i, j] > -1 else 0))
for v in pairs:
    df = pd.DataFrame(pairs[v]); df.to_csv(f"{O}/woa_vs_model_surface_{v}.csv", index=False)
    for lab, sub in [("all 1-deg cells", df), ("cells with >=1 observation", df[df.n_obs >= 1])]:
        rows.append(dict(variable=v, subset=lab, n_cells=len(sub), model_median=sub.model.median(), woa_median=sub.woa.median(),
                         median_bias=(sub.model - sub.woa).median(), model_over_woa_ratio_median=(sub.model / sub.woa).median()))
V = pd.DataFrame(rows); print(V.round(4).to_string(index=False)); V.to_csv(f"{O}/woa_validation_summary.csv", index=False)

# N:P at the surface
dn = pd.DataFrame(pairs["no3"]).merge(pd.DataFrame(pairs["po4"]), on=["month", "lat", "lon"], suffixes=("_n", "_p"))
print("surface N:P median  model:", round((dn.model_n / dn.model_p).median(), 1), " WOA:", round((dn.woa_n / dn.woa_p).median(), 1))

# profile comparison for the wake-core area in May: model 2023 vs WOA May
b = File(f"{DL}/bgc_profiles_2023_AprAug.nc"); Z = b.read("depth"); T = pd.date_range("2023-04-01", periods=153)
kmay = (T >= "2023-05-01") & (T <= "2023-05-08")
wake = REG == 1
WTg = np.cos(np.deg2rad(LAT))[:, None] * wake
def rm(a):
    ok = np.isfinite(a); return np.nansum(np.where(ok, a, 0) * WTg, axis=(-2, -1)) / np.sum(np.where(ok, WTg, 0), axis=(-2, -1))
mod_n = np.nanmean(rm(b.read("no3")[kmay]), 0); mod_p = np.nanmean(rm(b.read("po4")[kmay]), 0)
wm = np.zeros((wl.size, wo.size), bool)
for i, la in enumerate(wl):
    for j, lo in enumerate(wo):
        mm = (np.abs(LAT - la) < 0.5)[:, None] & (np.abs(LON - lo) < 0.5)[None, :]
        wm[i, j] = (wake & mm).sum() >= 4
mi5 = list(months).index(5)
woa_n = np.array([np.nanmean(WN[mi5, d][wm]) for d in range(wz.size)]); woa_p = np.array([np.nanmean(WP[mi5, d][wm]) for d in range(wz.size)])
cnt_n = np.array([int(np.nansum(np.where(WNd[mi5, d][wm] > 0, WNd[mi5, d][wm], 0))) for d in range(wz.size)])
pd.DataFrame({"depth": wz, "woa_no3": woa_n, "woa_po4": woa_p, "woa_no3_obs_count": cnt_n}).to_csv(f"{O}/woa_may_profile_wake.csv", index=False)
print("WOA May profile over wake (depth, NO3, PO4, n_obs):"); print(np.c_[wz, woa_n.round(2), woa_p.round(3), cnt_n][:14])
print("model 2023 1-8 May wake profile (depth, NO3, PO4):"); print(np.c_[Z, mod_n.round(2), mod_p.round(3)][1:10])

# ---------- Ekman pumping ----------
e = File(f"{DL}/era5_daily_2023_AprAug.nc"); tx = e.read("taux"); ty = e.read("tauy")
f = 2 * 7.2921e-5 * np.sin(np.deg2rad(LAT))[:, None]
dy = 6371e3 * np.deg2rad(0.25); dx = dy * np.cos(np.deg2rad(LAT))[:, None]
curl = (np.gradient(ty, axis=2) / dx) - (np.gradient(tx, axis=1) / dy)
wE = curl / (1025.0 * f) * 86400.0                                   # m day-1, positive = upwelling
Te = pd.date_range("2023-04-01", periods=tx.shape[0])
ks = (Te >= "2023-05-09") & (Te <= "2023-05-15")
cum = np.nansum(wE[ks], 0)                                           # metres of displacement over 9-15 May
np.save(f"{O}/ekman_cum_9_15May.npy", cum)
WT = np.cos(np.deg2rad(LAT))[:, None]
for rn, m in [("wake", REG == 1), ("wake_right", (REG == 1) & (SIDE == 1)), ("wake_left", (REG == 1) & (SIDE == -1)), ("control", REG == 3)]:
    mm = m & np.isfinite(cum)
    print(f"Ekman displacement 9-15 May, {rn:10s}: mean {np.sum(cum[mm] * WT.repeat(89, 1)[mm]) / np.sum(WT.repeat(89, 1)[mm]):+.1f} m, 90th pct {np.percentile(cum[mm], 90):+.1f} m")

# ---------- figure ----------
g = sio.netcdf_file(f"{D}/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::10].copy(); go = g.variables["lon"][::10].copy(); ge = g.variables["elevation"][::10, ::10].astype(float)
tr = pd.read_csv(f"{O2}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True).loc["2023-05-09":"2023-05-14 06:00"]
fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
for ax, v, lab in [(axs[0], "no3", "NO$_3$"), (axs[1], "po4", "PO$_4$")]:
    df = pd.DataFrame(pairs[v]); obs = df.n_obs >= 1
    ax.scatter(df.woa[~obs], df.model[~obs], s=10, c="0.6", label="WOA cell without data (interpolated)")
    ax.scatter(df.woa[obs], df.model[obs], s=16, c="#d7301f", label="WOA cell with ≥1 observation")
    mx = max(df.woa.max(), df.model.max()) * 1.05; ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.set_xlabel(f"WOA23 surface {lab} (mmol m$^{{-3}}$)"); ax.set_ylabel(f"model 1993–2022 clim. {lab}"); ax.set_title(f"Surface {lab}, Apr–Jun, open ocean (1° cells)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axs[2]
ax.plot(mod_n[1:], Z[1:], "-", color="#d7301f", label="model NO$_3$, wake, 1–8 May 2023"); ax.plot(woa_n, wz, "--", color="#d7301f", label="WOA23 NO$_3$, May")
ax2 = ax.twiny(); ax2.plot(mod_p[1:], Z[1:], "-", color="#2171b5"); ax2.plot(woa_p, wz, "--", color="#2171b5"); ax2.set_xlabel("PO$_4$ (mmol m$^{-3}$, blue)", color="#2171b5")
ax.set_ylim(100, 0); ax.set_xlim(0, 8); ax2.set_xlim(0, 1.0); ax.set_title("Wake area, May: model 2023 vs WOA23 (no WOA obs here in May)", fontsize=8); ax.set_xlabel("NO$_3$ (mmol m$^{-3}$, red)", color="#d7301f"); ax.set_ylabel("Depth (m)"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
ax = axs[3]; pc = ax.pcolormesh(LON, LAT, np.where(OCEAN, cum, np.nan), cmap="RdBu_r", vmin=-40, vmax=40, shading="nearest")
ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"]); ax.contour(go, gl, ge, levels=[0], colors="0.3", linewidths=0.4)
ax.plot(tr.lon, tr.lat, "k-", lw=1.5); ax.set_xlim(78, 100); ax.set_ylim(5, 25); ax.set_aspect(1 / np.cos(np.deg2rad(15)))
ax.set_title("ERA5 Ekman displacement 9–15 May (m, + = upwelling)", fontsize=9); fig.colorbar(pc, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f"{O}/fig_s3_woa_ekman.png", dpi=150)
