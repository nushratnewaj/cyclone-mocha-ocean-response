import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 3a/3b - stratification, barrier layer and nutrient profiles (2023, Apr-Aug, 0.25 deg).
Composites (cos-lat weighted) for wake core (right/left), outer wake and control:
  pre  = 1-7 May   (before the depression)
  post = 15-19 May (just after landfall)
  late = 1-7 June
Profiles start at 5 m (0 m level is outside the model levels and is NaN).
MLD: sigma0 +0.03 kg m-3 from 10 m; ILD: T -0.2 degC from 10 m; BLT = ILD - MLD (computed in Colab block D).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h5lite import File
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
D = DATA + "/dl"; O2 = OUTB + "/step2"; O = OUTB + "/step3"; os.makedirs(O, exist_ok=True)
G = np.load(f"{O2}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; REG = G["region"]; SIDE = G["side"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
R = {"wake": REG == 1, "wake_right": (REG == 1) & (SIDE == 1), "wake_left": (REG == 1) & (SIDE == -1), "outer": REG == 2, "control": REG == 3}
T = pd.date_range("2023-04-01", periods=153)
WIN = {"pre 1-7 May": ("2023-05-01", "2023-05-07"), "post 15-19 May": ("2023-05-15", "2023-05-19"), "1-7 June": ("2023-06-01", "2023-06-07")}


def rmean(a, m):
    w = WT * m; ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


p = File(f"{D}/phy_profiles_2023_AprAug.nc"); b = File(f"{D}/bgc_profiles_2023_AprAug.nc")
Z = p.read("depth"); assert np.allclose(Z, b.read("depth"))
F = {"T": p.read("thetao"), "S": p.read("so"), "NO3": b.read("no3"), "PO4": b.read("po4"), "Chl": b.read("chl"), "NPP": b.read("nppv")}
# potential density (sigma0) from UNESCO EOS-80 polynomial (adequate for profile shapes; gsw not installed here)
def sigma0(Tc, S):
    r0 = (999.842594 + 6.793952e-2 * Tc - 9.095290e-3 * Tc**2 + 1.001685e-4 * Tc**3 - 1.120083e-6 * Tc**4 + 6.536332e-9 * Tc**5)
    A = 8.24493e-1 - 4.0899e-3 * Tc + 7.6438e-5 * Tc**2 - 8.2467e-7 * Tc**3 + 5.3875e-9 * Tc**4
    B = -5.72466e-3 + 1.0227e-4 * Tc - 1.6546e-6 * Tc**2; C = 4.8314e-4
    return r0 + A * S + B * S**1.5 + C * S**2 - 1000
F["sigma0"] = sigma0(F["T"], F["S"])
D2 = {k: p.read(k) for k in ["mld_density", "ild_temperature", "barrier_layer_thickness", "N2_max_0_200m"]}
D2.update({k: b.read(k) for k in ["npp_int_0_200m", "npp_int_0_100m", "chl_int_0_200m", "depth_chl_max", "nitracline_depth_plus1uM"]})

# ---------- composite profiles ----------
prof = []
for rn, rm in R.items():
    for wn, (a, z) in WIN.items():
        k = (T >= a) & (T <= z)
        for v, arr in F.items():
            pr = np.nanmean(rmean(arr[k], rm), axis=0)
            for zi, val in zip(Z, pr): prof.append(dict(region=rn, window=wn, var=v, depth=zi, value=val))
PR = pd.DataFrame(prof); PR.to_csv(f"{O}/profiles_composite.csv", index=False)

# ---------- 2-D diagnostics: region time series and window means ----------
ts = pd.DataFrame({(v, rn): rmean(arr, rm) for v, arr in D2.items() for rn, rm in R.items()}, index=T)
ts.to_pickle(f"{O}/diag_region_timeseries.pkl")
summ = []
for v in D2:
    for rn in R:
        s = ts[(v, rn)]; row = dict(diagnostic=v, region=rn)
        for wn, (a, z) in WIN.items(): row[wn] = s.loc[a:z].mean()
        row["post - pre"] = row["post 15-19 May"] - row["pre 1-7 May"]; summ.append(row)
S = pd.DataFrame(summ); S.to_csv(f"{O}/diagnostics_by_window.csv", index=False)
pd.set_option("display.width", 220); print(S.round(2).to_string(index=False))

# fraction of wake cells with a barrier layer > 10 m before the storm
pre = (T >= "2023-05-01") & (T <= "2023-05-07")
blt_pre = np.nanmean(D2["barrier_layer_thickness"][pre], 0); mld_pre = np.nanmean(D2["mld_density"][pre], 0)
for rn in ["wake", "wake_right", "wake_left", "control"]:
    m = R[rn] & np.isfinite(blt_pre)
    print(f"{rn:11s}: pre-storm BLT>10 m in {100*np.mean(blt_pre[m] > 10):.0f}% of cells; median BLT {np.median(blt_pre[m]):.1f} m; median MLD {np.nanmedian(mld_pre[m]):.1f} m")

# ---------- figure 1: physical profiles ----------
reg_c = {"wake_right": "#99000d", "wake_left": "#fb6a4a", "control": "#2171b5"}
win_ls = {"pre 1-7 May": "-", "post 15-19 May": "--", "1-7 June": ":"}
fig, axs = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
for ax, v, lab in zip(axs, ["T", "S", "sigma0", "NO3"], ["Temperature (°C)", "Salinity (psu)", "σ$_0$ (kg m$^{-3}$)", "NO$_3$ (mmol m$^{-3}$)"]):
    for rn, c in reg_c.items():
        for wn, ls in win_ls.items():
            q = PR[(PR.region == rn) & (PR.window == wn) & (PR["var"] == v)].dropna()
            ax.plot(q.value, q.depth, ls=ls, color=c, lw=1.8 if ls == "-" else 1.4, label=f"{rn.replace('_', ' ')}, {wn}" if v == "T" else None)
    ax.set_xlabel(lab); ax.grid(alpha=0.3); ax.set_ylim(150, 0)
axs[0].set_ylabel("Depth (m)"); axs[0].legend(fontsize=6.5, loc="lower right")
fig.suptitle("Composite profiles: wake core right/left of track vs control, before (1–7 May), after (15–19 May) and 1–7 June 2023", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig_s3_profiles_physics.png", dpi=150)

# ---------- figure 2: biogeochemical profiles ----------
fig, axs = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
for ax, v, lab, xl in zip(axs, ["NO3", "PO4", "Chl", "NPP"], ["NO$_3$ (mmol m$^{-3}$)", "PO$_4$ (mmol m$^{-3}$)", "Chl (mg m$^{-3}$)", "NPP (mg C m$^{-3}$ d$^{-1}$)"],
                          [(0, 4), (0, 0.6), None, None]):
    for rn, c in reg_c.items():
        for wn, ls in win_ls.items():
            q = PR[(PR.region == rn) & (PR.window == wn) & (PR["var"] == v)].dropna()
            ax.plot(q.value, q.depth, ls=ls, color=c, lw=1.8 if ls == "-" else 1.4, label=f"{rn.replace('_', ' ')}, {wn}" if v == "NO3" else None)
    ax.set_xlabel(lab); ax.grid(alpha=0.3); ax.set_ylim(120, 0)
    if xl: ax.set_xlim(*xl)
axs[0].set_ylabel("Depth (m)"); axs[0].legend(fontsize=6.5, loc="lower right")
fig.suptitle("Nutrient, chlorophyll and NPP profiles (GLOBAL_MULTIYEAR_BGC_001_029). Note the surface nitrate maximum above a nitrate-poor layer at 30–60 m.", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig_s3_profiles_bgc.png", dpi=150)

# ---------- figure 3: MLD/BLT maps before & after, and time series ----------
g = sio.netcdf_file(DATA + "/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::10].copy(); go = g.variables["lon"][::10].copy(); ge = g.variables["elevation"][::10, ::10].astype(float)
tr = pd.read_csv(f"{O2}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True).loc["2023-05-09":"2023-05-14 06:00"]
post = (T >= "2023-05-15") & (T <= "2023-05-19")
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, 3)
maps = [("Barrier-layer thickness, 1–7 May (m)", blt_pre, "viridis", 0, 40), ("Mixed-layer depth, 1–7 May (m)", mld_pre, "viridis", 0, 60),
        ("MLD change: 15–19 May minus 1–7 May (m)", np.nanmean(D2["mld_density"][post], 0) - mld_pre, "RdBu_r", -40, 40)]
for j, (tt, f, cm, lo, hi) in enumerate(maps):
    ax = fig.add_subplot(gs[0, j]); pc = ax.pcolormesh(LON, LAT, np.where(G["ocean"], f, np.nan), cmap=cm, vmin=lo, vmax=hi, shading="nearest")
    ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"]); ax.contour(go, gl, ge, levels=[0], colors="0.3", linewidths=0.4)
    ax.plot(tr.lon, tr.lat, "k-", lw=1.5); ax.contour(LON, LAT, REG == 1, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
    ax.set_xlim(78, 100); ax.set_ylim(5, 25); ax.set_aspect(1 / np.cos(np.deg2rad(15))); ax.set_title(tt, fontsize=10); fig.colorbar(pc, ax=ax, shrink=0.8)
for j, (v, lab) in enumerate([("mld_density", "MLD (m)"), ("barrier_layer_thickness", "BLT (m)"), ("npp_int_0_100m", "NPP 0–100 m (mg C m$^{-2}$ d$^{-1}$)")]):
    ax = fig.add_subplot(gs[1, j])
    for rn, c in [("wake_right", "#99000d"), ("wake_left", "#fb6a4a"), ("control", "#2171b5")]:
        ax.plot(T, ts[(v, rn)], color=c, label=rn.replace("_", " "))
    ax.axvspan(pd.Timestamp("2023-05-09"), pd.Timestamp("2023-05-14 23:00"), color="k", alpha=0.1)
    ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.set_xlim(T[0], pd.Timestamp("2023-07-15"))
    if j == 0: ax.legend(fontsize=8)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"{O}/fig_s3_mld_blt.png", dpi=150)
