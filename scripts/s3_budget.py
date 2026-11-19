import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 3c - budget for the surface nitrate (and phosphate) change across the storm, per region.
Window: 9 May (depression) -> 16 May (surface nitrate minimum), 7 days.
Terms (all in mmol m-3 at the surface, region means):
  observed  : N(16 May) - N(9 May)
  mixing    : 1-D homogenisation of each cell's 9 May profile (5 m .. h) where h = that cell's
              maximum density-MLD during 9-16 May; predicted surface = depth-mean over 0..h
              (5 m value used for 0-5 m). Checked with the same calculation for SSS and SST.
  advection : sum over days of -(u dN/dx + v dN/dy) with daily surface currents and daily surface N
  rain      : -N * P / h  (dilution by ERA5 precipitation 9-16 May into the mixed layer)
  extra uptake from the storm: NPP anomaly (vs 1993-2022 climatology) summed over 9-16 May,
              converted with Redfield C:N:P = 106:16:1 (an upper bound: PISCES also takes up NH4)
  residual  : observed - (mixing + advection + rain - uptake)
Because 'mixing' uses the full observed MLD, it includes entrainment by wind mixing; upwelling
(Ekman pumping) that lifts the nitracline is NOT in the 1-D term and ends up in the residual.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings
from h5lite import File
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
D = DATA + ""; DL = f"{D}/dl"; O2 = OUTB + "/step2"; O = OUTB + "/step3"
G = np.load(f"{O2}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; REG = G["region"]; SIDE = G["side"]; OCEAN = G["ocean"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
R = {"wake": REG == 1, "wake_right": (REG == 1) & (SIDE == 1), "wake_left": (REG == 1) & (SIDE == -1), "outer": REG == 2, "control": REG == 3}
T = pd.date_range("2023-04-01", periods=153)
i0, i1 = int(np.where(T == "2023-05-09")[0][0]), int(np.where(T == "2023-05-16")[0][0])


def rmean(a, m):
    w = WT * m; ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


p = File(f"{DL}/phy_profiles_2023_AprAug.nc"); b = File(f"{DL}/bgc_profiles_2023_AprAug.nc"); Z = p.read("depth")
mld = p.read("mld_density"); h = np.nanmax(mld[i0:i1 + 1], 0)                       # (lat, lon)
prof0 = {"NO3": b.read("no3")[i0], "PO4": b.read("po4")[i0], "S": p.read("so")[i0], "T": p.read("thetao")[i0]}
surf = {"NO3": b.read("no3")[:, 1], "PO4": b.read("po4")[:, 1], "S": p.read("so")[:, 1], "T": p.read("thetao")[:, 1]}   # 5 m level


def mixed_mean(pr, h):
    # depth-mean of profile from 0 to h (pr at Z; Z[0]=0 is NaN -> use 5 m value for 0-5 m)
    zz = Z.copy(); q = pr.copy(); q[0] = q[1]
    out = np.full(h.shape, np.nan)
    for (i, j), hh in np.ndenumerate(h):
        if not np.isfinite(hh) or not np.isfinite(q[1, i, j]): continue
        hh = max(hh, 5.0); zs = np.r_[zz[zz < hh], hh]; vs = np.interp(zs, zz, q[:, i, j])
        if np.isnan(vs).any(): continue
        out[i, j] = np.trapezoid(vs, zs) / hh
    return out


mixpred = {v: mixed_mean(prof0[v], h) for v in prof0}

# advection with daily surface fields (0.5 m BGC surface; currents 0.494 m)
fb = File(f"{D}/nutrients_april-june-2023.nc"); tb = pd.to_datetime(fb.read("time"), unit="s")
kb = np.where((tb >= "2023-05-09") & (tb <= "2023-05-16"))[0]
Ns = {"NO3": np.where(OCEAN, fb.read("no3", sel=((kb[0], len(kb)), (0, 1), (0, 81), (0, 89)))[:, 0], np.nan),
      "PO4": np.where(OCEAN, fb.read("po4", sel=((kb[0], len(kb)), (0, 1), (0, 81), (0, 89)))[:, 0], np.nan)}
cu = File(f"{D}/currents_surface_2023_AprAug.nc"); tc = pd.Timestamp("1950-01-01") + pd.to_timedelta(cu.read("time"), unit="h")
kc = np.where((tc.floor("D") >= "2023-05-09") & (tc.floor("D") <= "2023-05-16"))[0]
U = cu.read("uo")[kc]; V = cu.read("vo")[kc]
dy = 6371e3 * np.deg2rad(0.25); dx = dy * np.cos(np.deg2rad(LAT))[:, None]


def advect(N):
    tot = np.zeros(N.shape[1:])
    for d in range(N.shape[0] - 1):                       # 7 daily steps, centred differences
        n = N[d]; gx = (np.roll(n, -1, 1) - np.roll(n, 1, 1)) / (2 * dx); gy = (np.roll(n, -1, 0) - np.roll(n, 1, 0)) / (2 * dy)
        tot += -(U[d] * gx + V[d] * gy) * 86400.0
    return tot


adv = {v: advect(Ns[v]) for v in Ns}
# rain dilution
e = File(f"{DL}/era5_daily_2023_AprAug.nc"); pr = e.read("precip_mm")
P = np.nansum(pr[i0:i1], 0) / 1000.0                       # m of rain 9-15 May (7 days)
rain = {v: -Ns[v][0] * P / np.maximum(h, 5.0) for v in Ns}
# storm-driven extra uptake: NPP anomaly vs climatology at the surface, 9-15 May
an_npp = np.load(f"{O2}/anom2023_npp.npy"); tbb = pd.to_datetime(np.load(f"{O2}/time2023_bgc.npy"))
kk = (tbb >= "2023-05-09") & (tbb <= "2023-05-15")
Csum = np.nansum(an_npp[kk], 0)                              # mg C m-3 over 7 days
upt = {"NO3": Csum / 12.011 * 16 / 106, "PO4": Csum / 12.011 / 106}
total_npp = np.nansum(np.where(OCEAN, fb.read("nppv", sel=((kb[0], 7), (0, 1), (0, 81), (0, 89)))[:, 0], np.nan), 0)
upt_gross = {"NO3": total_npp / 12.011 * 16 / 106, "PO4": total_npp / 12.011 / 106}

rows = []
for v in ["NO3", "PO4", "S", "T"]:
    obs = (surf[v][i1] - surf[v][i0])
    mix = mixpred[v] - surf[v][i0]
    for rn, rm in R.items():
        row = dict(variable=v, region=rn, h_mixing_m=rmean(h, rm), observed=rmean(obs, rm), mixing_1D=rmean(mix, rm))
        if v in ("NO3", "PO4"):
            row.update(advection=rmean(adv[v], rm), rain_dilution=rmean(rain[v], rm), storm_extra_uptake=-rmean(upt[v], rm),
                       gross_uptake_all_NPP=-rmean(upt_gross[v], rm))
            row["residual"] = row["observed"] - (row["mixing_1D"] + row["advection"] + row["rain_dilution"] + row["storm_extra_uptake"])
        else:
            row["residual"] = row["observed"] - row["mixing_1D"]
        rows.append(row)
B = pd.DataFrame(rows); B.to_csv(f"{O}/nutrient_budget_9_16May.csv", index=False)
pd.set_option("display.width", 240); print(B.round(4).to_string(index=False))
print("\nNote: observed uses the 5 m model level (BGC profiles file); S in psu, T in degC.")

# ---------------------------------------------------------------------------------------------
# Salinity-calibrated mixing depth: for each cell, the depth h* to which the 9 May column must be
# homogenised to reproduce the observed 16 May salinity at 5 m (rain only freshens, so h* is a
# lower bound on the effective depth of mixing + upwelling). Apply the same h* to NO3, PO4, T.
zf = np.arange(0, 201, 1.0)
def calib(S0, Sobs):
    q = S0.copy(); q[0] = q[1]
    hstar = np.full(Sobs.shape, np.nan)
    for (i, j), so in np.ndenumerate(Sobs):
        col = q[:, i, j]
        if not np.isfinite(so) or not np.isfinite(col[1]): continue
        ok = np.isfinite(col); zz = Z[ok]; cc = col[ok]
        if zz.max() < 20: continue
        f = np.interp(zf[zf <= zz.max()], zz, cc)
        cm = np.cumsum(f) / np.arange(1, f.size + 1)       # running depth-mean from 0 m
        k = np.where(cm >= so)[0]
        hstar[i, j] = zf[k[0]] if len(k) and so > cc[0] else (np.nan if so > cc[0] else 0.0)
    return hstar

def mean_to(pr, hs):
    q = pr.copy(); q[0] = q[1]; out = np.full(hs.shape, np.nan)
    for (i, j), hh in np.ndenumerate(hs):
        if not np.isfinite(hh): continue
        col = q[:, i, j]; ok = np.isfinite(col)
        if ok.sum() < 3: continue
        hh = max(hh, 1.0); zz = Z[ok]; f = np.interp(zf[zf <= hh], zz, col[ok]); out[i, j] = f.mean()
    return out

hstar = calib(prof0["S"], surf["S"][i1])
rows2 = []
for v in ["NO3", "PO4", "T", "S"]:
    pred = mean_to(prof0[v], hstar) - surf[v][i0]; obs = surf[v][i1] - surf[v][i0]
    for rn, rm in R.items():
        m = rm & np.isfinite(hstar)
        rows2.append(dict(variable=v, region=rn, cells_used=int(m.sum()), median_hstar_m=np.nanmedian(hstar[m]),
                          observed=rmean(np.where(np.isfinite(hstar), obs, np.nan), rm), predicted_by_S_calibrated_mixing=rmean(pred, m),
                          fraction_explained=rmean(pred, m) / rmean(np.where(np.isfinite(hstar), obs, np.nan), rm)))
B2 = pd.DataFrame(rows2); B2.to_csv(f"{O}/nutrient_budget_S_calibrated.csv", index=False)
print("\nSalinity-calibrated mixing (h* reproduces observed SSS rise):"); print(B2.round(4).to_string(index=False))
np.save(f"{O}/hstar.npy", hstar)
