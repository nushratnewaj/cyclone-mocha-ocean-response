import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""Step 2c - cold wake in observed SST (OSTIA, 0.05 deg) vs GLORYS12 (0.25 deg regridded), and
anomaly Hovmoller (replacement for manuscript Fig. 10).
Cooling per cell: min SST in [t_pass, t_pass + 5 d] minus mean SST in [t_pass - 10 d, t_pass - 3 d],
where t_pass is the time the cyclonic-storm track (>= 34 kt) passed closest to the cell."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scipy.io as sio, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h5lite import File
from s2_geometry import load_track, densify, hav
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
O = OUTB + "/step2"; D = DATA + ""
tr = load_track(); dense = densify(tr, t0="2023-05-11 00:00")
tl, tn, tt = dense.lat.values, dense.lon.values, dense.index


def geometry(lat, lon):
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    dist = np.full(LA.shape, np.inf); k = np.zeros(LA.shape, int)
    for i in range(len(tl)):
        d = hav(LA, LO, tl[i], tn[i]); b = d < dist; dist[b] = d[b]; k[b] = i
    k1 = np.clip(k + 1, 0, len(tl) - 1); k0 = np.clip(k - 1, 0, len(tl) - 1)
    hx = (tn[k1] - tn[k0]) * np.cos(np.deg2rad(tl[k])); hy = tl[k1] - tl[k0]
    cross = hx * (LA - tl[k]) - hy * (LO - tn[k]) * np.cos(np.deg2rad(LA))
    end = (k == 0) | (k == len(tl) - 1)
    return dist, k, np.where(cross < 0, 1, -1), end


def cooling(sst, days, k):
    out = np.full(k.shape, np.nan)
    for kk in np.unique(k):
        m = k == kk; t0 = tt[kk].floor("D")
        pre = (days >= t0 - pd.Timedelta(days=10)) & (days <= t0 - pd.Timedelta(days=3))
        post = (days >= t0) & (days <= t0 + pd.Timedelta(days=5))
        out[m] = (np.nanmin(sst[post][:, m], 0) - np.nanmean(sst[pre][:, m], 0))
    return out


# OSTIA
f = File(f"{D}/dl/ostia_sst_2023_15Apr_30Jun.nc")
olat = f.read("latitude"); olon = f.read("longitude"); od = pd.to_datetime(f.read("time"), unit="s", origin="1981-01-01").floor("D")
osst = f.read("analysed_sst")
dO, kO, sO, eO = geometry(olat, olon); cO = cooling(osst, od, kO)
# GLORYS (0.25 regridded, same as analysis)
G = np.load(f"{O}/geometry.npz"); glat, glon = G["lat"], G["lon"]
gsst = np.load(f"{O}/sst2023_grid.npy"); gd = pd.to_datetime(np.load(f"{O}/time2023_phy.npy"))
dG, kG, sG, eG = geometry(glat, glon); cG = cooling(gsst, gd, kG)
ocean_open = G["ocean"] & ~G["shelf"]

rows = []
for name, c, d, s, e, extra in [("OSTIA (obs, 0.05°)", cO, dO, sO, eO, np.isfinite(cO)), ("GLORYS12 (0.25°)", cG, dG, sG, eG, ocean_open)]:
    for lo, hi in [(0, 100), (0, 200), (200, 400)]:
        m = (d >= lo) & (d < hi) & ~e & extra & np.isfinite(c)
        rows.append(dict(product=name, band_km=f"{lo}-{hi}", mean_cooling=np.nanmean(c[m]), p05_cooling=np.nanpercentile(c[m], 5),
                         max_cooling=np.nanmin(c[m]), right_mean=np.nanmean(c[m & (s == 1)]), left_mean=np.nanmean(c[m & (s == -1)]), n=int(m.sum())))
W = pd.DataFrame(rows); W.to_csv(f"{O}/step2_cold_wake.csv", index=False)
print(W.round(2).to_string(index=False))

# along-track profile (6-hourly fixes): mean cooling within 100 km, and strongest cooling within 200 km
fixes = tr.loc["2023-05-11":"2023-05-14 06:00"]
prof = []
for t, r in fixes.iterrows():
    for name, c, lat, lon, e in [("OSTIA", cO, olat, olon, eO), ("GLORYS", cG, glat, glon, eG)]:
        LA, LO = np.meshgrid(lat, lon, indexing="ij"); dd = hav(LA, LO, r.lat, r.lon)
        m = np.isfinite(c) & ~e
        prof.append(dict(time=t, product=name, wind_kt=r.wmo_wind_kt if np.isfinite(r.wmo_wind_kt) else r.usa_wind_kt,
                         mean_100km=np.nanmean(c[m & (dd <= 100)]), min_200km=np.nanmin(c[m & (dd <= 200)]) if (m & (dd <= 200)).any() else np.nan))
Pf = pd.DataFrame(prof).pivot(index="time", columns="product")
Pf.to_csv(f"{O}/step2_cold_wake_alongtrack.csv")
print(Pf.round(2).to_string())

# figure: cooling maps + along-track
g = sio.netcdf_file(f"{D}/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::10].copy(); go = g.variables["lon"][::10].copy(); ge = g.variables["elevation"][::10, ::10].astype(float)
fig = plt.figure(figsize=(15, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
for j, (name, c, lat, lon) in enumerate([("OSTIA observed SST", cO, olat, olon), ("GLORYS12 SST (0.25°)", cG, glat, glon)]):
    ax = fig.add_subplot(gs[j]); pc = ax.pcolormesh(lon, lat, c, cmap="Blues_r", vmin=-3, vmax=0, shading="nearest")
    ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"]); ax.contour(go, gl, ge, levels=[0], colors="0.3", linewidths=0.4)
    oc = tr.loc["2023-05-09":"2023-05-14 06:00"]; ax.plot(oc.lon, oc.lat, "k-", lw=1.4)
    for t, r in fixes.iterrows():
        if t.hour == 0: ax.plot(r.lon, r.lat, "ko", ms=3); ax.annotate(t.strftime("%d May"), (r.lon, r.lat), xytext=(4, -2), textcoords="offset points", fontsize=7)
    ax.set_xlim(82, 96); ax.set_ylim(8, 22); ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    ax.set_title(f"{name}: post-passage cooling (°C)", fontsize=10); fig.colorbar(pc, ax=ax, shrink=0.8)
ax = fig.add_subplot(gs[2])
for name, col in [("OSTIA", "#08519c"), ("GLORYS", "#d7301f")]:
    ax.plot(Pf.index, Pf[("mean_100km", name)], "-o", color=col, ms=3, label=f"{name}: mean within 100 km")
    ax.plot(Pf.index, Pf[("min_200km", name)], "--", color=col, label=f"{name}: strongest within 200 km")
ax2 = ax.twinx(); ax2.plot(Pf.index, Pf[("wind_kt", "OSTIA")], color="0.5", lw=1); ax2.set_ylabel("max wind (kt)", color="0.5")
ax.axhline(-2, color="k", ls=":", lw=0.8); ax.text(Pf.index[0], -2.05, "  Puri (2026): ~2 °C along track", fontsize=7, va="top")
ax.set_ylabel("SST change (°C)"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3); ax.set_title("Cooling along the track (6-hourly fixes)", fontsize=10)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f"{O}/fig_s2_cold_wake.png", dpi=150)

# ---------- anomaly Hovmoller (open ocean, 10-20N) ----------
tb = pd.to_datetime(np.load(f"{O}/time2023_bgc.npy")); tp = gd
band = ocean_open & (glat[:, None] >= 10) & (glat[:, None] <= 20)
w = np.cos(np.deg2rad(glat))[:, None] * band
fig, axs = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
track_t = tr.loc["2023-05-09":"2023-05-14 06:00"]
for ax, (v, lab, L, cmap) in zip(axs, [("sst", "SST anomaly (°C)", 1.5, "RdBu_r"), ("no3", "NO$_3$ anomaly (mmol m$^{-3}$)", 1.0, "PuOr_r"),
                                         ("po4", "PO$_4$ anomaly (mmol m$^{-3}$)", 0.02, "PuOr_r"), ("chl", "Chl anomaly (mg m$^{-3}$)", 0.08, "PiYG")]):
    a = np.load(f"{O}/anom2023_{v}.npy"); t = tp if v == "sst" else tb
    sel = t <= pd.Timestamp("2023-07-31"); a = a[sel]; t = t[sel]
    num = np.nansum(np.where(np.isfinite(a), a, 0) * w, axis=1); den = np.nansum(np.where(np.isfinite(a), 1, 0) * w, axis=1)
    H = num / np.where(den > 0, den, np.nan)
    pc = ax.pcolormesh(glon, t, H, cmap=cmap, vmin=-L, vmax=L, shading="nearest")
    ax.plot(track_t.lon, track_t.index, "k-", lw=2); ax.set_xlim(80, 98)
    ax.set_title(lab + "\nopen ocean 10–20°N", fontsize=9); ax.set_xlabel("Longitude (°E)"); fig.colorbar(pc, ax=ax, shrink=0.8, orientation="horizontal", pad=0.08)
axs[0].invert_yaxis(); axs[0].set_ylabel("2023")
fig.suptitle("Anomaly Hovmöller vs 1993–2022 climatology, shelf excluded; black line = Mocha track longitude (9–14 May)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig_s2_hovmoller_anom.png", dpi=150)
