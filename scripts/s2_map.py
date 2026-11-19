import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
import numpy as np, pandas as pd, scipy.io as sio, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
O = OUTB + "/step2"; G = np.load(f"{O}/geometry.npz")
lat, lon, reg = G["lat"], G["lon"], G["region"].astype(float)
tr = pd.read_csv(f"{O}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True)
g = sio.netcdf_file(DATA + "/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
gl = g.variables["lat"][::8].copy(); go = g.variables["lon"][::8].copy(); ge = g.variables["elevation"][::8, ::8].astype(float)
def basemap(ax):
    ax.contourf(go, gl, ge, levels=[0, 9000], colors=["#d9d4c7"], zorder=0)
    ax.contour(go, gl, ge, levels=[0], colors="0.35", linewidths=0.5)
    ax.contour(go, gl, ge, levels=[-200], colors="0.5", linewidths=0.5, linestyles=":")
    ax.set_xlim(78, 100); ax.set_ylim(5, 25); ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
fig, ax = plt.subplots(figsize=(7.5, 7))
basemap(ax)
r = np.where(reg < 0, np.nan, reg)
cmap = ListedColormap(["#f2f2f2", "#d7301f", "#fc8d59", "#4575b4", "#91bfdb"])
pc = ax.pcolormesh(lon, lat, r, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest", alpha=0.85, zorder=1)
ax.contour(go, gl, ge, levels=[0], colors="0.35", linewidths=0.5, zorder=2)
oc = tr.loc["2023-05-09":"2023-05-14 06:00"]; st = tr.loc["2023-05-11":"2023-05-14 06:00"]
ax.plot(tr.lon, tr.lat, "k-", lw=1, zorder=3); ax.plot(oc.lon, oc.lat, "k--", lw=1.4, zorder=3); ax.plot(st.lon, st.lat, "k-", lw=2.4, zorder=3)
for t, rr in oc.iterrows():
    if t.hour == 0: ax.plot(rr.lon, rr.lat, "ko", ms=4, zorder=4); ax.annotate(t.strftime("%d May"), (rr.lon, rr.lat), xytext=(5, -3), textcoords="offset points", fontsize=7)
cb = fig.colorbar(pc, ax=ax, ticks=range(5), shrink=0.7)
cb.ax.set_yticklabels(["other open ocean", "wake core 0–200 km", "outer wake 200–400 km", "control: calm (<12 m/s), W Bay", "shelf <200 m"], fontsize=8)
ax.set_title("Analysis regions and Mocha track (IBTrACS v04r01)\nthick = cyclonic storm ≥34 kt (11–14 May), dashed = depression stage; dotted = 200 m isobath", fontsize=9)
fig.tight_layout(); fig.savefig(f"{O}/fig_s2_regions_map.png", dpi=150)
