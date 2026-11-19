import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""Shared style and helpers for the final publication figures."""
import numpy as np, pandas as pd, scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

FONT = "Liberation Sans"          # metric-compatible with Arial
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.labelweight": "bold", "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "legend.frameon": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out", "lines.linewidth": 1.2,
    "axes.grid": False, "grid.color": "#d9d9d9", "grid.linewidth": 0.4,
    "savefig.facecolor": "white", "figure.facecolor": "white", "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
FULL = 180 * MM       # double-column width
HALF = 88 * MM

# validated categorical colours (dataviz validator, all-pairs, light mode)
C_WAKE, C_OUTER, C_CTRL, C_SHELF = "#eb6834", "#1baf7a", "#2a78d6", "#6b6b6b"
C_SAT, C_MOD = "#1baf7a", "#6250d6"
C_OBS, C_GLO = "#2a78d6", "#eb6834"
C_INK, C_MUTED = "#1a1a1a", "#6b6b6b"
LAND = "#e8e5de"; COAST = "#4d4d4d"

D = DATA + ""; DL = f"{D}/dl"
O1 = OUTB + ""; O2 = f"{O1}/step2"; O3 = f"{O1}/step3"; O4 = f"{O1}/step4"
EXTENT = (78, 100, 5, 25)
ASPECT = 1 / np.cos(np.deg2rad(15))

_g = sio.netcdf_file(f"{D}/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
GL = _g.variables["lat"][::4].copy(); GO = _g.variables["lon"][::4].copy(); GE = _g.variables["elevation"][::4, ::4].astype(float)
GEO = np.load(f"{O2}/geometry.npz")
LAT, LON = GEO["lat"], GEO["lon"]
TRACK = pd.read_csv(f"{O2}/mocha_track_ibtracs.csv", index_col=0, parse_dates=True)
TR_OC = TRACK.loc["2023-05-09":"2023-05-14 06:00"]


def lon_fmt(v): return f"{int(v)}°E"
def lat_fmt(v): return f"{int(v)}°N"


def basemap(ax, extent=EXTENT, xt=(80, 85, 90, 95, 100), yt=(5, 10, 15, 20, 25), land=True, track=True, labels=(True, True)):
    if land:
        ax.contourf(GO, GL, GE, levels=[0, 1e4], colors=[LAND], zorder=3)
        ax.contour(GO, GL, GE, levels=[0], colors=COAST, linewidths=0.35, zorder=4)
    if track:
        ax.plot(TR_OC.lon, TR_OC.lat, color="k", lw=1.1, zorder=6, solid_capstyle="round")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect(ASPECT)
    ax.xaxis.set_major_locator(FixedLocator([x for x in xt if extent[0] <= x <= extent[1]]))
    ax.yaxis.set_major_locator(FixedLocator([y for y in yt if extent[2] <= y <= extent[3]]))
    ax.set_xticklabels([lon_fmt(x) for x in ax.get_xticks()] if labels[0] else [])
    ax.set_yticklabels([lat_fmt(y) for y in ax.get_yticks()] if labels[1] else [])
    ax.tick_params(length=2)


def outline(ax, mask, color, lw=0.9, ls="-", z=5):
    ax.contour(LON, LAT, mask.astype(float), levels=[0.5], colors=color, linewidths=lw, linestyles=ls, zorder=z)


def panel(ax, letter, x=-0.02, y=1.02, ha="right"):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha=ha)


def hcbar(fig, mappable, ax, label, ticks=None, extend="both", pad=0.09, frac=0.055, shrink=0.92):
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=pad, fraction=frac, shrink=shrink, extend=extend, aspect=28)
    cb.set_label(label, fontweight="bold", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    if ticks is not None: cb.set_ticks(ticks)
    return cb


def date_axis(ax, start="2023-04-01", end="2023-09-30", minor=False):
    import matplotlib.dates as mdates
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for lab in ax.get_xticklabels(): lab.set_rotation(0)


def event_band(ax, a="2023-05-09", b="2023-05-14 23:59"):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#bdbdbd", alpha=0.45, lw=0, zorder=0)


def save(fig, name, dpi):
    fig.savefig(f"{OUTB}/final/{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
