import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""Final publication figures. Usage: python final_figs.py <dpi> [fig ids...]"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from final_style import *
from h5lite import File
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import RegularGridInterpolator
np.seterr(all="ignore")
os.makedirs(OUTB + "/final", exist_ok=True)
REG = GEO["region"]; SIDE = GEO["side"]; OCEAN = GEO["ocean"]; SHELF = GEO["shelf"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))


def rmean(a, m):
    w = WT * m; ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


# IMD intensity categories (kt) and a single-hue sequential ramp
CATS = [("D", 17, 27), ("DD", 28, 33), ("CS", 34, 47), ("SCS", 48, 63), ("VSCS", 64, 89), ("ESCS", 90, 119), ("SuCS", 120, 400)]
CAT_COLS = ["#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#3f007d"]


def cat_of(w):
    for i, (n, lo, hi) in enumerate(CATS):
        if lo <= w <= hi: return i
    return -1


# =====================================================================================
def fig01(dpi):
    """Study area, analysis regions and Mocha track; intensity time series."""
    fig = plt.figure(figsize=(FULL, 105 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    depth = np.where(GE < 0, -GE, np.nan)
    bl = ax.contourf(GO, GL, depth, levels=[0, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000], cmap="Blues", alpha=0.55, zorder=1, extend="max")
    basemap(ax, track=False)
    ax.contour(GO, GL, GE, levels=[-200], colors="#4d4d4d", linewidths=0.5, linestyles="--", zorder=4)
    outline(ax, REG == 2, C_OUTER, lw=1.1, z=5); outline(ax, REG == 3, C_CTRL, lw=1.1, z=5); outline(ax, REG == 1, C_WAKE, lw=1.4, z=6)
    ax.plot(TRACK.lon, TRACK.lat, color="k", lw=0.8, zorder=6)
    t6 = TRACK[TRACK.index.hour % 6 == 0]
    w = t6.wmo_wind_kt.fillna(t6.usa_wind_kt)
    for (t, r), wv in zip(t6.iterrows(), w):
        ci = cat_of(wv)
        if ci < 0 or r.dist2land_km == 0: mk, fc = "o", "white"
        else: mk, fc = "o", CAT_COLS[ci]
        ax.plot(r.lon, r.lat, marker=mk, ms=4.2, mfc=fc, mec="k", mew=0.45, zorder=7)
        if t.hour == 0 and t <= pd.Timestamp("2023-05-14"):
            ax.annotate(t.strftime("%-d May"), (r.lon, r.lat), xytext=(-5, 1), textcoords="offset points", fontsize=6.3, ha="right", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.plot(92.90, 20.15, marker="*", ms=9, mfc="#ffd700", mec="k", mew=0.6, zorder=9)
    ax.annotate("Landfall near Sittwe\n14 May 2023", (92.9, 20.15), xytext=(94.0, 18.0), fontsize=6.3, zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k"), bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for txt, x, y, kw in [("INDIA", 79.4, 17.0, {}), ("BANGLADESH", 90.6, 24.2, {}), ("MYANMAR", 96.6, 21.6, {}), ("SRI\nLANKA", 80.7, 7.6, {"fontsize": 5.5}),
                          ("Bay of Bengal", 84.2, 9.6, {"style": "italic", "color": "#08306b", "fontsize": 7.5}),
                          ("Andaman\nSea", 96.3, 12.2, {"style": "italic", "color": "#08306b", "fontsize": 6.5})]:
        ax.text(x, y, txt, ha="center", va="center", zorder=8, fontsize=kw.pop("fontsize", 6.5), fontweight="bold" if txt.isupper() or "\n" in txt and txt.split("\n")[0].isupper() else "normal", **kw)
    # scale bar (500 km at 6.2N)
    x0, y0 = 94.6, 5.85; dlon = 500 / (111.32 * np.cos(np.deg2rad(y0)))
    ax.plot([x0, x0 + dlon], [y0, y0], color="k", lw=1.6, zorder=8, solid_capstyle="butt"); ax.text(x0 + dlon / 2, y0 + 0.3, "500 km", ha="center", fontsize=6.3, zorder=8)
    cb = fig.colorbar(bl, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, aspect=30, shrink=0.9)
    cb.set_label("Water depth (m, GEBCO 2025)", fontweight="bold", fontsize=7.5); cb.ax.tick_params(labelsize=6.2, length=2); cb.outline.set_linewidth(0.5)
    handles = [mlines.Line2D([], [], color=C_WAKE, lw=1.3, label="Wake core (0–200 km)"), mlines.Line2D([], [], color=C_OUTER, lw=1.1, label="Outer wake (200–400 km)"),
               mlines.Line2D([], [], color=C_CTRL, lw=1.1, label="Control region"), mlines.Line2D([], [], color="#4d4d4d", lw=0.6, ls=(0, (3, 2)), label="200 m isobath (shelf edge)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.995), fontsize=6, handlelength=1.8, framealpha=0.9, frameon=True, edgecolor="none")
    panel(ax, "a", x=0.0, ha="left")
    # (b) intensity
    bx = fig.add_subplot(gs[1])
    tw = TRACK.copy(); tw = tw[tw.index <= pd.Timestamp("2023-05-15 00:00")]
    for i, (n, lo, hi) in enumerate(CATS[:6]):
        bx.axhspan(lo, min(hi + 1, 150), color=CAT_COLS[i], alpha=0.55, lw=0, zorder=0)
        bx.text(pd.Timestamp("2023-05-15 02:00"), (lo + min(hi, 140)) / 2, n, fontsize=6, va="center", ha="left", color=C_INK, clip_on=False)
    bx.plot(tw.index, tw.wmo_wind_kt, color=C_INK, lw=1.4, marker="o", ms=2.2, label="IMD (3-min mean), IBTrACS")
    bx.plot(tw.index, tw.usa_wind_kt, color=C_MUTED, lw=1.0, ls=(0, (3, 1.5)), label="JTWC (1-min mean), IBTrACS")
    bx.axvline(pd.Timestamp("2023-05-14 07:00"), color="k", lw=0.7, ls=":"); bx.text(pd.Timestamp("2023-05-14 05:00"), 8, "landfall", rotation=90, fontsize=6, ha="right", va="bottom")
    bx.set_ylim(0, 150); bx.set_xlim(pd.Timestamp("2023-05-08 12:00"), pd.Timestamp("2023-05-15 00:00"))
    import matplotlib.dates as mdates
    bx.xaxis.set_major_locator(mdates.DayLocator()); bx.xaxis.set_major_formatter(mdates.DateFormatter("%-d May"))
    plt.setp(bx.get_xticklabels(), rotation=45, ha="right")
    bx.set_ylabel("Maximum sustained wind (kt)"); bx.set_xlabel("Date (2023, UTC)")
    bx.grid(axis="y", alpha=0.6); bx.legend(loc="upper left", fontsize=6.2)
    bx.set_box_aspect(1.25)
    panel(bx, "b", x=0.0, ha="left")
    save(fig, "Figure_01_study_area_track", dpi)


# =====================================================================================
def fig02(dpi):
    """ERA5 atmospheric forcing (replaces TerraClimate)."""
    e = File(f"{DL}/era5_daily_2023_AprAug.nc"); te = pd.date_range("2023-04-01", periods=153)
    k = (te >= "2023-05-09") & (te <= "2023-05-15")
    wmax = np.nanmax(e.read("wspd_max")[k], 0); rain = np.nansum(e.read("precip_mm")[k], 0)
    ek = np.load(f"{O3}/ekman_cum_9_15May.npy")
    fig = plt.figure(figsize=(FULL, 150 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.8], hspace=0.42, wspace=0.12)
    gsb = gs[1, :].subgridspec(1, 3, wspace=0.42)
    specs = [(wmax, "YlOrRd", 0, 30, "Maximum daily 10 m wind (m s$^{-1}$)", np.arange(0, 31, 5), "max"),
             (rain, "Blues", 0, 400, "Accumulated rainfall (mm)", np.arange(0, 401, 100), "max"),
             (np.where(OCEAN, ek, np.nan), "RdBu_r", -40, 40, "Ekman displacement (m; + upwelling)", np.arange(-40, 41, 20), "both")]
    for j, (fld, cm, lo, hi, lab, tk, ext) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j])
        pc = ax.pcolormesh(LON, LAT, fld, cmap=cm, vmin=lo, vmax=hi, shading="nearest", zorder=1, rasterized=True)
        basemap(ax, labels=(True, j == 0)); outline(ax, REG == 1, "k", lw=0.6, ls="--")
        hcbar(fig, pc, ax, lab, ticks=tk, extend=ext, pad=0.10)
        panel(ax, "abc"[j], x=0.0, ha="left")
    titles = ["(a)", "(b)", "(c)"]
    E = {}
    for v in ["wspd", "precip_mm"]:
        a = e.read(v); E[v] = {r: pd.Series(rmean(a, REG == c), index=te) for r, c in [("wake", 1), ("control", 3)]}
    for j, (v, lab) in enumerate([("wspd", "Daily mean 10 m wind (m s$^{-1}$)"), ("precip_mm", "Daily rainfall (mm d$^{-1}$)")]):
        ax = fig.add_subplot(gsb[0, j])
        event_band(ax)
        ax.plot(E[v]["wake"].index, E[v]["wake"], color=C_WAKE, lw=1.3, label="Wake core")
        ax.plot(E[v]["control"].index, E[v]["control"], color=C_CTRL, lw=1.1, label="Control")
        ax.set_ylabel(lab); date_axis(ax, "2023-04-15", "2023-06-15"); ax.grid(axis="y", alpha=0.6)
        if j == 0: ax.legend(loc="upper left")
        panel(ax, "de"[j], x=0.0, ha="left")
    # (f) along-track maximum wind (ERA5 vs best track) as a sanity check of ERA5 intensity
    ax = fig.add_subplot(gsb[0, 2])
    ws = e.read("wspd_max"); vals = []
    for t, r in TR_OC.iterrows():
        d = te == t.floor("D"); i = np.argmin(abs(LAT - r.lat)); jx = np.argmin(abs(LON - r.lon))
        vals.append(np.nanmax(ws[d][0, max(i - 4, 0):i + 5, max(jx - 4, 0):jx + 5]))
    bt = TR_OC.wmo_wind_kt.dropna(); ax.plot(bt.index, bt * 0.5144, color=C_INK, lw=1.3, label="IMD best track")
    ax.plot(TR_OC.index, vals, color=C_WAKE, lw=1.2, ls=(0, (3, 1.5)), label="ERA5 (±1° of centre)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d"))
    ax.set_xlabel("Day of May 2023"); ax.set_ylabel("Maximum wind (m s$^{-1}$)"); ax.grid(axis="y", alpha=0.6); ax.legend(loc="upper left")
    panel(ax, "f", x=0.0, ha="left")
    save(fig, "Figure_02_ERA5_forcing", dpi)


# =====================================================================================
def _cooling():
    from s2_geometry import load_track, densify, hav
    dense = densify(load_track(), t0="2023-05-11 00:00"); tl, tn, tt = dense.lat.values, dense.lon.values, dense.index
    def geom(lat, lon):
        LA, LO = np.meshgrid(lat, lon, indexing="ij"); dist = np.full(LA.shape, np.inf); k = np.zeros(LA.shape, int)
        for i in range(len(tl)):
            d = hav(LA, LO, tl[i], tn[i]); b = d < dist; dist[b] = d[b]; k[b] = i
        end = (k == 0) | (k == len(tl) - 1); return dist, k, end
    def cool(sst, days, k):
        out = np.full(k.shape, np.nan)
        for kk in np.unique(k):
            m = k == kk; t0 = tt[kk].floor("D")
            pre = (days >= t0 - pd.Timedelta(days=10)) & (days <= t0 - pd.Timedelta(days=3)); post = (days >= t0) & (days <= t0 + pd.Timedelta(days=5))
            out[m] = np.nanmin(sst[post][:, m], 0) - np.nanmean(sst[pre][:, m], 0)
        return out
    f = File(f"{DL}/ostia_sst_2023_15Apr_30Jun.nc"); ol, oo = f.read("latitude"), f.read("longitude")
    od = pd.to_datetime(f.read("time"), unit="s", origin="1981-01-01").floor("D"); dO, kO, eO = geom(ol, oo); cO = cool(f.read("analysed_sst"), od, kO)
    gs_ = np.load(f"{O2}/sst2023_grid.npy"); gd = pd.to_datetime(np.load(f"{O2}/time2023_phy.npy")); dG, kG, eG = geom(LAT, LON); cG = cool(gs_, gd, kG)
    return (ol, oo, cO), (LAT, LON, np.where(OCEAN, cG, np.nan))


def fig03(dpi):
    (ol, oo, cO), (gl, go, cG) = _cooling()
    A = pd.read_csv(f"{O2}/step2_cold_wake_alongtrack.csv", header=[0, 1], index_col=0, parse_dates=True)
    fig = plt.figure(figsize=(FULL, 82 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.30)
    ext = (82, 96, 8, 22)
    for j, (lab, la, lo, c) in enumerate([("OSTIA (observed, 0.05°)", ol, oo, cO), ("GLORYS12 (reanalysis, 0.25°)", gl, go, cG)]):
        ax = fig.add_subplot(gs[j])
        pc = ax.pcolormesh(lo, la, c, cmap="Blues_r", vmin=-3.5, vmax=0, shading="nearest", rasterized=True)
        basemap(ax, extent=ext, xt=(84, 88, 92, 96), yt=(8, 12, 16, 20), labels=(True, j == 0))
        for t, r in TR_OC.iterrows():
            if t.hour == 0 and t >= pd.Timestamp("2023-05-11"):
                ax.plot(r.lon, r.lat, "o", ms=2.6, mfc="white", mec="k", mew=0.5, zorder=7)
                ax.annotate(t.strftime("%-d May"), (r.lon, r.lat), xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=5.8, zorder=8,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
        ax.text(0.98, 0.02, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        panel(ax, "ab"[j], x=0.0, ha="left", y=1.01)
        if j == 1:
            cb = fig.colorbar(pc, ax=fig.axes[:2], orientation="horizontal", pad=0.12, fraction=0.05, aspect=40, shrink=0.9, extend="min")
            cb.set_label("SST change after storm passage (°C)", fontweight="bold", fontsize=7.5); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.5)
    ax = fig.add_subplot(gs[2])
    for prod, col, nm in [("OSTIA", C_OBS, "OSTIA"), ("GLORYS", C_GLO, "GLORYS12")]:
        ax.plot(A.index, A[("mean_100km", prod)], color=col, lw=1.4, marker="o", ms=2.2, label=f"{nm}: mean within 100 km")
        ax.plot(A.index, A[("min_200km", prod)], color=col, lw=1.0, ls=(0, (3, 1.5)), label=f"{nm}: strongest within 200 km")
    ax.axhline(-2, color=C_MUTED, lw=0.7, ls=":"); ax.text(A.index[0], -1.93, "~2 °C along track (Puri, 2026)", fontsize=6, ha="left", va="bottom", color=C_MUTED)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d May"))
    ax.set_ylabel("SST change (°C)"); ax.set_xlabel("Time of storm passage (2023, UTC)"); ax.set_ylim(-5, 0)
    ax.grid(axis="y", alpha=0.6); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_box_aspect(0.95)
    panel(ax, "c", x=0.0, ha="left", y=1.01)
    save(fig, "Figure_03_cold_wake", dpi)


# =====================================================================================
P = pd.read_pickle(f"{O2}/pentad_region_anoms.pkl")
DD = pd.read_pickle(f"{O2}/daily_region_series.pkl")
QUIET = [1993, 1995, 2000, 2001, 2005, 2012, 2014, 2015, 2025]
VLAB = {"sst": ("SST", "°C"), "sss": ("SSS", "psu"), "no3": ("NO$_3$", "mmol m$^{-3}$"), "po4": ("PO$_4$", "mmol m$^{-3}$"),
        "chl": ("Chl", "mg m$^{-3}$"), "npp": ("NPP", "mg C m$^{-3}$ d$^{-1}$")}


def fig04(dpi):
    """2023 among 33 seasons."""
    items = [("sst", "w"), ("sss", "w"), ("no3", "w"), ("po4", "w"), ("chl", "w"), ("npp", "w"), ("chl", "j"), ("npp", "j")]
    fig, axs = plt.subplots(2, 4, figsize=(FULL, 112 * MM), gridspec_kw=dict(wspace=0.62, hspace=0.45))
    rng = np.random.default_rng(3)
    for k, (ax, (v, kind)) in enumerate(zip(axs.flat, items)):
        if kind == "w":
            W, C = P[(v, "wake")], P[(v, "control")]; s = (W[9] - W[6]) - (C[9] - C[6])
            ttl = f"{VLAB[v][0]}, wake excess"
        else:
            s = P[(v, "open_all")][[17, 18]].mean(axis=1); ttl = f"{VLAB[v][0]}, late June"
        ref = s.loc[1993:2022]; mu, sd = ref.mean(), ref.std(ddof=1); z = (s.loc[2023] - mu) / sd
        sign = np.sign(s.loc[2023] - mu); rank = int((s * sign >= s.loc[2023] * sign).sum())
        x = rng.uniform(-0.22, 0.22, len(s))
        ax.axhspan(mu - sd, mu + sd, color="#ececec", zorder=0, lw=0); ax.axhline(0, color=C_MUTED, lw=0.5, zorder=1)
        for i, (y, val) in enumerate(s.items()):
            if y == 2023: continue
            c = C_CTRL if y in QUIET else "#a6a6a6"
            ax.scatter(x[i], val, s=14, c=c, edgecolor="white", lw=0.4, zorder=2)
        i23 = list(s.index).index(2023)
        ax.scatter(0, s.loc[2023], s=90, marker="*", c=C_WAKE, edgecolor="k", lw=0.5, zorder=4)
        ax.annotate("2023", (0, s.loc[2023]), xytext=(7, 0), textcoords="offset points", fontsize=6.5, fontweight="bold", va="center")
        others = s.drop(2023); top = others.reindex(others.abs().sort_values(ascending=False).index[:2])
        for y, val in top.items():
            i = list(s.index).index(y); ax.annotate(str(y), (x[i], val), xytext=(5, 0), textcoords="offset points", fontsize=5.8, va="center", color=C_MUTED)
        ax.set_xlim(-0.5, 0.75); ax.set_xticks([])
        ax.set_ylabel(VLAB[v][1]); ax.set_title(f"({'abcdefgh'[k]}) {ttl}", fontsize=7.2, pad=3, loc="left")
        below = s.loc[2023] < mu
        ax.text(0.5, -0.03, f"2023: z = {z:+.1f}, rank {rank}/{len(s)}", transform=ax.transAxes, fontsize=6.3, ha="center", va="top", fontweight="bold")
    h = [mlines.Line2D([], [], marker="*", ls="", ms=8, mfc=C_WAKE, mec="k", label="2023 (Mocha)"),
         mlines.Line2D([], [], marker="o", ls="", ms=4.5, mfc=C_CTRL, mec="white", label="year without a ≥34 kt storm in Apr–Jun"),
         mlines.Line2D([], [], marker="o", ls="", ms=4.5, mfc="#a6a6a6", mec="white", label="year with a ≥34 kt storm in Apr–Jun"),
         mpatches.Patch(color="#ececec", label="±1 SD of 1993–2022")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    save(fig, "Figure_04_2023_vs_33_seasons", dpi)


def fig05(dpi):
    """Daily 2023 anomalies vs climatology by region."""
    days = pd.date_range("2023-04-01", "2023-09-30")
    fig, axs = plt.subplots(3, 2, figsize=(FULL, 150 * MM), sharex=True, gridspec_kw=dict(hspace=0.18, wspace=0.28))
    for k, (ax, v) in enumerate(zip(axs.flat, ["sst", "sss", "no3", "po4", "chl", "npp"])):
        sdp = P[(v, "wake")].loc[1993:2022].std(ddof=1).values; sd = pd.Series(sdp[np.minimum((days - days[0]).days // 5, 35)], index=days)
        ax.fill_between(days, -2 * sd, 2 * sd, color="#f0f0f0", lw=0, step="mid", zorder=0)
        ax.fill_between(days, -sd, sd, color="#dcdcdc", lw=0, step="mid", zorder=0)
        event_band(ax); ax.axhline(0, color=C_MUTED, lw=0.5)
        for r, c, lw in [("control", C_CTRL, 1.0), ("outer", C_OUTER, 1.0), ("wake", C_WAKE, 1.5)]:
            s = DD[(2023, v, r, "anom")].dropna(); ax.plot(s.index, s.values, color=c, lw=lw, zorder=3)
        if v in ("sst", "sss"):
            pw = P[(v, "wake")].loc[2023].values; pdays = days[0] + pd.to_timedelta(np.arange(36) * 5 + 2, "D")
            ax.plot(pdays[18:], pw[18:], color=C_WAKE, lw=1.0, ls=(0, (2, 1.2)), marker="o", ms=1.8, zorder=3)
        ax.set_ylabel(f"{VLAB[v][0]} anomaly ({VLAB[v][1]})")
        date_axis(ax); panel(ax, "abcdef"[k], x=0.0, ha="left", y=1.0)
    h = [mlines.Line2D([], [], color=C_WAKE, lw=1.5, label="Wake core (0–200 km)"), mlines.Line2D([], [], color=C_OUTER, lw=1.0, label="Outer wake (200–400 km)"),
         mlines.Line2D([], [], color=C_CTRL, lw=1.0, label="Control"), mlines.Line2D([], [], color=C_WAKE, lw=1.0, ls=(0, (2, 1.2)), marker="o", ms=2, label="Wake core, 5-day means (Jul–Sep)"),
         mpatches.Patch(color="#dcdcdc", label="±1 SD"), mpatches.Patch(color="#f0f0f0", label="±2 SD (1993–2022, wake core)"), mpatches.Patch(color="#bdbdbd", alpha=0.45, label="9–14 May")]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.975), fontsize=6.4)
    save(fig, "Figure_05_anomaly_timeseries", dpi)


MAPV = [("sst", "RdBu_r", 2.0, [-2, -1, 0, 1, 2]), ("sss", "BrBG_r", 1.0, [-1, -0.5, 0, 0.5, 1]), ("no3", "PuOr_r", 1.5, [-1.5, -0.75, 0, 0.75, 1.5]),
        ("po4", "PuOr_r", 0.03, [-0.03, -0.015, 0, 0.015, 0.03]), ("chl", "BrBG", 0.15, [-0.15, -0.075, 0, 0.075, 0.15]), ("npp", "BrBG", 6, [-6, -3, 0, 3, 6])]


def fig06(dpi):
    """Maps of the change in anomaly across the storm."""
    tb = pd.to_datetime(np.load(f"{O2}/time2023_bgc.npy")); tp = pd.to_datetime(np.load(f"{O2}/time2023_phy.npy"))
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 150 * MM), gridspec_kw=dict(wspace=0.10, hspace=0.12))
    for k, (ax, (v, cm, L, tk)) in enumerate(zip(axs.flat, MAPV)):
        a = np.load(f"{O2}/anom2023_{v}.npy"); t = tp if v in ("sst", "sss") else tb
        d = np.nanmean(a[(t >= "2023-05-15") & (t <= "2023-05-19")], 0) - np.nanmean(a[(t >= "2023-05-01") & (t <= "2023-05-07")], 0)
        pc = ax.pcolormesh(LON, LAT, np.where(OCEAN, d, np.nan), cmap=cm, vmin=-L, vmax=L, shading="nearest", rasterized=True)
        basemap(ax, labels=(k >= 3, k % 3 == 0)); outline(ax, REG == 1, "k", lw=0.6, ls="--"); outline(ax, REG == 3, C_CTRL, lw=0.8)
        hcbar(fig, pc, ax, f"Δ{VLAB[v][0]} anomaly ({VLAB[v][1]})", ticks=tk, pad=0.06 if k < 3 else 0.10)
        panel(ax, "abcdef"[k], x=0.0, ha="left", y=1.01)
    save(fig, "Figure_06_change_maps", dpi)


def fig07(dpi):
    """Anomaly Hovmoller, open ocean 10-20N."""
    tb = pd.to_datetime(np.load(f"{O2}/time2023_bgc.npy")); tp = pd.to_datetime(np.load(f"{O2}/time2023_phy.npy"))
    band = OCEAN & ~SHELF & (LAT[:, None] >= 10) & (LAT[:, None] <= 20); w = np.cos(np.deg2rad(LAT))[:, None] * band
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 95 * MM), sharey=True, gridspec_kw=dict(wspace=0.10))
    import matplotlib.dates as mdates
    for k, (ax, (v, cm, L, tk)) in enumerate(zip(axs, [("sst", "RdBu_r", 1.5, [-1.5, 0, 1.5]), ("no3", "PuOr_r", 1.0, [-1, 0, 1]),
                                                        ("po4", "PuOr_r", 0.02, [-0.02, 0, 0.02]), ("chl", "BrBG", 0.08, [-0.08, 0, 0.08])])):
        a = np.load(f"{O2}/anom2023_{v}.npy"); t = tp if v == "sst" else tb; sel = t <= pd.Timestamp("2023-07-31"); a = a[sel]; t = t[sel]
        num = np.nansum(np.where(np.isfinite(a), a, 0) * w, 1); den = np.nansum(np.where(np.isfinite(a), 1, 0) * w, 1)
        H = num / np.where(den > 0, den, np.nan)
        pc = ax.pcolormesh(LON, t, H, cmap=cm, vmin=-L, vmax=L, shading="nearest", rasterized=True)
        ax.plot(TR_OC.lon, TR_OC.index, color="k", lw=1.4)
        ax.set_xlim(80, 98); ax.set_ylim(pd.Timestamp("2023-07-31"), pd.Timestamp("2023-04-01"))
        ax.xaxis.set_major_locator(FixedLocator([80, 85, 90, 95])); ax.set_xticklabels([lon_fmt(x) for x in [80, 85, 90, 95]])
        ax.yaxis.set_major_locator(mdates.MonthLocator()); ax.yaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
        ax.set_xlabel("Longitude")
        if k == 0: ax.set_ylabel("Date (2023)")
        hcbar(fig, pc, ax, f"{VLAB[v][0]} anomaly ({VLAB[v][1]})", ticks=tk, pad=0.13, frac=0.045)
        panel(ax, "abcd"[k], x=0.0, ha="left", y=1.01)
    save(fig, "Figure_07_hovmoller", dpi)


def fig08(dpi):
    """Satellite vs model chlorophyll."""
    f = File(f"{DL}/satellite_chl_2023_0p25.nc"); ts_ = (pd.Timestamp("1900-01-01") + pd.to_timedelta(f.read("time"), unit="D")).floor("D"); sat = f.read("chl_sat")
    fb = File(f"{D}/nutrients_april-june-2023.nc"); tb = pd.to_datetime(fb.read("time"), unit="s"); kb = np.where((tb >= ts_[0]) & (tb <= ts_[-1]))[0]
    mod = fb.read("chl", sel=((kb[0], len(kb)), (0, 1), (0, 81), (0, 89)))[:, 0]
    both = OCEAN[None] & np.isfinite(sat) & np.isfinite(mod); sat = np.where(both, sat, np.nan); mod = np.where(both, mod, np.nan)
    ch = lambda a: np.nanmean(a[(ts_ >= "2023-05-15") & (ts_ <= "2023-05-19")], 0) - np.nanmean(a[(ts_ >= "2023-05-01") & (ts_ <= "2023-05-07")], 0)
    dS, dM = ch(sat), ch(mod); opn = OCEAN & ~SHELF; m = opn & np.isfinite(dS) & np.isfinite(dM); near = m & (GEO["dist_all_km"] < 600)
    S = pd.read_csv(f"{O3}/satellite_vs_model_chl_regions.csv", header=[0, 1], index_col=0, parse_dates=True)
    fig = plt.figure(figsize=(FULL, 140 * MM)); gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.75], hspace=0.40, wspace=0.34)
    for j, (lab, d) in enumerate([("Satellite (Copernicus-GlobColour L4)", dS), ("Model (BGC 001_029)", dM)]):
        ax = fig.add_subplot(gs[0, j]); pc = ax.pcolormesh(LON, LAT, np.where(OCEAN, d, np.nan), cmap="BrBG", vmin=-0.3, vmax=0.3, shading="nearest", rasterized=True)
        basemap(ax, labels=(True, j == 0)); outline(ax, REG == 1, "k", lw=0.6, ls="--"); outline(ax, REG == 3, C_CTRL, lw=0.8)
        ax.text(0.98, 0.02, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, fontweight="bold", zorder=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        panel(ax, "ab"[j], x=0.0, ha="left", y=1.01)
    cb = fig.colorbar(pc, ax=fig.axes[:2], orientation="horizontal", pad=0.10, fraction=0.05, aspect=40, shrink=0.85, extend="both")
    cb.set_label("Surface Chl change, 15–19 May minus 1–7 May (mg m$^{-3}$)", fontweight="bold", fontsize=7.5); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.5)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(dS[m & ~near], dM[m & ~near], s=3, c="#bdbdbd", lw=0, label=">600 km from track", rasterized=True)
    ax.scatter(dS[near], dM[near], s=4, c=C_WAKE, lw=0, label="≤600 km from track", rasterized=True)
    ax.plot([-0.2, 0.8], [-0.2, 0.8], color=C_INK, lw=0.6, ls="--"); ax.axhline(0, color=C_MUTED, lw=0.4); ax.axvline(0, color=C_MUTED, lw=0.4)
    r_all = np.corrcoef(dS[m], dM[m])[0, 1]
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-0.2, 0.4); ax.set_xlabel("Satellite Chl change (mg m$^{-3}$)"); ax.set_ylabel("Model Chl change (mg m$^{-3}$)")
    ax.text(0.97, 0.04, f"r = {r_all:.2f} (open ocean)", transform=ax.transAxes, ha="right", fontsize=6.5); ax.legend(loc="upper left", markerscale=3)
    ax.set_box_aspect(1); panel(ax, "c", x=0.0, ha="left", y=1.01)
    for j, (rn, lab) in enumerate([("wake_right", "Wake core, right of track"), ("wake_left", "Wake core, left of track"), ("control", "Control")]):
        ax = fig.add_subplot(gs[1, j]); event_band(ax)
        ax.plot(S.index, S[("satellite", rn)], color=C_SAT, lw=1.4, label="Satellite"); ax.plot(S.index, S[("model", rn)], color=C_MOD, lw=1.2, ls=(0, (3, 1.5)), label="Model")
        ax.set_ylabel("Surface Chl (mg m$^{-3}$)"); date_axis(ax, "2023-04-15", "2023-07-31"); ax.set_title(f"({'def'[j]}) {lab}", fontsize=7.2, pad=3, loc="left"); ax.grid(axis="y", alpha=0.6)
        if j == 0: ax.legend(loc="upper left")
    save(fig, "Figure_08_satellite_vs_model_chl", dpi)


def fig09(dpi):
    """Stratification, mixed layer and integrated NPP."""
    p = File(f"{DL}/phy_profiles_2023_AprAug.nc"); T = pd.date_range("2023-04-01", periods=153)
    pre = (T >= "2023-05-01") & (T <= "2023-05-07"); post = (T >= "2023-05-15") & (T <= "2023-05-19")
    blt = p.read("barrier_layer_thickness"); mld = p.read("mld_density")
    TS = pd.read_pickle(f"{O3}/diag_region_timeseries.pkl")
    fig = plt.figure(figsize=(FULL, 150 * MM)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.8], hspace=0.40, wspace=0.12)
    gsb = gs[1, :].subgridspec(1, 3, wspace=0.42)
    for j, (fld, cm, lo, hi, lab, tk, ext) in enumerate([(np.nanmean(blt[pre], 0), "viridis", 0, 30, "Barrier-layer thickness, 1–7 May (m)", [0, 10, 20, 30], "max"),
                                                       (np.nanmean(mld[pre], 0), "viridis", 0, 40, "Mixed-layer depth, 1–7 May (m)", [0, 10, 20, 30, 40], "max"),
                                                       (np.nanmean(mld[post], 0) - np.nanmean(mld[pre], 0), "RdBu_r", -30, 30, "MLD change, 15–19 May minus 1–7 May (m)", [-30, -15, 0, 15, 30], "both")]):
        ax = fig.add_subplot(gs[0, j]); pc = ax.pcolormesh(LON, LAT, np.where(OCEAN, fld, np.nan), cmap=cm, vmin=lo, vmax=hi, shading="nearest", rasterized=True)
        basemap(ax, labels=(True, j == 0)); outline(ax, REG == 1, "w" if j < 2 else "k", lw=0.7, ls="--"); outline(ax, REG == 3, C_CTRL if j == 2 else "#9ecae1", lw=0.8)
        hcbar(fig, pc, ax, lab, ticks=tk, extend=ext, pad=0.10); panel(ax, "abc"[j], x=0.0, ha="left", y=1.01)
    for j, (v, lab) in enumerate([("mld_density", "Mixed-layer depth (m)"), ("barrier_layer_thickness", "Barrier-layer thickness (m)"), ("npp_int_0_100m", "NPP, 0–100 m (mg C m$^{-2}$ d$^{-1}$)")]):
        ax = fig.add_subplot(gsb[0, j]); event_band(ax)
        ax.plot(T, TS[(v, "wake_right")], color=C_WAKE, lw=1.4, label="Wake core, right of track")
        ax.plot(T, TS[(v, "wake_left")], color=C_WAKE, lw=1.0, ls=(0, (3, 1.5)), label="Wake core, left of track")
        ax.plot(T, TS[(v, "control")], color=C_CTRL, lw=1.1, label="Control")
        ax.set_ylabel(lab); date_axis(ax, "2023-04-01", "2023-07-15"); ax.grid(axis="y", alpha=0.6)
        if j == 0: ax.legend(loc="upper left", fontsize=6)
        panel(ax, "def"[j], x=0.0, ha="left", y=1.01)
    save(fig, "Figure_09_stratification_mixing", dpi)


def fig10(dpi):
    """Composite profiles."""
    PR = pd.read_csv(f"{O3}/profiles_composite.csv")
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 150 * MM), sharey=True, gridspec_kw=dict(wspace=0.12, hspace=0.42))
    win = [("pre 1-7 May", "-", 1.5, "1–7 May (before)"), ("post 15-19 May", (0, (4, 1.6)), 1.3, "15–19 May (after)"), ("1-7 June", (0, (1, 1.2)), 1.2, "1–7 June")]
    specs = [("T", "Temperature (°C)", None), ("S", "Salinity (psu)", None), ("NO3", "NO$_3$ (mmol m$^{-3}$)", (0, 8)),
             ("PO4", "PO$_4$ (mmol m$^{-3}$)", (0, 1.0)), ("Chl", "Chl (mg m$^{-3}$)", None), ("NPP", "NPP (mg C m$^{-3}$ d$^{-1}$)", None)]
    for k, (ax, (v, lab, xl)) in enumerate(zip(axs.flat, specs)):
        for rn, c in [("wake_right", C_WAKE), ("control", C_CTRL)]:
            for wn, ls, lw, _ in win:
                q = PR[(PR.region == rn) & (PR.window == wn) & (PR["var"] == v)].dropna()
                ax.plot(q.value, q.depth, color=c, ls=ls, lw=lw)
        ax.set_ylim(150, 0); ax.set_xlabel(lab); ax.grid(alpha=0.5)
        if xl: ax.set_xlim(*xl)
        if k % 3 == 0: ax.set_ylabel("Depth (m)")
        ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
        panel(ax, "abcdef"[k], x=-0.02, y=1.13)
    h = [mlines.Line2D([], [], color=C_WAKE, lw=1.5, label="Wake core, right of track"), mlines.Line2D([], [], color=C_CTRL, lw=1.5, label="Control")] + \
        [mlines.Line2D([], [], color=C_INK, ls=ls, lw=lw, label=l) for _, ls, lw, l in win]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    save(fig, "Figure_10_profiles", dpi)


def fig11(dpi):
    """Nutrient and heat budget, 9-16 May."""
    B = pd.read_csv(f"{O3}/nutrient_budget_9_16May.csv"); Bs = pd.read_csv(f"{O3}/nutrient_budget_S_calibrated.csv")
    regs = [("wake", "Wake\ncore"), ("wake_right", "Right of\ntrack"), ("wake_left", "Left of\ntrack"), ("control", "Control")]
    fig = plt.figure(figsize=(FULL, 92 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    terms = [("observed", "Total change (model)", "#333333", None), ("mix", "Vertical mixing (S-calibrated)", C_CTRL, None), ("advection", "Horizontal advection", C_WAKE, None),
             ("storm_extra_uptake", "Storm-driven uptake", C_OUTER, None), ("rain_dilution", "Rain dilution", "#eda100", None), ("residual", "Residual", "#e0e0e0", "////")]
    width = 0.13; x = np.arange(len(regs))
    for i, (key, lab, col, hatch) in enumerate(terms):
        vals = []
        for rn, _ in regs:
            b = B[(B.variable == "NO3") & (B.region == rn)].iloc[0]; bs = Bs[(Bs.variable == "NO3") & (Bs.region == rn)].iloc[0]
            mix = bs.predicted_by_S_calibrated_mixing
            d = {"observed": b.observed, "mix": mix, "advection": b.advection, "storm_extra_uptake": b.storm_extra_uptake, "rain_dilution": b.rain_dilution}
            d["residual"] = b.observed - (mix + b.advection + b.storm_extra_uptake + b.rain_dilution)
            vals.append(d[key])
        ax.bar(x + (i - 2.5) * width, vals, width * 0.92, color=col, hatch=hatch, edgecolor="#7f7f7f" if hatch else "none", lw=0.4, label=lab)
    ax.axhline(0, color=C_INK, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([l for _, l in regs], fontsize=7)
    ax.set_ylabel("Surface NO$_3$ change, 9–16 May (mmol m$^{-3}$)"); ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=6.2, ncol=3); ax.set_title("(a) Nitrate budget", fontsize=7.5, loc="left")
    short = [("wake", "Wake"), ("wake_right", "Right"), ("wake_left", "Left"), ("control", "Ctrl")]
    axes2 = []
    for j, (v, lab, ttl) in enumerate([("PO4", "Surface PO$_4$ change (mmol m$^{-3}$)", "(b) Phosphate"), ("T", "Temperature change at 5 m (°C)", "(c) Temperature")]):
        ax = fig.add_subplot(gs[j + 1]); axes2.append(ax); xs = np.arange(len(short)); wd = 0.38
        obs = [Bs[(Bs.variable == v) & (Bs.region == rn)].observed.iloc[0] for rn, _ in short]
        prd = [Bs[(Bs.variable == v) & (Bs.region == rn)].predicted_by_S_calibrated_mixing.iloc[0] for rn, _ in short]
        ax.bar(xs - wd / 2, obs, wd * 0.92, color="#333333", label="Total change (model or reanalysis)"); ax.bar(xs + wd / 2, prd, wd * 0.92, color=C_CTRL, label="Predicted by vertical mixing alone (S-calibrated)")
        ax.axhline(0, color=C_INK, lw=0.6); ax.set_xticks(xs); ax.set_xticklabels([l for _, l in short], fontsize=7); ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.6)
        ax.set_title(ttl, fontsize=7.5, loc="left")
    axes2[0].legend(loc="upper center", bbox_to_anchor=(1.25, -0.20), fontsize=6.2, ncol=1)
    save(fig, "Figure_11_budget", dpi)


def fig12(dpi):
    """Model vs World Ocean Atlas 2023."""
    fig, axs = plt.subplots(2, 2, figsize=(HALF * 1.6, 128 * MM), gridspec_kw=dict(wspace=0.45, hspace=0.45))
    for k, (v, lab) in enumerate([("no3", "NO$_3$"), ("po4", "PO$_4$")]):
        df = pd.read_csv(f"{O3}/woa_vs_model_surface_{v}.csv"); ob = df.n_obs >= 1; ax = axs[0, k]
        ax.scatter(df.woa[~ob], df.model[~ob], s=4, c="#bdbdbd", lw=0, label="WOA cell, interpolated", rasterized=True)
        ax.scatter(df.woa[ob], df.model[ob], s=10, c=C_WAKE, edgecolor="k", lw=0.25, label="WOA cell with observations")
        lim = (0, 2) if v == "no3" else (0, 0.5); ylim = (0, 6) if v == "no3" else (0, 0.5)
        ax.plot([0, 6], [0, 6], color=C_INK, lw=0.6, ls="--")
        ax.set_xlim(*lim); ax.set_ylim(*ylim); ax.set_xlabel(f"WOA23 surface {lab} (mmol m$^{{-3}}$)"); ax.set_ylabel(f"Model surface {lab} (mmol m$^{{-3}}$)")
        ax.legend(loc="upper right", fontsize=5.8, markerscale=1.5); ax.set_title(f"({'ab'[k]}) Surface {lab}, Apr–Jun", fontsize=7.5, loc="left")
    W = pd.read_csv(f"{O3}/woa_may_profile_wake.csv"); PR = pd.read_csv(f"{O3}/profiles_composite.csv")
    for k, (v, wv, lab, xl) in enumerate([("NO3", "woa_no3", "NO$_3$ (mmol m$^{-3}$)", (0, 8)), ("PO4", "woa_po4", "PO$_4$ (mmol m$^{-3}$)", (0, 1.0))]):
        ax = axs[1, k]; q = PR[(PR.region == "wake") & (PR.window == "pre 1-7 May") & (PR["var"] == v)].dropna()
        ax.plot(q.value, q.depth, color=C_MOD, lw=1.5, label="Model, wake core,\n1–7 May 2023"); ax.plot(W[wv], W.depth, color=C_INK, lw=1.2, ls=(0, (3, 1.5)), label="WOA23, May\nclimatology")
        ax.set_ylim(100, 0); ax.set_xlim(*xl); ax.set_xlabel(lab); ax.set_ylabel("Depth (m)"); ax.grid(alpha=0.5); ax.legend(loc="lower left", fontsize=5.8)
        ax.set_title(f"({'cd'[k]}) {lab.split(' ')[0]} profile, wake core", fontsize=7.5, loc="left")
    save(fig, "Figure_12_WOA_validation", dpi)


def figA1(dpi):
    C = pd.read_csv(f"{O4}/correlations_effective_n.csv"); C = C[C.source == "wake"]
    labs = ["SST", "SSS", "NO3", "PO4", "Chl", "NPP"]; tl = ["SST", "SSS", "NO$_3$", "PO$_4$", "Chl", "NPP"]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 82 * MM), gridspec_kw=dict(wspace=0.15))
    for ax, w, ttl in zip(axs, ["pre", "post"], ["Before: 1 Apr–8 May", "After: 15 May–30 Jun"]):
        M = np.full((6, 6), np.nan); Pv = np.full((6, 6), np.nan)
        for _, r in C[C.window == w].iterrows():
            a, b = r.pair.split("-"); i, j = labs.index(a), labs.index(b); M[i, j] = M[j, i] = r.r; Pv[i, j] = Pv[j, i] = r.p
        Mm = np.where(np.tril(np.ones((6, 6)), -1) > 0, M, np.nan)[1:, :-1]
        im = ax.imshow(Mm, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(1, 6):
            for j in range(i):
                ax.text(j, i - 1, f"{M[i, j]:.2f}" + ("*" if Pv[i, j] < 0.05 else ""), ha="center", va="center", fontsize=6.8,
                        fontweight="bold" if Pv[i, j] < 0.05 else "normal", color="white" if abs(M[i, j]) > 0.6 else C_INK)
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(tl[:-1]); ax.set_yticklabels(tl[1:])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0); ax.set_title(ttl, fontsize=7.5, pad=4); panel(ax, "ab"[["pre", "post"].index(w)], x=0.0, ha="left", y=1.01)
    cb = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.025, pad=0.02); cb.set_label("Pearson r (detrended anomalies)", fontweight="bold"); cb.outline.set_linewidth(0.5)
    save(fig, "Figure_A1_correlations", dpi)


def figA2(dpi):
    L = pd.read_csv(f"{O4}/lag_correlation_curves.csv", index_col=0); lags = L.index.values
    cases = [("basin raw, whole Apr-Jun (as published, sign corrected)", C_MUTED, "Basin mean, raw, 1 Apr–30 Jun (original analysis)"),
             ("wake, whole Apr-Jun (detrended)", C_OUTER, "Wake core, detrended anomalies, 1 Apr–30 Jun"),
             ("wake, post-event 15 May-30 Jun (detrended)", C_WAKE, "Wake core, detrended anomalies, 15 May–30 Jun")]
    fig, ax = plt.subplots(figsize=(HALF * 1.35, 75 * MM))
    for name, c, lab in cases:
        ax.fill_between(lags, L[f"{name}|lo95"], L[f"{name}|hi95"], color=c, alpha=0.12, lw=0)
        ax.plot(lags, L[f"{name}|r"], color=c, lw=1.4, marker="o", ms=2, label=lab)
    ax.axhline(0, color=C_INK, lw=0.5); ax.axvline(0, color=C_MUTED, lw=0.4, ls=":")
    ax.set_xlabel("Lag L (days)"); ax.set_ylabel("r(L) = corr[NPP(t), NO$_3$(t − L)]"); ax.set_ylim(-1, 1); ax.set_xlim(-15, 15)
    ax.text(-14.5, -0.97, "← NPP leads NO$_3$", fontsize=6.3, va="bottom"); ax.text(14.5, -0.97, "NO$_3$ leads NPP →", fontsize=6.3, ha="right", va="bottom")
    ax.grid(alpha=0.5); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=6.2)
    save(fig, "Figure_A2_lag_correlation", dpi)


FIGS = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05, "06": fig06, "07": fig07, "08": fig08,
        "09": fig09, "10": fig10, "11": fig11, "12": fig12, "A1": figA1, "A2": figA2}

if __name__ == "__main__":
    dpi = int(sys.argv[1]); ids = sys.argv[2:] or list(FIGS)
    for i in ids:
        FIGS[i](dpi); print("done", i)
