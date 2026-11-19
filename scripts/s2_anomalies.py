import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 2b - 2023 anomalies against the 1993-2022 daily climatology, by region, plus the
interannual context from the 1993-2025 pentad archive.

Region means are cos(lat)-weighted over the common ocean mask (geometry.npz).
Physics: original 1/12 deg file (Apr-Jun 2023, daily) regridded EXACTLY as in Colab
(3x3 block nanmean, then bilinear to the 0.25 grid), checked against the Colab pentads.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
from h5lite import File
np.seterr(all="ignore")
import warnings; warnings.filterwarnings("ignore")

D = DATA + ""; DL = f"{D}/dl"; O = OUTB + "/step2"
G = np.load(f"{O}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; REG = G["region"]; SIDE = G["side"]
OCEAN = G["ocean"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
REGIONS = {"wake": REG == 1, "outer": REG == 2, "control": REG == 3, "shelf": REG == 4,
           "open_all": OCEAN & ~G["shelf"], "wake_right": (REG == 1) & (SIDE == 1), "wake_left": (REG == 1) & (SIDE == -1)}
QUIET = [1993, 1995, 2000, 2001, 2005, 2012, 2014, 2015, 2025]   # no >=34 kt storm in domain Apr-Jun (IBTrACS)


def rmean(a, m):
    """cos-lat weighted mean over mask m for array (..., lat, lon)."""
    w = WT * m
    ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


def to_grid_phy(a, lat_f, lon_f):
    """replicates Colab to_grid: 3x3 block nanmean (trim) then bilinear onto GRID with extrapolation."""
    nt = a.shape[0]; ny = (a.shape[1] // 3) * 3; nx = (a.shape[2] // 3) * 3
    b = a[:, :ny, :nx].reshape(nt, ny // 3, 3, nx // 3, 3)
    cb = np.nanmean(b, axis=(2, 4))
    la = lat_f[:ny].reshape(-1, 3).mean(1); lo = lon_f[:nx].reshape(-1, 3).mean(1)
    P = np.stack(np.meshgrid(LAT, LON, indexing="ij"), -1)
    out = np.empty((nt, LAT.size, LON.size), dtype="float32")
    for t in range(nt):
        f = RegularGridInterpolator((la, lo), cb[t], method="linear", bounds_error=False, fill_value=None)
        out[t] = f(P)
    return out


def leap_safe_doy(tt):
    return np.asarray(tt.dayofyear - ((tt.is_leap_year) & (tt.month > 2)).astype(int))


def clim_on(tt, clim, v):
    doy = clim["doy"]; k = leap_safe_doy(tt) - doy[0]
    return clim[f"{v}_mean"][k], clim[f"{v}_std"][k]


def read_clim(path, vars_):
    f = File(path); c = {"doy": f.read("doy").astype(int)}
    for v in vars_:
        c[f"{v}_mean"] = f.read(f"{v}_mean"); c[f"{v}_std"] = f.read(f"{v}_std")
    return c


if __name__ == "__main__":
    BV = {"chl": "chl", "no3": "no3", "po4": "po4", "npp": "nppv"}
    PV = {"sst": "thetao", "sss": "so"}
    cb = read_clim(f"{DL}/bgc_surface_clim_doy.nc", BV.values())
    cp = read_clim(f"{DL}/phy_surface_clim_doy.nc", ["thetao", "so", "mlotst"])

    # ---------- daily 2023-2025 BGC (original file, Apr-Sep each year) ----------
    fb = File(f"{D}/nutrients_april-june-2023.nc"); tb = pd.to_datetime(fb.read("time"), unit="s")
    daily = {}
    for key, v in BV.items():
        arr = fb.read(v)[:, 0]
        arr = np.where(OCEAN, arr, np.nan)
        for y in [2023, 2024, 2025]:
            k = np.where((tb >= f"{y}-04-01") & (tb <= f"{y}-09-30"))[0]
            tt = tb[k]; x = arr[k]
            m, s = clim_on(tt, cb, v)
            an = x - m
            for rn, rm in REGIONS.items():
                daily[(y, key, rn, "anom")] = pd.Series(rmean(an, rm), index=tt)
                daily[(y, key, rn, "value")] = pd.Series(rmean(x, rm), index=tt)
                daily[(y, key, rn, "clim")] = pd.Series(rmean(np.where(np.isfinite(x), m, np.nan), rm), index=tt)
            if y == 2023:
                np.save(f"{O}/anom2023_{key}.npy", an.astype("float32")); np.save(f"{O}/z2023_{key}.npy", (an / s).astype("float32"))
    np.save(f"{O}/time2023_bgc.npy", tb[(tb >= "2023-04-01") & (tb <= "2023-09-30")].values)

    # ---------- daily 2023 physics Apr-Jun (regridded) ----------
    fp = File(f"{D}/sst-sss-april-june-2023.nc"); tp = pd.to_datetime(fp.read("time"), unit="s")
    lat_f = fp.read("latitude"); lon_f = fp.read("longitude"); n = len(tp)
    for key, v in PV.items():
        a = fp.read(v, sel=((0, n), (0, 1), (0, lat_f.size), (0, lon_f.size)))[:, 0]
        g = to_grid_phy(a, lat_f, lon_f); g = np.where(OCEAN, g, np.nan)
        m, s = clim_on(tp, cp, v); an = g - m
        for rn, rm in REGIONS.items():
            daily[(2023, key, rn, "anom")] = pd.Series(rmean(an, rm), index=tp)
            daily[(2023, key, rn, "value")] = pd.Series(rmean(g, rm), index=tp)
            daily[(2023, key, rn, "clim")] = pd.Series(rmean(np.where(np.isfinite(g), m, np.nan), rm), index=tp)
        np.save(f"{O}/anom2023_{key}.npy", an.astype("float32")); np.save(f"{O}/z2023_{key}.npy", (an / s).astype("float32"))
        if key == "sst": np.save(f"{O}/sst2023_grid.npy", g.astype("float32"))
    np.save(f"{O}/time2023_phy.npy", tp.values)

    dd = pd.DataFrame(daily); dd.columns = pd.MultiIndex.from_tuples(dd.columns, names=["year", "var", "region", "kind"])
    dd.to_pickle(f"{O}/daily_region_series.pkl")

    # ---------- pentads 1993-2025: region-mean anomalies vs climatological pentads ----------
    pent = {}
    for path, cl, vars_ in [(f"{DL}/bgc_surface_pentads_by_year.nc", cb, BV), (f"{DL}/phy_surface_pentads_by_year.nc", cp, {**PV, "mld": "mlotst"})]:
        f = File(path); yrs = f.read("year").astype(int)
        for key, v in vars_.items():
            arr = np.where(OCEAN, f.read(v), np.nan)                    # (year, pentad, lat, lon)
            cm = np.stack([np.nanmean(cl[f"{v}_mean"][5 * j:5 * j + 5], 0) for j in range(36)])   # clim pentads
            for rn, rm in REGIONS.items():
                pent[(key, rn)] = pd.DataFrame(rmean(arr - cm[None], rm), index=yrs, columns=range(36))
    pd.to_pickle(pent, f"{O}/pentad_region_anoms.pkl")

    # sanity: Apr-Jun 2023 physics regrid vs Colab pentads (should agree to ~1e-3)
    s23 = dd[(2023, "sst", "open_all", "anom")]
    p23 = pent[("sst", "open_all")].loc[2023]
    mine = np.array([s23.iloc[5 * j:5 * j + 5].mean() for j in range(18)])
    print("check physics regrid vs Colab pentads (SST anomaly, open ocean): max |diff| =", np.nanmax(abs(mine - p23.values[:18])).round(4))
    b23 = dd[(2023, "no3", "open_all", "anom")]; pb = pent[("no3", "open_all")].loc[2023]
    print("check BGC daily vs pentads (NO3 anomaly): max |diff| =", np.nanmax(abs(np.array([b23.iloc[5 * j:5 * j + 5].mean() for j in range(36)]) - pb.values)).round(4))
