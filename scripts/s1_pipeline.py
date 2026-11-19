import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 1 - clean pipeline for the Cyclone Mocha revision.

Builds daily domain-mean surface series for every variable two ways:
  * 'uw'  : unweighted mean over ocean cells (what the original notebook did)
  * 'w'   : cos(latitude) area-weighted mean over ocean cells (corrected)
Physics (GLORYS12) is taken at the surface level only (depth index 0, 0.494 m);
the original notebook mixed 0.494 m and 1.541 m for SSS.

Outputs (in ../out):
  series_daily.csv   - daily series 2023-04-01..2023-06-30 (physics) and
                       2023-04-01..2026-01-31 (BGC), both weightings
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from h5lite import File

D = DATA + ""; OUT = OUTB + ""; os.makedirs(OUT, exist_ok=True)


def domain_means(arr, lat):
    """arr: (t, lat, lon) with NaN on land. Returns unweighted and cos-lat weighted means."""
    valid = np.isfinite(arr)
    uw = np.nanmean(arr, axis=(1, 2))
    w2d = np.cos(np.deg2rad(lat))[None, :, None] * np.ones_like(arr)
    w2d = np.where(valid, w2d, 0.0)
    w = np.nansum(np.where(valid, arr, 0.0) * w2d, axis=(1, 2)) / w2d.sum(axis=(1, 2))
    return uw, w


def load_bgc():
    f = File(f"{D}/nutrients_april-june-2023.nc")
    t = pd.to_datetime(f.read("time"), unit="s")
    lat = f.read("latitude"); lon = f.read("longitude")
    out = {}
    for v in ["chl", "no3", "po4", "nppv", "o2"]:
        a = f.read(v)[:, 0]          # single depth 0.5058 m
        out[v] = a
    f.close()
    return t, lat, lon, out


def load_phy():
    f = File(f"{D}/sst-sss-april-june-2023.nc")
    t = pd.to_datetime(f.read("time"), unit="s")
    lat = f.read("latitude"); lon = f.read("longitude"); dep = f.read("depth")
    n = len(t)
    sst = f.read("thetao", sel=((0, n), (0, 1), (0, len(lat)), (0, len(lon))))[:, 0]
    sss = f.read("so", sel=((0, n), (0, 1), (0, len(lat)), (0, len(lon))))[:, 0]
    # second level only to quantify the original two-depth bug
    sss1 = f.read("so", sel=((0, n), (1, 1), (0, len(lat)), (0, len(lon))))[:, 0]
    f.close()
    return t, lat, lon, dep, sst, sss, sss1


if __name__ == "__main__":
    tb, latb, lonb, bgc = load_bgc()
    tp, latp, lonp, dep, sst, sss, sss1 = load_phy()

    cols = {}
    for v, a in bgc.items():
        uw, w = domain_means(a, latb)
        cols[f"{v}_uw"] = pd.Series(uw, index=tb); cols[f"{v}_w"] = pd.Series(w, index=tb)
    for v, a in [("sst", sst), ("sss", sss), ("sss_1p5m", sss1)]:
        uw, w = domain_means(a, latp)
        cols[f"{v}_uw"] = pd.Series(uw, index=tp); cols[f"{v}_w"] = pd.Series(w, index=tp)
    df = pd.DataFrame(cols); df.index.name = "date"
    df.to_csv(f"{OUT}/series_daily.csv", float_format="%.6g")

    # grid bookkeeping for the methods section
    nb = np.isfinite(bgc["no3"][0]).sum(); npc = np.isfinite(sst[0]).sum()
    print("BGC grid", latb.size, "x", lonb.size, "ocean cells", nb, "| depth 0.5058 m")
    print("PHY grid", latp.size, "x", lonp.size, "ocean cells", npc, "| depths", dep)
    print(df.loc["2023-05-08":"2023-05-18", ["no3_uw", "no3_w", "sst_uw", "sst_w", "sss_uw", "sss_w"]].round(4))
