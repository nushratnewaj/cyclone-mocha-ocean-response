import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 2a - storm geometry and masks on the 0.25 deg BGC grid (5-25N, 78-100E).

Track: IBTrACS v04r01 (same source as manuscript Fig. 1), 3-hourly, from 9 May 00 UTC to
landfall on 14 May (~06-07 UTC, Sittwe), densified to 15-min steps.  For every grid cell:
  dist_km  - distance to the closest over-ocean track point
  t_pass   - time of closest approach
  side     - +1 right of track (direction of motion), -1 left
Masks:
  ocean    - valid in the 2023 BGC file AND in the 1993-2022 climatology (common mask)
  shelf    - median GEBCO depth in the 0.25 deg cell shallower than 200 m
Regions (open ocean = not shelf). Wake distances use the cyclonic-storm segment (>=34 kt,
11 May 00 UTC -> landfall); JTWC R34 was ~220-420 km.
  1 wake core   : open ocean, 0-200 km from the >=34 kt segment
  2 outer wake  : open ocean, 200-400 km
  3 control     : open ocean, 8-20N, west of 87E (Bay proper, not Andaman Sea), >400 km from
                  the track AND ERA5 daily-max 10 m wind < 12 m/s on every day 9-15 May
  4 shelf       : shelf cells (any distance)
  0 other open ocean
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scipy.io as sio
from h5lite import File

D = DATA + ""; OUT = OUTB + "/step2"; os.makedirs(OUT, exist_ok=True)
R_EARTH = 6371.0


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2); dl = np.deg2rad(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def load_track():
    f = File(f"{D}/IBTrACS.NI.v04r01.nc")
    name = np.array([b"".join(r).decode().strip() for r in f.read("name")])
    i = int(np.where(name == "MOCHA")[0][0])
    t = f.read("time")[i]; ok = np.isfinite(t) & (t > -1e30)
    tt = pd.Timestamp("1858-11-17") + pd.to_timedelta(t[ok], unit="D")
    tr = pd.DataFrame({"lat": f.read("lat")[i][ok], "lon": f.read("lon")[i][ok],
                       "wmo_wind_kt": f.read("wmo_wind", raw=True)[i][ok].astype(float),
                       "usa_wind_kt": f.read("usa_wind", raw=True)[i][ok].astype(float),
                       "dist2land_km": f.read("dist2land", raw=True)[i][ok].astype(float)}, index=tt.round("min"))
    tr = tr.replace(-9999, np.nan)
    return tr


def densify(tr, t0="2023-05-09 00:00", t1="2023-05-14 06:00", step="15min"):
    seg = tr.loc[t0:t1, ["lat", "lon"]]
    idx = pd.date_range(seg.index[0], seg.index[-1], freq=step)
    d = seg.reindex(seg.index.union(idx)).interpolate("time").reindex(idx)
    return d


if __name__ == "__main__":
    tr = load_track(); tr.to_csv(f"{OUT}/mocha_track_ibtracs.csv")
    dense = densify(tr)
    b = File(f"{D}/nutrients_april-june-2023.nc"); lat = b.read("latitude"); lon = b.read("longitude")
    no3_0 = b.read("no3", sel=((0, 1), (0, 1), (0, 81), (0, 89)))[0, 0]
    clim = File(f"{D}/dl/bgc_surface_clim_doy.nc"); cm = clim.read("no3_mean")[0]
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    ocean = np.isfinite(no3_0) & np.isfinite(cm)
    print("ocean cells: 2023 file", int(np.isfinite(no3_0).sum()), "| common with climatology", int(ocean.sum()))

    # distance / passage time / side
    def nearest(dd):
        tl, tn = dd.lat.values, dd.lon.values
        dist = np.full(LAT.shape, np.inf); kmin = np.zeros(LAT.shape, int)
        for k in range(len(tl)):
            dk = hav(LAT, LON, tl[k], tn[k]); better = dk < dist
            dist[better] = dk[better]; kmin[better] = k
        return dist, kmin
    dist_all, _ = nearest(dense)                                   # whole over-ocean track (depression onward)
    dense = densify(tr, t0="2023-05-11 00:00")                       # cyclonic-storm segment
    tl, tn = dense.lat.values, dense.lon.values
    dist, kmin = nearest(dense)
    t_pass = dense.index.values[kmin]
    # heading at closest point (forward difference), cross product in local tangent plane
    k1 = np.clip(kmin + 1, 0, len(tl) - 1); k0 = np.clip(kmin - 1, 0, len(tl) - 1)
    hx = (tn[k1] - tn[k0]) * np.cos(np.deg2rad(tl[kmin])); hy = tl[k1] - tl[k0]
    rx = (LON - tn[kmin]) * np.cos(np.deg2rad(LAT)); ry = LAT - tl[kmin]
    cross = hx * ry - hy * rx
    side = np.where(cross < 0, 1, -1)           # NH: right of motion = negative z-cross
    # cells beyond the track ends (closest point is the first/last fix) are not 'beside' the track
    at_end = (kmin == 0) | (kmin == len(tl) - 1)

    # shelf from GEBCO: median elevation within each 0.25 deg cell
    g = sio.netcdf_file(f"{D}/gebco_2025_n25_0_s5_0_w78_0_e100_0.nc", "r", mmap=False)
    gl = g.variables["lat"][:].copy(); go = g.variables["lon"][:].copy(); ge = g.variables["elevation"][:].astype(np.float32)
    med = np.full(LAT.shape, np.nan)
    for i, la in enumerate(lat):
        ri = np.where(np.abs(gl - la) <= 0.125)[0]
        if not len(ri): continue
        rows = ge[ri[0]:ri[-1] + 1]
        for j, lo in enumerate(lon):
            cj = np.where(np.abs(go - lo) <= 0.125)[0]
            if len(cj): med[i, j] = np.median(rows[:, cj[0]:cj[-1] + 1])
    shelf = ocean & (med > -200)

    region = np.zeros(LAT.shape, int)
    opn = ocean & ~shelf
    region[opn & (dist <= 200) & ~at_end] = 1
    region[opn & (dist > 200) & (dist <= 400) & ~at_end] = 2
    e = File(f"{D}/dl/era5_daily_2023_AprAug.nc"); ws = e.read("wspd_max")
    te = pd.date_range("2023-04-01", periods=ws.shape[0])
    wmax = np.nanmax(ws[(te >= "2023-05-09") & (te <= "2023-05-15")], axis=0)
    region[opn & (LAT >= 8) & (LAT <= 20) & (LON < 87) & (dist_all > 400) & (wmax < 12)] = 3
    region[shelf] = 4
    region[~ocean] = -1
    np.savez_compressed(f"{OUT}/geometry.npz", lat=lat, lon=lon, dist_km=dist, side=side, at_end=at_end,
                        t_pass=t_pass.astype("datetime64[m]").astype("int64"), shelf=shelf, ocean=ocean,
                        region=region, gebco_median_m=med, dist_all_km=dist_all, era5_wmax_9_15May=wmax)
    names = {1: "wake core 0-200 km", 2: "outer wake 200-400 km", 3: "control (calm, W Bay)", 4: "shelf <200 m", 0: "other open ocean"}
    for k, n in names.items():
        m = region == k
        extra = f" | right/left: {int((m & (side == 1)).sum())}/{int((m & (side == -1)).sum())}" if k in (1, 2) else ""
        print(f"  region {k} {n:26s}: {int(m.sum()):5d} cells{extra} | ERA5 max wind 9-15 May median {np.median(wmax[m]):.1f} m/s")
    print("shelf cells lost from common mask (coastal):", int((np.isfinite(no3_0) & ~np.isfinite(cm) & (med > -200)).sum()))
