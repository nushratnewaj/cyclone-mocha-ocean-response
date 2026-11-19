import json
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Cyclone Mocha revision — data downloads (run in Google Colab)

This notebook only **downloads and reduces** data. All analysis is done afterwards in one consistent pipeline.
Every block cuts the data to the study domain (5–25°N, 78–100°E) and to the 0.25° BGC grid **before saving**, so the files you upload back stay small.

| Block | What | Answers reviewer item | Approx. output |
|---|---|---|---|
| A | Setup + logins | – | – |
| B | BGC surface climatology 1993–2022 + every year 1993–2025 (chl, no3, po4, nppv) | (a) climatology / non-cyclone years | ~60–120 MB |
| C | Physics surface climatology 1993–2022 + every year (SST, SSS, MLD `mlotst`) | (a), (c) | ~60–120 MB |
| D | 2023 Apr–Aug physics profiles → MLD, ILD, barrier layer, T/S at standard depths | (c) stratification / barrier layer | ~40 MB |
| E | 2023 Apr–Aug BGC profiles → depth-integrated NPP & chl, nitracline, no3/chl/po4 profiles | (c), depth-integrated NPP, subsurface chl max | ~80 MB |
| F | ERA5 hourly winds & rain 2023 Apr–Aug → daily wind stress & rain | (c) wind forcing, replaces TerraClimate | ~30 MB |
| G | OSTIA observed SST 15 Apr–30 Jun 2023 | cold-wake comparison with Puri (2026) | ~25 MB |
| H | WOA23 nitrate & phosphate (Apr–Jun) + BGC-Argo floats in the box | (f) phosphate validation | small |
| I | Zip everything | – | – |

**Run order:** A first, then B–H in any order (each is independent and saves to `/content/mocha_out/`).

**Blocks B and C** download 33 seasons each. They run `N_WORKERS` years in parallel and save every finished year to Google Drive (`MyDrive/mocha_cache`). If Colab disconnects, just run block A again and then re-run B or C: years already on Drive are skipped. Memory stays under ~2 GB. Time one year first (see the tip in block B) to estimate the total. If one block fails, the others still work — tell me which one and paste the error.

Accounts needed: **Copernicus Marine** (you already have one) and, for block F only, a free **Copernicus Climate Data Store (CDS)** account with your API key.
""")

code(r"""
# ===== A. SETUP =====
!pip -q install copernicusmarine xarray netCDF4 dask gsw cdsapi argopy
import os, numpy as np, pandas as pd, xarray as xr, warnings, time
warnings.filterwarnings("ignore")
import copernicusmarine as cm
OUT = "/content/mocha_out"; os.makedirs(OUT, exist_ok=True)

# Study domain and the target 0.25-degree grid (identical to the BGC product grid used in the paper)
LON0, LON1, LAT0, LAT1 = 78.0, 100.0, 5.0, 25.0
GRID_LAT = np.round(np.arange(5.0, 25.0001, 0.25), 2)    # 81 points
GRID_LON = np.round(np.arange(78.0, 100.0001, 0.25), 2)  # 89 points
SEASON = ("04-01", "09-30")                             # Apr-Sep window kept for every year
ENC = lambda ds: {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}

# ---- speed / safety settings for blocks B and C ----
N_WORKERS = 4          # years downloaded in parallel (network-bound; 4-6 is a good range)
USE_DRIVE = True       # cache each finished year on Google Drive -> a disconnect loses nothing
if USE_DRIVE:
    from google.colab import drive; drive.mount("/content/drive")
    CACHE = "/content/drive/MyDrive/mocha_cache"
else:
    CACHE = "/content/mocha_cache"
os.makedirs(CACHE, exist_ok=True)

cm.login()   # enter your Copernicus Marine username / password when prompted
""")

code(r"""
# Helper: open a Copernicus dataset lazily, trying the multi-year ('my') and interim ('myint') parts
def open_parts(base_id, variables, depth=(0.0, 1.0)):
    parts = {}
    for tag in ["my", "myint"]:
        dsid = base_id.replace("_my_", f"_{tag}_")
        try:
            ds = cm.open_dataset(dataset_id=dsid, variables=variables,
                                 minimum_longitude=LON0, maximum_longitude=LON1,
                                 minimum_latitude=LAT0, maximum_latitude=LAT1,
                                 minimum_depth=depth[0], maximum_depth=depth[1])
            t = pd.to_datetime(ds.time.values)
            parts[tag] = ds
            print(f"  {dsid}: {t.min().date()} -> {t.max().date()}  vars={list(ds.data_vars)}")
        except Exception as e:
            print(f"  {dsid}: not available ({str(e)[:120]})")
    return parts

def year_slice(parts, year):
    # pick whichever part covers the season of this year; concatenate if it straddles both
    a, b = pd.Timestamp(f"{year}-{SEASON[0]}"), pd.Timestamp(f"{year}-{SEASON[1]}")
    pieces = []
    for tag, ds in parts.items():
        sub = ds.sel(time=slice(a, b))
        if sub.time.size: pieces.append(sub)
    if not pieces: return None
    out = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
    _, idx = np.unique(out.time.values, return_index=True)
    return out.isel(time=np.sort(idx))

def to_grid(da):
    # 1/12 deg -> 0.25 deg: 3x3 block mean, then put on the exact BGC grid
    if da.latitude.size > 100:
        da = da.coarsen(latitude=3, longitude=3, boundary="trim").mean()
    return da.interp(latitude=GRID_LAT, longitude=GRID_LON, method="linear", kwargs={"fill_value": None})

from concurrent.futures import ThreadPoolExecutor, as_completed

def grid_year(parts, variables, y, prefix):
    # Download one season (1 Apr-30 Sep) of one year, put it on the 0.25-deg grid, cache it as .npz
    fn = f"{CACHE}/{prefix}_{y}.npz"
    if os.path.exists(fn): return y, "cached"
    err = ""
    for attempt in range(3):                      # retry on network hiccups
        try:
            sub = year_slice(parts, y)
            if sub is None: return y, "no data"
            sub = sub[variables]
            if "depth" in sub.dims: sub = sub.isel(depth=0)
            sub = sub.load()
            tt = pd.to_datetime(sub.time.values)
            d = tt.dayofyear - ((tt.is_leap_year) & (tt.month > 2)).astype(int)   # leap-safe day-of-year
            out = {"doy": np.asarray(d, dtype="int16"), "time": tt.values.astype("datetime64[D]").astype("int64")}
            for v in variables: out[v] = to_grid(sub[v]).values.astype("float32")
            del sub
            np.savez_compressed(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn)
            return y, "ok"
        except Exception as e:
            err = str(e)[:150]; time.sleep(15 * (attempt + 1))
    return y, "FAILED: " + err

def run_years(parts, variables, prefix, years=range(1993, 2026)):
    t0 = time.time()
    with ThreadPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(grid_year, parts, variables, y, prefix) for y in years]
        for f in as_completed(futs):
            y, st = f.result()
            print(f"{y}: {st:<10s} {(time.time() - t0) / 60:6.1f} min elapsed", flush=True)
    missing = [y for y in years if not os.path.exists(f"{CACHE}/{prefix}_{y}.npz")]
    print("missing years:", missing if missing else "none - ready for build_outputs()")

def build_outputs(variables, prefix, years=range(1993, 2026), clim_years=(1993, 2022)):
    # (1) daily climatology mean/std/n per day-of-year and cell from clim_years,
    # (2) 5-day means for EVERY year (for non-cyclone-year comparisons)
    doy = np.arange(91, 274)                      # 1 Apr .. 30 Sep (non-leap numbering)
    shp = (doy.size, GRID_LAT.size, GRID_LON.size)
    S = {v: np.zeros(shp) for v in variables}; Q = {v: np.zeros(shp) for v in variables}; N = {v: np.zeros(shp) for v in variables}
    pent = {v: [] for v in variables}; got = []
    for y in years:
        fn = f"{CACHE}/{prefix}_{y}.npz"
        if not os.path.exists(fn): print("missing", y); continue
        z = np.load(fn); d = z["doy"].astype(int) - 91
        tt = pd.to_datetime(z["time"], unit="D"); kp = (tt - pd.Timestamp(f"{y}-04-01")).days // 5
        for v in variables:
            g = z[v].astype("float64")
            if clim_years[0] <= y <= clim_years[1]:
                for i, k in enumerate(d):
                    if 0 <= k < doy.size:
                        ok = np.isfinite(g[i]); S[v][k][ok] += g[i][ok]; Q[v][k][ok] += g[i][ok] ** 2; N[v][k][ok] += 1
            p = np.full((36,) + shp[1:], np.nan, dtype="float32")
            for j in range(36):
                sel = np.asarray(kp == j)
                if sel.any(): p[j] = np.nanmean(g[sel], axis=0)
            pent[v].append(p)
        got.append(y)
    clim = xr.Dataset(coords={"doy": doy, "latitude": GRID_LAT, "longitude": GRID_LON})
    for v in variables:
        n = np.where(N[v] > 0, N[v], np.nan); m = S[v] / n
        clim[f"{v}_mean"] = (("doy", "latitude", "longitude"), m.astype("float32"))
        clim[f"{v}_std"] = (("doy", "latitude", "longitude"), np.sqrt(np.maximum(Q[v] / n - m ** 2, 0)).astype("float32"))
        clim[f"{v}_n"] = (("doy", "latitude", "longitude"), N[v].astype("int16"))
    clim.attrs.update(note=f"daily climatology {clim_years[0]}-{clim_years[1]}, doy 91 = 1 Apr (non-leap numbering)")
    clim.to_netcdf(f"{OUT}/{prefix}_clim_doy.nc", encoding=ENC(clim))
    py = xr.Dataset({v: (("year", "pentad", "latitude", "longitude"), np.stack(pent[v])) for v in variables},
                    coords={"year": got, "pentad": np.arange(36), "latitude": GRID_LAT, "longitude": GRID_LON})
    py.attrs.update(note="5-day means starting 1 Apr each year; pentad 0 = 1-5 Apr")
    py.to_netcdf(f"{OUT}/{prefix}_pentads_by_year.nc", encoding=ENC(py))
    print("saved", prefix, "| years used:", got[0], "-", got[-1], f"({len(got)})")
""")

code(r"""
# ===== B. BGC SURFACE: climatology 1993-2022 + pentads for every year (parallel + resumable) =====
BGC_VARS = ["chl", "no3", "po4", "nppv"]
bgc_parts = open_parts("cmems_mod_glo_bgc_my_0.25deg_P1D-m", BGC_VARS, depth=(0.0, 1.0))
# Tip: time one year first, e.g. run_years(bgc_parts, BGC_VARS, "bgc_surface", years=[2022])
run_years(bgc_parts, BGC_VARS, "bgc_surface")     # re-run this cell after any disconnect: finished years are skipped
build_outputs(BGC_VARS, "bgc_surface")
""")

code(r"""
# ===== C. PHYSICS SURFACE: SST, SSS and model mixed-layer depth (parallel + resumable) =====
PHY_VARS = ["thetao", "so", "mlotst"]
phy_parts = open_parts("cmems_mod_glo_phy_my_0.083deg_P1D-m", PHY_VARS, depth=(0.0, 1.0))
run_years(phy_parts, PHY_VARS, "phy_surface")     # re-run after any disconnect: finished years are skipped
build_outputs(PHY_VARS, "phy_surface")
""")

code(r"""
# ===== D. 2023 PHYSICS PROFILES -> stratification diagnostics (Apr-Aug 2023) =====
import gsw
STD_Z = np.array([0, 5, 10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200])
phy3d = open_parts("cmems_mod_glo_phy_my_0.083deg_P1D-m", ["thetao", "so"], depth=(0.0, 300.0))
res = []
for m in ["04", "05", "06", "07", "08"]:
    t0 = time.time()
    a = pd.Timestamp(f"2023-{m}-01"); b = a + pd.offsets.MonthEnd(0)
    pieces = [ds.sel(time=slice(a, b)) for ds in phy3d.values() if ds.sel(time=slice(a, b)).time.size]
    sub = xr.concat(pieces, "time") if len(pieces) > 1 else pieces[0]
    sub = sub.sel(latitude=GRID_LAT, longitude=GRID_LON, method="nearest").load()   # 0.25-deg sampling
    sub = sub.assign_coords(latitude=GRID_LAT, longitude=GRID_LON)
    z = sub.depth.values
    T = sub.thetao.transpose("time", "depth", "latitude", "longitude").values
    SP = sub.so.transpose("time", "depth", "latitude", "longitude").values
    P = z[None, :, None, None]; LO = GRID_LON[None, None, None, :]; LA = GRID_LAT[None, None, :, None]
    SA = gsw.SA_from_SP(SP, P, LO, LA); CT = gsw.CT_from_pt(SA, T)
    R = gsw.sigma0(SA, CT)                                             # (t, z, y, x)
    i10 = int(np.argmin(abs(z - 10)))                                   # 10 m reference (de Boyer Montegut et al. 2004)
    def crossing(field, ref, thr, sign):
        d = (field - ref[:, None]) * sign                              # positive when criterion is approached
        mask = d >= thr; mask[:, :i10] = False
        k = np.argmax(mask, axis=1); has = mask.any(axis=1)
        k0 = np.clip(k - 1, 0, None)
        f1 = np.take_along_axis(d, k[:, None], 1)[:, 0]; f0 = np.take_along_axis(d, k0[:, None], 1)[:, 0]
        z1 = z[k]; z0 = z[k0]
        frac = np.where(f1 != f0, (thr - f0) / (f1 - f0), 0.0)
        out = z0 + frac * (z1 - z0); return np.where(has, out, np.nan)
    MLD = crossing(R, R[:, i10], 0.03, +1)                             # density criterion 0.03 kg m-3
    ILD = crossing(T, T[:, i10], 0.2, -1)                              # temperature criterion 0.2 degC
    BLT = ILD - MLD
    N2 = 9.81 / 1025.0 * np.gradient(R, z, axis=1)
    N2max = np.nanmax(np.where(z[None, :, None, None] <= 200, N2, np.nan), axis=1)
    out = xr.Dataset({"mld_density": (("time", "latitude", "longitude"), MLD.astype("float32")),
                      "ild_temperature": (("time", "latitude", "longitude"), ILD.astype("float32")),
                      "barrier_layer_thickness": (("time", "latitude", "longitude"), BLT.astype("float32")),
                      "N2_max_0_200m": (("time", "latitude", "longitude"), N2max.astype("float32"))},
                     coords={"time": sub.time.values, "latitude": GRID_LAT, "longitude": GRID_LON})
    prof = sub[["thetao", "so"]].interp(depth=STD_Z.astype(float)).astype("float32")
    res.append(xr.merge([out, prof])); print("2023-" + m, "done", round(time.time() - t0), "s")
D = xr.concat(res, "time")
D.attrs.update(note="MLD: sigma0 +0.03 from 10 m; ILD: T -0.2 degC from 10 m; BLT = ILD - MLD; sampled every 3rd 1/12-deg point (0.25 deg)")
D.to_netcdf(f"{OUT}/phy_profiles_2023_AprAug.nc", encoding=ENC(D)); print(D)
""")

code(r"""
# ===== E. 2023 BGC PROFILES (Apr-Aug): depth-integrated NPP/chl, nitracline, standard-depth profiles =====
bgc3d = open_parts("cmems_mod_glo_bgc_my_0.25deg_P1D-m", ["chl", "no3", "po4", "nppv"], depth=(0.0, 300.0))
STD_Z = np.array([0, 5, 10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200])
res = []
for m in ["04", "05", "06", "07", "08"]:
    t0 = time.time()
    a = pd.Timestamp(f"2023-{m}-01"); b = a + pd.offsets.MonthEnd(0)
    pieces = [ds.sel(time=slice(a, b)) for ds in bgc3d.values() if ds.sel(time=slice(a, b)).time.size]
    sub = (xr.concat(pieces, "time") if len(pieces) > 1 else pieces[0]).load()
    sub = sub.interp(latitude=GRID_LAT, longitude=GRID_LON, method="nearest")
    z = sub.depth.values
    # layer thickness for trapezoidal integration down to 200 m
    zz = z[z <= 200]; s200 = sub.sel(depth=zz)
    npp_int = s200.nppv.integrate("depth")                               # mg C m-2 d-1
    chl_int = s200.chl.integrate("depth")                                # mg m-2
    npp_int100 = sub.sel(depth=z[z <= 100]).nppv.integrate("depth")
    zscm = s200.chl.idxmax("depth")                                      # depth of subsurface chl maximum
    # nitracline: first depth where NO3 exceeds surface value + 1 mmol m-3
    no3 = s200.no3; exc = (no3 - no3.isel(depth=0)) >= 1.0
    zn = no3.depth.where(exc).min("depth")
    out = xr.Dataset({"npp_int_0_200m": npp_int, "npp_int_0_100m": npp_int100, "chl_int_0_200m": chl_int,
                      "depth_chl_max": zscm, "nitracline_depth_plus1uM": zn})
    prof = sub[["chl", "no3", "po4", "nppv"]].interp(depth=STD_Z.astype(float))
    res.append(xr.merge([out, prof]).astype("float32")); print("2023-" + m, "done", round(time.time() - t0), "s")
E = xr.concat(res, "time")
E.attrs.update(note="integrals trapezoidal over model levels; units: npp mg C m-2 d-1, chl mg m-2")
E.to_netcdf(f"{OUT}/bgc_profiles_2023_AprAug.nc", encoding=ENC(E)); print(E)
""")

md(r"""
### Block F needs a CDS key
1. Register (free) at https://cds.climate.copernicus.eu, open your profile page and copy the **API key**.
2. Accept the licence of *ERA5 hourly data on single levels* on its dataset page (Download tab → licence).
3. Paste the key into the cell below.
""")

code(r"""
# ===== F. ERA5 hourly 10 m wind + precipitation, Apr-Aug 2023 -> daily wind stress and rain =====
CDS_KEY = "PASTE-YOUR-CDS-API-KEY-HERE"
open(os.path.expanduser("~/.cdsapirc"), "w").write(f"url: https://cds.climate.copernicus.eu/api\nkey: {CDS_KEY}\n")
import cdsapi
c = cdsapi.Client()
days = [f"{d:02d}" for d in range(1, 32)]; hours = [f"{h:02d}:00" for h in range(24)]
daily = []
for m in ["04", "05", "06", "07", "08"]:
    fn = f"/content/era5_2023_{m}.nc"
    if not os.path.exists(fn):
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"],
            "year": "2023", "month": m, "day": days, "time": hours,
            "area": [LAT1, LON0, LAT0, LON1], "data_format": "netcdf", "download_format": "unarchived"}, fn)
    ds = xr.open_dataset(fn)
    if "valid_time" in ds.dims or "valid_time" in ds.coords: ds = ds.rename(valid_time="time")
    if "expver" in ds.dims: ds = ds.max("expver")
    ds = ds.sortby("latitude")
    U, V = ds.u10, ds.v10; W = np.sqrt(U ** 2 + V ** 2)
    # Large & Pond (1981) drag coefficient, rho_air = 1.22 kg m-3; stress computed hourly, then averaged
    Cd = xr.where(W < 11, 1.2e-3, (0.49 + 0.065 * W) * 1e-3)
    taux = 1.22 * Cd * W * U; tauy = 1.22 * Cd * W * V
    d = xr.Dataset({"taux": taux, "tauy": tauy, "wspd": W}).resample(time="1D").mean()
    d["wspd_max"] = W.resample(time="1D").max()
    d["precip_mm"] = (ds.tp * 1000).resample(time="1D").sum()
    daily.append(d.interp(latitude=GRID_LAT, longitude=GRID_LON).astype("float32")); print("ERA5 2023-" + m, "done")
F = xr.concat(daily, "time")
F.attrs.update(note="ERA5 hourly -> daily mean stress (N m-2, Large & Pond 1981 Cd), daily max wind, daily total precip (mm)")
F.to_netcdf(f"{OUT}/era5_daily_2023_AprAug.nc", encoding=ENC(F)); print(F)
""")

code(r"""
# ===== G. OSTIA observed SST, 15 Apr - 30 Jun 2023 (cold wake vs Puri 2026) =====
G = None
for dsid in ["METOFFICE-GLO-SST-L4-REP-OBS-SST", "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"]:
    try:
        G = cm.open_dataset(dataset_id=dsid, variables=["analysed_sst"],
                            minimum_longitude=LON0, maximum_longitude=LON1, minimum_latitude=LAT0, maximum_latitude=LAT1,
                            start_datetime="2023-04-15T00:00:00", end_datetime="2023-06-30T00:00:00").load()
        print("OSTIA from", dsid, pd.to_datetime(G.time.values).min().date(), "->", pd.to_datetime(G.time.values).max().date()); break
    except Exception as e:
        print(dsid, "failed:", str(e)[:150])
if G is not None:
    G["analysed_sst"] = (G.analysed_sst - 273.15).astype("float32"); G.analysed_sst.attrs["units"] = "degC"
    G.to_netcdf(f"{OUT}/ostia_sst_2023_15Apr_30Jun.nc", encoding=ENC(G))
""")

code(r"""
# ===== H1. World Ocean Atlas 2023 monthly nitrate & phosphate (1 deg), Apr-Jun, upper 100 m =====
import urllib.request
woa = {"n": [], "p": []}
for var, code_ in [("nitrate", "n"), ("phosphate", "p")]:
    for mm in ["04", "05", "06"]:
        url = f"https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/{var}/netcdf/all/1.00/woa23_all_{code_}{mm}_01.nc"
        fn = f"/content/woa23_{code_}{mm}.nc"
        try:
            if not os.path.exists(fn): urllib.request.urlretrieve(url, fn)
            ds = xr.open_dataset(fn, decode_times=False)
            keep = [v for v in ds.data_vars if v.endswith(("_an", "_mn", "_dd", "_sd"))]
            sub = ds[keep].sel(lat=slice(LAT0, LAT1), lon=slice(LON0, LON1), depth=slice(0, 100)).squeeze()
            sub = sub.expand_dims(month=[int(mm)]); woa[code_].append(sub); print("WOA23", var, mm, "ok")
        except Exception as e:
            print("WOA23", var, mm, "FAILED:", str(e)[:150], "\n   url:", url)
parts_ = [xr.concat(v, "month") for v in woa.values() if v]
if parts_:
    W = xr.merge(parts_, compat="override")
    W.to_netcdf(f"{OUT}/woa23_no3_po4_AprJun_0_100m.nc"); print(W)
""")

code(r"""
# ===== H2. BGC-Argo floats in the box (Mar-Aug 2023, any year 2019-2025 as backup) =====
try:
    from argopy import ArgoIndex
    idx = ArgoIndex(index_file="bgc-s")
    for box in [[LON0, LON1, LAT0, LAT1, "2023-03-01", "2023-08-31"], [LON0, LON1, LAT0, LAT1, "2019-01-01", "2025-12-31"]]:
        idx.search_lat_lon_tim(box)
        df = idx.to_dataframe()
        print(box[-2:], "BGC profiles:", len(df))
        if len(df):
            df.to_csv(f"{OUT}/bgc_argo_index_{box[-2][:4]}_{box[-1][:4]}.csv", index=False)
            print(df[["date", "latitude", "longitude", "parameters"]].head(10).to_string())
    from argopy import DataFetcher
    try:
        A = DataFetcher(ds="bgc", mode="expert", params="all").region([LON0, LON1, LAT0, LAT1, 0, 300, "2019-01-01", "2025-12-31"]).to_xarray()
        A.to_netcdf(f"{OUT}/bgc_argo_profiles_2019_2025.nc"); print(A)
    except Exception as e:
        print("profile download failed (index saved; that is enough for now):", str(e)[:200])
except Exception as e:
    print("argopy failed:", str(e)[:200])
""")

code(r"""
# ===== I. ZIP EVERYTHING and download =====
import shutil, glob
for f in sorted(glob.glob(f"{OUT}/*")): print(f"{os.path.getsize(f)/1e6:8.1f} MB  {os.path.basename(f)}")
shutil.make_archive("/content/mocha_downloads", "zip", OUT)
print("\nZip size:", round(os.path.getsize('/content/mocha_downloads.zip') / 1e6, 1), "MB")
from google.colab import files; files.download("/content/mocha_downloads.zip")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("Mocha_revision_downloads_Colab.ipynb", "w"), indent=1)
print("cells:", len(cells))
