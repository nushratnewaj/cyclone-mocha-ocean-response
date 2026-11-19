import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 4c - recovery-time sensitivity (reviewer item d).

For each variable/source in the wake core:
  departure(t) = X(t) - mean of X over 1-7 May   (X = reference series below; 2023 was already
                 anomalous before the storm, so recovery is measured back to the pre-storm level)
  references   : R1 = anomaly vs 1993-2022 climatology (where a climatology exists)
                 R2 = wake minus control (removes basin-wide weather/seasonal anomalies)
  peak         : largest |departure| between 14 May (landfall) and 31 May
  thresholds   : T1 = 10% of the peak departure (rule applied in the original code)
                 T2 = 1 SD, T3 = 2 SD of natural variability:
                   - climatology-based series: interannual SD (1993-2022) of the same region-mean
                     pentad quantity at that time of year
                   - satellite chl, OSTIA SST, integrated NPP (no multi-year record here): SD of the
                     detrended daily series 15 Apr - 7 May (intraseasonal variability, stricter)
  recovered    : first day after the peak from which |departure| stays <= threshold for >= 7 days
Days are counted from landfall (14 May 2023).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, warnings
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from h5lite import File
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
O2 = OUTB + "/step2"; O3 = OUTB + "/step3"; O = OUTB + "/step4"
LANDFALL = pd.Timestamp("2023-05-14")
dd = pd.read_pickle(f"{O2}/daily_region_series.pkl"); P = pd.read_pickle(f"{O2}/pentad_region_anoms.pkl")
G = np.load(f"{O2}/geometry.npz"); LAT, LON = G["lat"], G["lon"]; REG = G["region"]
WT = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))


def rmean(a, m):
    w = WT * m; ok = np.isfinite(a)
    return np.nansum(np.where(ok, a, 0) * w, axis=(-2, -1)) / np.sum(np.where(ok, w, 0), axis=(-2, -1))


def sd_from_pentads(key, idx):
    sdp = key.loc[1993:2022].std(ddof=1).values
    k = np.minimum((idx - pd.Timestamp("2023-04-01")).days // 5, 35)
    return pd.Series(sdp[k], index=idx)


def sd_intraseasonal(x):
    s = x.loc["2023-04-15":"2023-05-07"].dropna(); t = np.arange(len(s))
    res = s.values - np.polyval(np.polyfit(t, s.values, 1), t)
    return pd.Series(np.std(res, ddof=1), index=x.index)


def recover(x, sd):
    x = x.dropna(); pre = x.loc["2023-05-01":"2023-05-07"].mean(); dep = x - pre
    ev = dep.loc["2023-05-14":"2023-05-31"]; pk = ev.abs().idxmax(); pv = dep[pk]
    out = {"peak_date": pk.date(), "peak_departure": pv}
    after = dep.loc[pk:]
    for name, thr in [("10% of peak", pd.Series(0.10 * abs(pv), index=after.index)), ("1 SD", sd.reindex(after.index)), ("2 SD", 2 * sd.reindex(after.index))]:
        ok = after.abs() <= thr
        run = ok.astype(int).groupby((~ok).cumsum()).cumsum(); hit = run[run >= 7].index
        r = hit[0] - pd.Timedelta(days=6) if len(hit) else None
        out[f"{name}: threshold"] = float(np.nanmedian(thr.values))
        if abs(pv) <= thr.iloc[0]:
            out[f"{name}: days after landfall"] = "not exceeded"
        else:
            out[f"{name}: days after landfall"] = (r - LANDFALL).days if r is not None else f">{(x.index[-1] - LANDFALL).days - 6}"
    out["record_ends"] = x.index[-1].date()
    return out


rows = []
# climatology-based (model / GLORYS)
for v, lab in [("sst", "SST (GLORYS12)"), ("sss", "SSS (GLORYS12)"), ("no3", "NO3 (BGC model)"), ("po4", "PO4 (BGC model)"), ("chl", "Chl (BGC model)"), ("npp", "NPP surface (BGC model)")]:
    a1 = dd[(2023, v, "wake", "anom")]
    rows.append(dict(variable=lab, reference="R1 anomaly vs 1993-2022 climatology", **recover(a1, sd_from_pentads(P[(v, "wake")], a1.index))))
    a2 = dd[(2023, v, "wake", "anom")] - dd[(2023, v, "control", "anom")]
    rows.append(dict(variable=lab, reference="R2 wake minus control", **recover(a2, sd_from_pentads(P[(v, "wake")] - P[(v, "control")], a2.index))))
# satellite chl (wake minus control; also raw wake)
SC = pd.read_csv(f"{O3}/satellite_vs_model_chl_regions.csv", header=[0, 1], index_col=0, parse_dates=True)
for ref, x in [("R2 wake minus control", SC[("satellite", "wake")] - SC[("satellite", "control")]), ("wake level (no reference)", SC[("satellite", "wake")])]:
    rows.append(dict(variable="Chl (satellite OC L4)", reference=ref, **recover(x, sd_intraseasonal(x))))
# integrated NPP 0-100 m (model, 2023 only)
TS = pd.read_pickle(f"{O3}/diag_region_timeseries.pkl")
x = TS[("npp_int_0_100m", "wake")] - TS[("npp_int_0_100m", "control")]
rows.append(dict(variable="NPP 0-100 m integrated (BGC model)", reference="R2 wake minus control", **recover(x, sd_intraseasonal(x))))
# OSTIA SST: 0.05 -> 0.25 deg (5x5 block mean + bilinear), wake minus control
f = File(DATA + "/dl/ostia_sst_2023_15Apr_30Jun.nc")
ol = f.read("latitude"); oo = f.read("longitude"); sst = f.read("analysed_sst"); td = pd.to_datetime(f.read("time"), unit="s", origin="1981-01-01").floor("D")
ny, nx = (sst.shape[1] // 5) * 5, (sst.shape[2] // 5) * 5
cb = np.nanmean(sst[:, :ny, :nx].reshape(len(td), ny // 5, 5, nx // 5, 5), axis=(2, 4))
la = ol[:ny].reshape(-1, 5).mean(1); lo = oo[:nx].reshape(-1, 5).mean(1)
Pg = np.stack(np.meshgrid(LAT, LON, indexing="ij"), -1)
g = np.stack([RegularGridInterpolator((la, lo), cb[t], bounds_error=False, fill_value=None)(Pg) for t in range(len(td))])
g = np.where(G["ocean"], g, np.nan)
xo = pd.Series(rmean(g, REG == 1) - rmean(g, REG == 3), index=td)
rows.append(dict(variable="SST (OSTIA observed)", reference="R2 wake minus control", **recover(xo, sd_intraseasonal(xo))))
xo1 = pd.Series(rmean(g, REG == 1), index=td)
rows.append(dict(variable="SST (OSTIA observed)", reference="wake level (no reference)", **recover(xo1, sd_intraseasonal(xo1))))

T = pd.DataFrame(rows); T.to_csv(f"{O}/recovery_sensitivity.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 40)
cols = ["variable", "reference", "peak_date", "peak_departure", "10% of peak: days after landfall", "1 SD: days after landfall", "2 SD: days after landfall", "record_ends"]
print(T[cols].round(3).to_string(index=False))
