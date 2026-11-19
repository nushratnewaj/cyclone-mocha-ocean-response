import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""Recovery of the wake signal measured as wake-minus-control (removes basin-wide anomalies such as the
2023 warm year). Threshold: 1 SD of the 1993-2022 interannual wake-minus-control pentad differences.
Recovered = first day after the storm extreme when |diff - pre-storm diff| <= 1 SD for >= 7 consecutive days."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
O = OUTB + "/step2"
dd = pd.read_pickle(f"{O}/daily_region_series.pkl"); P = pd.read_pickle(f"{O}/pentad_region_anoms.pkl")
rows = []
for v in ["sst", "sss", "no3", "po4", "chl", "npp"]:
    d = (dd[(2023, v, "wake", "anom")] - dd[(2023, v, "control", "anom")]).dropna()
    sdp = (P[(v, "wake")] - P[(v, "control")]).loc[1993:2022].std(ddof=1).values
    sd = pd.Series(sdp[np.minimum((d.index - pd.Timestamp("2023-04-01")).days // 5, 35)], index=d.index)
    pre = d.loc["2023-05-01":"2023-05-07"].mean(); ev = d.loc["2023-05-09":"2023-05-31"]; ke = (ev - pre).abs().idxmax()
    ok = (d.loc[ke:] - pre).abs() <= sd.loc[ke:]
    run = ok.astype(int).groupby((~ok).cumsum()).cumsum(); hit = run[run >= 7].index
    r = (hit[0] - pd.Timedelta(days=6)) if len(hit) else None
    rows.append(dict(variable=v, peak_date=ke.date(), wake_minus_control_change=d[ke] - pre, sd_threshold=sd[ke],
                     recovered=r.date() if r is not None else f"not by {d.index[-1].date()}",
                     days_after_landfall=(r - pd.Timestamp("2023-05-14")).days if r is not None else None))
R = pd.DataFrame(rows); R.to_csv(f"{O}/step2_recovery_wake_minus_control.csv", index=False); print(R.round(4).to_string(index=False))
