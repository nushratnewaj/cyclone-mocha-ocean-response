import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import DATA, OUTB  # noqa
"""
Step 2b metrics - is the 2023 change unusual compared with 33 seasons?
Pentads: p6 = 1-5 May, p7 = 6-10 May, p8 = 11-15 May, p9 = 16-20 May, p10 = 21-25 May,
         p17 = 25-29 Jun, p18 = 30 Jun-4 Jul, p24-29 = 30 Jul-28 Aug.
Event change      : anomaly(p9) - anomaly(p6)   (seasonal cycle already removed)
Wake excess       : event change in wake core - event change in control
Late-season state : mean anomaly p17-p18 (25 Jun-4 Jul) and p24-p29 (Aug)
Each 2023 value is placed against 1993-2022 (mean, SD, z) and ranked among all 33 years.
"""
import numpy as np, pandas as pd
O = OUTB + "/step2"
P = pd.read_pickle(f"{O}/pentad_region_anoms.pkl")
QUIET = [1993, 1995, 2000, 2001, 2005, 2012, 2014, 2015, 2025]
VARS = ["sst", "sss", "mld", "no3", "po4", "chl", "npp"]
UNIT = {"sst": "degC", "sss": "psu", "mld": "m", "no3": "mmol m-3", "po4": "mmol m-3", "chl": "mg m-3", "npp": "mg C m-3 d-1"}
rows = []


def place(series, label, var, region):
    ref = series.loc[1993:2022]
    v23 = series.loc[2023]
    z = (v23 - ref.mean()) / ref.std(ddof=1)
    # rank: 1 = most extreme in the direction of the 2023 value
    rank = int((series * np.sign(v23 - ref.mean()) >= v23 * np.sign(v23 - ref.mean())).sum())
    q = series.loc[[y for y in QUIET if y in series.index]]
    rows.append(dict(metric=label, variable=var, unit=UNIT[var], region=region, value_2023=v23,
                     clim_mean_1993_2022=ref.mean(), clim_sd=ref.std(ddof=1), z_2023=z,
                     rank_of_33=f"{rank}/{series.notna().sum()}", quiet_years_min=q.min(), quiet_years_max=q.max(),
                     value_2024=series.get(2024, np.nan), value_2025=series.get(2025, np.nan)))


for v in VARS:
    for reg in ["wake", "outer", "control", "shelf", "open_all"]:
        A = P[(v, reg)]
        place(A[9] - A[6], "event change p9-p6 (16-20 May minus 1-5 May)", v, reg)
        place(A[[17, 18]].mean(1), "late-June anomaly (25 Jun-4 Jul)", v, reg)
        place(A[list(range(24, 30))].mean(1), "August anomaly (30 Jul-28 Aug)", v, reg)
    W, C = P[(v, "wake")], P[(v, "control")]
    place((W[9] - W[6]) - (C[9] - C[6]), "WAKE EXCESS: event change wake minus control", v, "wake-control")
    place(W[[17, 18]].mean(1) - C[[17, 18]].mean(1), "late-June wake minus control", v, "wake-control")
    R, L = P[(v, "wake_right")], P[(v, "wake_left")]
    place((R[9] - R[6]) - (L[9] - L[6]), "event change right minus left of track", v, "right-left")

M = pd.DataFrame(rows)
M.to_csv(f"{O}/step2_metrics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
show = M[M.region.isin(["wake", "control", "shelf", "wake-control", "right-left"])].copy()
for c in ["value_2023", "clim_mean_1993_2022", "clim_sd", "z_2023", "quiet_years_min", "quiet_years_max", "value_2024", "value_2025"]:
    show[c] = show[c].map(lambda x: f"{x:.4g}")
print(show[["metric", "variable", "region", "value_2023", "clim_mean_1993_2022", "clim_sd", "z_2023", "rank_of_33", "quiet_years_min", "quiet_years_max"]].to_string(index=False))
