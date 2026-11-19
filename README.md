# Cyclone Mocha (2023): upper-ocean and biogeochemical response in the Bay of Bengal

Code for the paper:

> Newaj, N. B., Sarder, S. R., Musa, M. A., Sarah, A. S., Trisha, I. J. M., Sowrav, S. F. F., & Rahman, S. M. M. *Transient upper-ocean and biogeochemical response to Cyclone Mocha (2023) in the Bay of Bengal: separating the storm signal from the pre-monsoon seasonal cycle.* Journal of Earth and Ocean Science (under review).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Code:** https://github.com/[username]/[repository]
**Archived code and processed data (Zenodo):** https://doi.org/10.5281/zenodo.XXXXXXX

---

## What this repository does

The scripts separate the ocean response to Cyclone Mocha (9–15 May 2023) from the pre-monsoon seasonal cycle. They:

- build a 1993–2022 daily climatology and compare 2023 with 33 pre-monsoon seasons (1993–2025);
- define track-centred analysis regions (wake core, outer wake, control and shelf);
- compute the cold wake from observed (OSTIA) and reanalysis (GLORYS12) SST;
- diagnose mixed-layer depth, barrier layer, profiles, ERA5 wind stress and Ekman pumping;
- close a surface nutrient budget and test the biogeochemical model against satellite chlorophyll and the World Ocean Atlas 2023;
- estimate recovery times for several thresholds and test correlations and lags with autocorrelation-aware statistics;
- produce every figure and table of the paper.

## Repository layout

```
scripts/      analysis scripts, run in the order below
notebooks/    Google Colab notebook that downloads and pre-processes the multi-year data
requirements.txt
LICENSE
```

The input data, gridded intermediates and all results are not stored on GitHub because of their size. Download them from Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX) and unpack them next to this repository:

```
<root>/Mocha_code/                  this repository
<root>/Mocha_processed_data/data/   inputs and gridded intermediates
<root>/Mocha_processed_data/out/    results
```

The scripts find the data through `scripts/paths.py`. To use other folders, set the environment variables `MOCHA_DATA` and `MOCHA_OUT`.

## Requirements

- Python 3.10 or later
- `pip install -r requirements.txt` (numpy, pandas, scipy, matplotlib, openpyxl)
- The system HDF5 library, used by `scripts/h5lite.py` to read NetCDF files (for example `sudo apt install libhdf5-103` or `conda install hdf5`). If it is not found automatically, set `HDF5_LIB` to its path.

The Colab notebook installs its own packages (xarray, netCDF4, gsw, copernicusmarine, cdsapi). It needs a free Copernicus Marine account and a Copernicus Climate Data Store API key.

## Run order

Run each script from `scripts/` with `python <script>`. All outputs are already provided on Zenodo, so any step can be rerun on its own.

| Step | Script | Purpose | Paper |
|---|---|---|---|
| 0 (optional) | `notebooks/Mocha_revision_downloads_Colab.ipynb` | Download data; build climatology, 5-day means, profiles and ERA5 forcing (several hours) | Section 2.2 |
| 1 | `s1_pipeline.py`, `s1_audit.py`, `s1_corrected.py` | Reproduce and audit the basin-mean series of the earlier analysis | Response to reviewers |
| 2 | `s2_geometry.py` | Track geometry, analysis regions, shelf mask | Section 2.3, Fig. 1 |
| 2 | `s2_anomalies.py` | Anomalies relative to the 1993–2022 climatology | Section 2.4 |
| 2 | `s2_metrics.py` | Wake excess, z-scores and ranks among 33 seasons | Table 2, Fig. 4 |
| 2 | `s2_wake.py` | Cold-wake metric (OSTIA, GLORYS12) | Section 2.5, Fig. 3 |
| 2 | `s2_recovery_wc.py`, `s2_map.py`, `s2_figs.py` | Preliminary estimates and working figures | – |
| 3 | `s3_profiles.py` | MLD, ILD, barrier layer, composite profiles | Section 3.5, Figs 9–10 |
| 3 | `s3_budget.py` | Surface nutrient budget | Section 3.6, Fig. 11, Table A1 |
| 3 | `s3_woa_ekman.py` | Comparison with WOA23; Ekman pumping | Sections 3.1, 3.7, Fig. 12 |
| 3 | `s3_satellite.py` | Satellite versus model chlorophyll | Section 3.4, Fig. 8 |
| 4 | `s4_stats.py` | Correlations with effective sample size, lag correlation with surrogates | Section 3.9, Figs A1–A2 |
| 4 | `s4_recovery.py` | Recovery times for three thresholds and two references | Section 3.8, Table 3 |
| Figures | `final_figs.py <dpi> [ids]` | All paper figures, e.g. `python final_figs.py 800` or `python final_figs.py 300 01 11` | Figs 1–12, A1–A2 |

## Data sources

Copernicus Marine Service: GLORYS12 reanalysis (GLOBAL_MULTIYEAR_PHY_001_030, https://doi.org/10.48670/moi-00021), global biogeochemical hindcast (GLOBAL_MULTIYEAR_BGC_001_029, https://doi.org/10.48670/moi-00019), OSTIA reprocessed SST (https://doi.org/10.48670/moi-00168) and Copernicus-GlobColour chlorophyll (https://doi.org/10.48670/moi-00281). Also used: ERA5 (https://doi.org/10.24381/cds.adbb2d47), IBTrACS v04r01 (https://doi.org/10.25921/82ty-9e16), World Ocean Atlas 2023 and GEBCO_2025 (https://doi.org/10.5285/37c52e96-24ea-67ce-e063-7086abc05f29).

This work uses E.U. Copernicus Marine Service Information and modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use of this information.

## Licence

Code: MIT (see `LICENSE`). Processed data on Zenodo: CC BY 4.0.

## How to cite

If you use this code or data, please cite the paper and the Zenodo archive:

> Newaj, N. B., et al. (2026). Code and processed data for: Transient upper-ocean and biogeochemical response to Cyclone Mocha (2023) in the Bay of Bengal (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## Contact

Sheikh Fahim Faysal Sowrav, Department of Oceanography and Hydrography, Bangladesh Maritime University, Dhaka, Bangladesh (sowrav.ocn@bmu.edu.bd)
