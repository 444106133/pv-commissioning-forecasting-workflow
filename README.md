# Cross-Site PV Commissioning Forecasting and Early-Telemetry Calibration

Reference code and workflow configurations for *"Forecasting Newly Commissioned Photovoltaic Plants with Range-Gated Physical Features and Early Telemetry"* (A. M. Bin Dayil, Y. Bin Salamah, and E. A. Al-Ammar, Department of Electrical Engineering, King Saud University).

Newly commissioned photovoltaic (PV) plants enter grid service without local multi-season power records. Cross-site transfer from existing fleets fills that cold-start gap, although axis-aligned tree models extrapolate poorly when a target array's DC capacity or fixed tilt sits outside the training fleet's range. Across retrospective simulations on 14 public site groups spanning Australia, Europe, and North America (63 primary and 70 total PV configurations from `1 kW` to `38.7 MW`), `stage1_qualification.py` drops raw log-capacity inputs (`G1-NoCap`), converts site geometry into range-clipped plane-of-array (POA) irradiance gain (`x_poa_ghi_ratio` clipped to `[0, 3]`) and Faiman thermally derated per-unit power (`G1-Phys`), and applies a pre-issue fixed-tilt support gate (`G1-PhysGate`) that reverts to weather-only transfer (`G0`) whenever a documented fixed tilt falls outside `[beta_min_train, beta_max_train]`.

Once plant metadata are physically qualified, `stage2_equivalence.py` evaluates cross-climate fleet pooling across Arid (`B`), Temperate (`C`), and Cold (`D`) regimes under leave-one-group-out (LOGO) splits with 8-day chronological buffers, pairing 168-h moving-block and two-level (site-group plus 168-h block) bootstrap intervals against `+/- 0.20` percentage-point Two One-Sided Tests (TOST) bounds and Benjamini-Hochberg false discovery rate control (`q = 0.05`). Longer forecast horizons need strict issue-time causality. In `stage3_lifecycle.py`, `select_causal_nwp_step` enforces `t_cycle + 6 h <= t_issue(dh)` across `1-48 h` Global Forecast System (GFS) leads with solar-hour fallback when archived forecast steps end at `f012`, while `ShrinkageResidualCalibrator` separates Day `0-90` live target-lag activation from zero-intercept irradiance-scaled Ridge residual calibration blended by `w(D) = D / (D + tau)` (`tau = 7` operating days).

## Repository Layout

```text
src/pv_commissioning/
  stage1_qualification.py   POA irradiance gain, Faiman cell temperature, and fixed-tilt range gate
  stage2_equivalence.py     168-h moving-block bootstrap, two-level bootstrap, TOST, and BH-FDR
  stage3_lifecycle.py       Issue-time causal GFS step selection and shrinkage Ridge calibration
  cli.py                    Cross-site LOGO and commissioning evaluation runner
configs/
  workflow_config_template.json   Split buffers, physical constants, TOST bounds, and shrinkage settings
examples/
  run_synthetic_fleet_demo.py     Multi-site cross-climate example with out-of-range target plants
tests/
  test_workflow.py                Regression tests for feature bounds, range gating, TOST, and CLI output
environment.yml                   Conda environment specification
```

## Data Sources and Input Format

Public array telemetry and weather archives referenced in the paper are available directly from the [DKASC Alice Springs portal](https://dkasolarcentre.com.au/download?location=alice-springs), the [DOE OEDI PVDAQ archive](https://doi.org/10.25984/1846021), the [Coopernico Portugal fleet archive](https://doi.org/10.17632/dbh93b6vp8.3), the [UNISOLAR campus dataset](https://doi.org/10.1109/HSI55341.2022.9869474), the [FMI outdoor solar labs](https://doi.org/10.1002/gdj3.70039), the [NOAA 0.25-deg GFS archive](https://doi.org/10.5065/D65D8PWK), and the [1-km Koppen-Geiger maps](https://doi.org/10.1038/s41597-023-02549-6). Prepare a combined hourly table (`.parquet` or `.csv`) matching the fields below:

| Column | Units | Notes |
|---|---|---|
| `timestamp` | UTC ISO-8601 | Hourly interval start timestamp |
| `unit` | String | Site-group identifier keeping co-located arrays in one LOGO fold |
| `climate_class` | String | Broad (`B`, `C`, `D`) or sub-class Koppen-Geiger label |
| `dc_kw` | `kW` | Nameplate DC capacity (`P_DC`) used for target normalization |
| `power_kw` or `y_pu` | `kW` or `[0, 1.2]` | Measured hourly active power or pre-normalized per-unit output |
| `tracking_mode` | String | `fixed`, `single_axis`, or `dual` |
| `tilt_deg`, `azimuth_deg` | `deg` | Documented array tilt and azimuth (`NaN` when undocumented) |
| `solar_elevation_deg`, `solar_azimuth_deg` | `deg` | Interval-midpoint solar geometry (scored when `> 10 deg`) |
| `cs_ghi_wm2`, `cs_poa_wm2` | `W/m^2` | Clear-sky horizontal and transposed plane-of-array irradiance |
| `nwp_ghi_wm2`, `nwp_csi`, `nwp_cloud_frac` | `W/m^2`, `-`, `[0, 1]` | Causal NWP irradiance, clear-sky index, and cloud cover |
| `nwp_temp_c`, `nwp_wind_ms`, `nwp_lead_min` | `deg C`, `m/s`, `min` | Causal NWP 2-m air temperature, 10-m wind speed, and cycle lead |

## Installation and Execution

Requires Python 3.11 or newer.

```bash
pip install -e .

pv-commissioning \
  --data /path/to/hourly_fleet.parquet \
  --config configs/workflow_config_template.json \
  --output-dir outputs
```

Running the CLI writes `outputs/workflow_summary.json` containing per-unit and pooled LOGO cold-start `nMAE` across `G0` through `G2-PhysGate`, paired 168-h block and two-level bootstrap TOST intervals, and the Day `0-90` shrinkage calibration curve. To verify the installation without external archives, run `python examples/run_synthetic_fleet_demo.py` and `python -m unittest discover -s tests`.

## License

MIT License (`LICENSE`).
