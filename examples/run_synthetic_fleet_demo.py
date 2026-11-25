from __future__ import annotations

import json
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_commissioning.cli import run_logo_workflow


def make_synthetic_fleet(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plants = [
        ("Site_A1", "B", 15.0, 15.0, "fixed", 0.00),
        ("Site_A2", "B", 4500.0, 18.0, "single_axis", +0.06),
        ("Site_C1", "C", 25.0, 18.0, "fixed", -0.01),
        ("Site_C2", "C", 60.0, 20.0, "fixed", 0.00),
        ("Site_D1", "D", 20.0, 15.0, "fixed", -0.01),
        ("Target_LargeUtility", "C", 38000.0, 20.0, "fixed", -0.01),
        ("Target_SteepFixed", "B", 8.0, 35.0, "fixed", -0.01),
    ]
    hours = pd.date_range("2021-04-01", periods=24 * 120, freq="h", tz="UTC")
    hod = hours.hour.to_numpy()
    elev = np.maximum(0.0, 65.0 * np.sin(np.pi * (hod - 6.0) / 12.0))
    day_mask = elev > 10.0
    hours = hours[day_mask]
    elev = elev[day_mask]

    rows = []
    for unit, clim, dc_kw, tilt, mode, site_bias in plants:
        n = len(hours)
        cs_ghi = 1000.0 * np.sin(np.deg2rad(elev))
        gain = 1.0 + 0.012 * (tilt - 15.0) if mode == "fixed" else 1.22
        cs_poa = cs_ghi * gain
        csi = np.clip(rng.normal(0.78 if clim == "B" else 0.68, 0.18, size=n), 0.10, 1.05)
        nwp_ghi = cs_ghi * csi
        temp_c = (28.0 if clim == "B" else (18.0 if clim == "C" else 9.0)) + rng.normal(0.0, 4.0, size=n)
        wind_ms = np.clip(rng.normal(3.2, 1.1, size=n), 0.2, 12.0)
        t_cell = temp_c + (csi * cs_poa) / (25.0 + 6.84 * wind_ms)
        y_true = np.clip(
            (csi * cs_poa / 1000.0) * (1.0 - 0.004 * (t_cell - 25.0)) + site_bias + rng.normal(0.0, 0.03, size=n),
            0.0,
            1.0,
        )
        rows.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "timestamp": hours,
                    "climate_class": clim,
                    "dc_kw": dc_kw,
                    "tilt_deg": tilt,
                    "azimuth_deg": 180.0,
                    "tracking_mode": mode,
                    "solar_elevation_deg": elev,
                    "solar_azimuth_deg": 180.0,
                    "cs_ghi_wm2": cs_ghi,
                    "cs_poa_wm2": cs_poa,
                    "nwp_ghi_wm2": nwp_ghi,
                    "nwp_csi": csi,
                    "nwp_cloud_frac": 1.0 - np.clip(csi, 0.0, 1.0),
                    "nwp_temp_c": temp_c,
                    "nwp_wind_ms": wind_ms,
                    "nwp_lead_min": 510.0,
                    "y_pu": y_true,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cfg_path = ROOT / "configs" / "workflow_config_template.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["stage2_equivalence_testing"]["bootstrap_replicates_curves"] = 500
    cfg["stage2_equivalence_testing"]["random_seeds"] = [17]
    df = make_synthetic_fleet()
    summary = run_logo_workflow(df, cfg)

    for variant, stats in summary["cold_start_nmae"].items():
        u_large = stats["per_unit_nmae_pct"]["Target_LargeUtility"]
        print(f"{variant:12s}  mean_nmae={stats['pooled_nmae_pct']:.2f}%  large_utility={u_large:.2f}%")

    c2 = summary["contrasts"]["G2_PhysGate_minus_G1_PhysGate"]
    print(
        f"G2-PhysGate vs G1-PhysGate: {c2['delta_nmae_pp']:+.3f} pp "
        f"90% CI {c2['two_level_ci90_pp']} ({c2['decision_two_level']})"
    )
    print("Lag-free calibration curve (%):", summary["lag_free_shrinkage_curve_pct"])


if __name__ == "__main__":
    main()
