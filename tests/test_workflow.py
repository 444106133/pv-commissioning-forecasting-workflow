from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_commissioning import (
    FixedTiltRangeGate,
    ShrinkageResidualCalibrator,
    benjamini_hochberg,
    build_feature_matrix,
    compute_physical_poa_features,
    evaluate_paired_tost,
    select_causal_nwp_step,
)
from pv_commissioning.cli import main as cli_main


class TestCommissioningWorkflow(unittest.TestCase):
    def test_capacity_pruning_and_physical_bounds(self) -> None:
        df = pd.DataFrame(
            {
                "dc_kw": [10.0, 38000.0],
                "tilt_deg": [15.0, 35.0],
                "azimuth_deg": [180.0, 180.0],
                "tracking_mode": ["fixed", "fixed"],
                "solar_elevation_deg": [45.0, 25.0],
                "solar_azimuth_deg": [180.0, 180.0],
                "cs_ghi_wm2": [700.0, 300.0],
                "cs_poa_wm2": [750.0, 1200.0],
                "nwp_ghi_wm2": [600.0, 250.0],
                "nwp_csi": [0.85, 0.80],
                "nwp_cloud_frac": [0.2, 0.2],
                "nwp_temp_c": [25.0, 30.0],
                "nwp_wind_ms": [3.0, 2.0],
                "nwp_lead_min": [480.0, 480.0],
            }
        )
        df_phys = compute_physical_poa_features(df, tilt_bounds=(2.0, 20.0))
        self.assertTrue((df_phys["x_poa_ghi_ratio"] <= 3.0).all())
        self.assertTrue((df_phys["x_phys_poa_pu"] >= 0.0).all())

        X_raw = build_feature_matrix(df_phys, "G1-Raw")
        X_phys = build_feature_matrix(df_phys, "G1-Phys")
        self.assertIn("log10_dc_kw", X_raw.columns)
        self.assertNotIn("log10_dc_kw", X_phys.columns)
        self.assertEqual(float(X_phys["tilt_deg"].iloc[1]), 20.0)

    def test_fixed_tilt_range_gate_abstains_out_of_range(self) -> None:
        train_meta = pd.DataFrame(
            {"tracking_mode": ["fixed", "fixed", "dual"], "tilt_deg": [5.0, 20.0, np.nan]}
        )
        gate = FixedTiltRangeGate.from_training_fleet(train_meta)
        modes = pd.Series(["fixed", "fixed", "single_axis"])
        tilts = pd.Series([15.0, 35.0, np.nan])
        combined = gate.combine(
            y_hat_phys=np.array([0.50, 0.80, 0.60]),
            y_hat_g0=np.array([0.40, 0.45, 0.55]),
            tracking_mode=modes,
            tilt_deg=tilts,
        )
        np.testing.assert_allclose(combined, np.array([0.50, 0.45, 0.60]))

    def test_tost_and_fdr_rules(self) -> None:
        self.assertEqual(evaluate_paired_tost([-0.14, -0.02], [-0.16, -0.01], 0.20), "Equivalence")
        self.assertEqual(evaluate_paired_tost([-0.39, -0.26], [-0.41, -0.25], 0.20), "Meaningful improvement")
        self.assertEqual(evaluate_paired_tost([-0.01, +0.57], [-0.05, +0.62], 0.20), "Inconclusive")
        q = benjamini_hochberg([0.001, 0.02, 0.50])
        self.assertTrue(q[0] < q[1] < q[2])

    def test_causal_nwp_and_shrinkage_calibration(self) -> None:
        t_target = pd.Timestamp("2021-06-15 12:00:00Z")
        cycles = pd.DataFrame(
            {
                "cycle_time": [
                    pd.Timestamp("2021-06-14 06:00:00Z"),
                    pd.Timestamp("2021-06-15 00:00:00Z"),
                    pd.Timestamp("2021-06-15 06:00:00Z"),
                ],
                "valid_time": [
                    pd.Timestamp("2021-06-14 12:00:00Z"),
                    pd.Timestamp("2021-06-15 12:00:00Z"),
                    pd.Timestamp("2021-06-15 12:00:00Z"),
                ],
                "cs_ghi_wm2": [850.0, 850.0, 850.0],
            }
        )
        sel_1h = select_causal_nwp_step(t_target, horizon_hours=1, available_cycles=cycles)
        self.assertEqual(sel_1h["mode"], "native")
        self.assertEqual(sel_1h["causal_lead_hours"], 6.0)

        sel_3h = select_causal_nwp_step(t_target, horizon_hours=3, available_cycles=cycles)
        self.assertEqual(sel_3h["mode"], "native")
        self.assertEqual(sel_3h["causal_lead_hours"], 12.0)

        sel_24h = select_causal_nwp_step(t_target, horizon_hours=24, available_cycles=cycles)
        self.assertEqual(sel_24h["mode"], "diurnal_24h_fallback")

        cal = ShrinkageResidualCalibrator(tau_days=7.0, alpha_per_sample=0.5)
        self.assertEqual(cal.weight(0), 0.0)
        self.assertAlmostEqual(cal.weight(7), 0.5)

    def test_cli_runner(self) -> None:
        hours = pd.date_range("2021-06-01 08:00:00Z", periods=48, freq="h")
        rows = []
        for unit, tilt in (("Site_1", 15.0), ("Site_2", 20.0)):
            rows.append(
                pd.DataFrame(
                    {
                        "unit": unit,
                        "timestamp": hours,
                        "climate_class": "B",
                        "dc_kw": 10.0,
                        "tilt_deg": tilt,
                        "azimuth_deg": 180.0,
                        "tracking_mode": "fixed",
                        "solar_elevation_deg": 35.0,
                        "solar_azimuth_deg": 180.0,
                        "cs_ghi_wm2": 700.0,
                        "cs_poa_wm2": 740.0,
                        "nwp_ghi_wm2": 620.0,
                        "nwp_csi": 0.88,
                        "nwp_cloud_frac": 0.12,
                        "nwp_temp_c": 26.0,
                        "nwp_wind_ms": 3.0,
                        "nwp_lead_min": 480.0,
                        "y_pu": 0.58,
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            csv_path = tmp_dir / "fleet.csv"
            cfg_path = tmp_dir / "cfg.json"
            out_dir = tmp_dir / "out"
            df.to_csv(csv_path, index=False)
            cfg = json.loads((ROOT / "configs" / "workflow_config_template.json").read_text(encoding="utf-8"))
            cfg["stage2_equivalence_testing"]["bootstrap_replicates_curves"] = 50
            cfg["stage2_equivalence_testing"]["random_seeds"] = [17]
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            cli_main(["--data", str(csv_path), "--config", str(cfg_path), "--output-dir", str(out_dir)])
            self.assertTrue((out_dir / "workflow_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
