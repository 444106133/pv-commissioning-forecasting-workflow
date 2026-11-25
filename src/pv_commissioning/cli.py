from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .stage1_qualification import FixedTiltRangeGate, build_feature_matrix, compute_physical_poa_features
from .stage2_equivalence import evaluate_paired_tost, moving_block_bootstrap_contrasts
from .stage3_lifecycle import ShrinkageResidualCalibrator


def load_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def run_logo_workflow(df: pd.DataFrame, cfg: dict[str, object]) -> dict[str, object]:
    s1 = cfg.get("stage1_metadata_qualification", {})
    s2 = cfg.get("stage2_equivalence_testing", {})
    s3 = cfg.get("stage3_horizon_and_telemetry", {})
    t_cfg = cfg.get("target", {})

    min_elev = float(t_cfg.get("daylight_elevation_min_deg", 10.0))
    work = df[df["solar_elevation_deg"] > min_elev].reset_index(drop=True)
    if "y_pu" not in work.columns:
        work["y_pu"] = np.clip(
            pd.to_numeric(work["power_kw"], errors="coerce")
            / np.maximum(pd.to_numeric(work["dc_kw"], errors="coerce"), 1e-6),
            0.0,
            1.2,
        )

    units = sorted(work["unit"].astype(str).unique())
    variants = ("G0", "G1-Raw", "G1-NoCap", "G1-Phys", "G1-PhysGate", "G2-PhysGate")
    preds = {v: np.zeros(len(work), dtype="float64") for v in variants}
    seeds = list(s2.get("random_seeds", [17, 29, 43]))

    for held_out in units:
        tr_mask = work["unit"].astype(str) != held_out
        te_mask = ~tr_mask
        tr = work.loc[tr_mask].copy()
        te = work.loc[te_mask].copy()

        train_meta = tr[["unit", "tracking_mode", "tilt_deg"]].drop_duplicates()
        gate = FixedTiltRangeGate.from_training_fleet(train_meta)
        bounds = (gate.beta_min_train, gate.beta_max_train)
        clip_pair = tuple(s1.get("poa_ghi_ratio_clip", [0.0, 3.0]))
        tr_p = compute_physical_poa_features(
            tr,
            tilt_bounds=bounds,
            u0=float(s1.get("faiman_u0", 25.0)),
            u1=float(s1.get("faiman_u1", 6.84)),
            gamma_t=float(s1.get("silicon_temp_coeff_per_c", -0.004)),
            ratio_clip=(float(clip_pair[0]), float(clip_pair[1])),
            min_ghi_denom=float(s1.get("poa_ghi_min_denom_wm2", 10.0)),
        )
        te_p = compute_physical_poa_features(
            te,
            tilt_bounds=bounds,
            u0=float(s1.get("faiman_u0", 25.0)),
            u1=float(s1.get("faiman_u1", 6.84)),
            gamma_t=float(s1.get("silicon_temp_coeff_per_c", -0.004)),
            ratio_clip=(float(clip_pair[0]), float(clip_pair[1])),
            min_ghi_denom=float(s1.get("poa_ghi_min_denom_wm2", 10.0)),
        )

        fold_out: dict[str, np.ndarray] = {}
        for variant in ("G0", "G1-Raw", "G1-NoCap", "G1-Phys", "G2-PhysGate"):
            X_tr = build_feature_matrix(tr_p, variant)
            X_te = build_feature_matrix(te_p, variant).reindex(columns=X_tr.columns, fill_value=0.0)
            seed_preds = []
            for seed in seeds:
                reg = HistGradientBoostingRegressor(max_depth=5, max_iter=150, random_state=int(seed))
                reg.fit(X_tr, tr_p["y_pu"])
                seed_preds.append(np.clip(reg.predict(X_te), 0.0, None))
            fold_out[variant] = np.mean(seed_preds, axis=0)

        for variant in ("G0", "G1-Raw", "G1-NoCap", "G1-Phys"):
            preds[variant][te_mask] = fold_out[variant]
        preds["G1-PhysGate"][te_mask] = gate.combine(
            fold_out["G1-Phys"], fold_out["G0"], te_p["tracking_mode"], te_p["tilt_deg"]
        )
        preds["G2-PhysGate"][te_mask] = gate.combine(
            fold_out["G2-PhysGate"], fold_out["G0"], te_p["tracking_mode"], te_p["tilt_deg"]
        )

    y_true = work["y_pu"].to_numpy(dtype="float64")
    cold_summary = {}
    for variant, p in preds.items():
        per_unit = work.assign(ae=100.0 * np.abs(p - y_true)).groupby("unit")["ae"].mean()
        cold_summary[variant] = {
            "pooled_nmae_pct": float(per_unit.mean()),
            "per_unit_nmae_pct": {str(k): float(v) for k, v in per_unit.items()},
        }

    block_h = int(s2.get("moving_block_bootstrap_hours", 168))
    reps = int(s2.get("bootstrap_replicates_curves", 2000))
    sesoi = float(s2.get("tost_sesoi_pp", 0.20))

    c_stage1 = moving_block_bootstrap_contrasts(
        work, y_true, preds["G1-PhysGate"], preds["G0"], block_hours=block_h, replicates=reps
    )
    c_stage1["decision_two_level"] = evaluate_paired_tost(
        c_stage1["two_level_ci90_pp"], c_stage1["two_level_ci95_pp"], delta_bound_pp=sesoi
    )

    c_stage2 = moving_block_bootstrap_contrasts(
        work, y_true, preds["G2-PhysGate"], preds["G1-PhysGate"], block_hours=block_h, replicates=reps
    )
    c_stage2["decision_two_level"] = evaluate_paired_tost(
        c_stage2["two_level_ci90_pp"], c_stage2["two_level_ci95_pp"], delta_bound_pp=sesoi
    )

    tau = float(s3.get("shrinkage_half_saturation_days", 7.0))
    alpha_ps = float(s3.get("residual_ridge_alpha_per_sample", 0.5))
    days_grid = [int(d) for d in s3.get("commissioning_days", [0, 1, 3, 7, 14, 30, 60, 90])]
    calib_curve: dict[str, float] = {}
    for d_days in days_grid:
        unit_errs = []
        for u in units:
            u_mask = (work["unit"].astype(str) == u).to_numpy()
            sub = work.loc[u_mask].sort_values("timestamp").reset_index(drop=True)
            base_u = preds["G1-PhysGate"][u_mask][sub.index]
            X_u = sub[["nwp_ghi_wm2", "cs_ghi_wm2"]].to_numpy(dtype="float64") / 1000.0
            y_u = sub["y_pu"].to_numpy(dtype="float64")
            mid = len(sub) // 2
            t0 = sub["timestamp"].iloc[0]
            adapt_rows = (sub["timestamp"].iloc[:mid] < (t0 + pd.Timedelta(days=d_days))).to_numpy()
            cal = ShrinkageResidualCalibrator(tau_days=tau, alpha_per_sample=alpha_ps)
            cal.fit(X_u[:mid][adapt_rows], y_u[:mid][adapt_rows], base_u[:mid][adapt_rows], days_accumulated=d_days)
            y_adapt = cal.predict(X_u[mid:], base_u[mid:])
            unit_errs.append(float(100.0 * np.mean(np.abs(y_adapt - y_u[mid:]))))
        calib_curve[f"{d_days}d"] = float(np.mean(unit_errs))

    return {
        "n_units": len(units),
        "n_scored_rows": int(len(work)),
        "cold_start_nmae": cold_summary,
        "contrasts": {
            "G1_PhysGate_minus_G0": c_stage1,
            "G2_PhysGate_minus_G1_PhysGate": c_stage2,
        },
        "lag_free_shrinkage_curve_pct": calib_curve,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pv-commissioning")
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Hourly multi-site CSV or Parquet table")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Workflow JSON configuration")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs"))
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    df = load_table(args.data)
    summary = run_logo_workflow(df, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "workflow_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
