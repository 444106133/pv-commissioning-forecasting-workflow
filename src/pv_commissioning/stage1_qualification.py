from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


WEATHER_COLS = [
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "cs_ghi_wm2",
    "nwp_ghi_wm2",
    "nwp_csi",
    "nwp_cloud_frac",
    "nwp_temp_c",
    "nwp_wind_ms",
    "nwp_lead_min",
]


def compute_physical_poa_features(
    df: pd.DataFrame,
    tilt_bounds: tuple[float, float] | None = None,
    u0: float = 25.0,
    u1: float = 6.84,
    gamma_t: float = -0.004,
    ratio_clip: tuple[float, float] = (0.0, 3.0),
    min_ghi_denom: float = 10.0,
) -> pd.DataFrame:
    out = df.copy()
    if tilt_bounds is not None and "tilt_deg" in out.columns:
        b_min, b_max = tilt_bounds
        modes = out.get("tracking_mode", pd.Series("fixed", index=out.index)).astype(str)
        raw_tilt = pd.to_numeric(out["tilt_deg"], errors="coerce")
        out["tilt_clipped_deg"] = np.where(
            (modes == "fixed") & raw_tilt.notna(),
            np.clip(raw_tilt, b_min, b_max),
            raw_tilt,
        )

    poa = pd.to_numeric(out["cs_poa_wm2"], errors="coerce").to_numpy(dtype="float64")
    ghi = pd.to_numeric(out["cs_ghi_wm2"], errors="coerce").to_numpy(dtype="float64")
    csi = pd.to_numeric(out["nwp_csi"], errors="coerce").to_numpy(dtype="float64")
    t_air = pd.to_numeric(out["nwp_temp_c"], errors="coerce").to_numpy(dtype="float64")
    wind = pd.to_numeric(out["nwp_wind_ms"], errors="coerce").to_numpy(dtype="float64")

    valid_poa = np.isfinite(poa) & np.isfinite(ghi) & (ghi > 0.0)
    denom = np.maximum(ghi, min_ghi_denom)
    lo, hi = ratio_clip
    with np.errstate(divide="ignore", invalid="ignore"):
        poa_ratio = np.where(valid_poa, np.clip(poa / denom, lo, hi), np.nan)
        e_poa = csi * poa
        w_eff = np.where(np.isfinite(wind), np.maximum(wind, 0.0), 0.0)
        t_cell = t_air + e_poa / (u0 + u1 * w_eff)
        phys_pu = (e_poa / 1000.0) * (1.0 + gamma_t * (t_cell - 25.0))
        valid_phys = np.isfinite(e_poa) & np.isfinite(t_cell)
        phys_pu = np.where(valid_phys, np.where(ghi <= 0.0, 0.0, np.maximum(phys_pu, 0.0)), np.nan)

    out["x_poa_ghi_ratio"] = poa_ratio
    out["x_phys_poa_pu"] = phys_pu
    return out


@dataclass
class FixedTiltRangeGate:
    beta_min_train: float
    beta_max_train: float

    @classmethod
    def from_training_fleet(cls, train_meta: pd.DataFrame) -> FixedTiltRangeGate:
        fixed = train_meta[
            (train_meta["tracking_mode"].astype(str) == "fixed") & train_meta["tilt_deg"].notna()
        ]["tilt_deg"].astype(float)
        if fixed.empty:
            return cls(beta_min_train=0.0, beta_max_train=90.0)
        return cls(beta_min_train=float(fixed.min()), beta_max_train=float(fixed.max()))

    def indicator(self, tracking_mode: pd.Series, tilt_deg: pd.Series) -> np.ndarray:
        tilt = pd.to_numeric(tilt_deg, errors="coerce").to_numpy(dtype="float64")
        is_doc_fixed = (tracking_mode.astype(str).to_numpy() == "fixed") & np.isfinite(tilt)
        out_of_range = is_doc_fixed & ((tilt < self.beta_min_train) | (tilt > self.beta_max_train))
        return np.where(out_of_range, 0.0, 1.0)

    def combine(
        self,
        y_hat_phys: np.ndarray,
        y_hat_g0: np.ndarray,
        tracking_mode: pd.Series,
        tilt_deg: pd.Series,
    ) -> np.ndarray:
        g = self.indicator(tracking_mode, tilt_deg)
        return g * np.asarray(y_hat_phys, dtype="float64") + (1.0 - g) * np.asarray(y_hat_g0, dtype="float64")


def build_feature_matrix(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    cols = list(WEATHER_COLS)
    work = df.copy()
    tilt_col = "tilt_clipped_deg" if (
        variant in {"G1-Phys", "G1-PhysGate", "G2-PhysGate", "G2-ContPhys"} and "tilt_clipped_deg" in work.columns
    ) else "tilt_deg"

    if variant in {"G1-Raw", "G2"}:
        work["log10_dc_kw"] = np.log10(np.maximum(pd.to_numeric(work["dc_kw"], errors="coerce"), 1e-3))
        cols += ["log10_dc_kw", "tilt_deg", "azimuth_deg", "cs_poa_wm2"]
    elif variant in {"G1-NoCap", "G1-Phys", "G1-PhysGate", "G2-PhysGate", "G2-ContPhys"}:
        cols += [tilt_col, "azimuth_deg", "cs_poa_wm2"]

    if variant in {"G1-Phys", "G1-PhysGate", "G2-PhysGate", "G2-ContPhys"}:
        cols += ["x_poa_ghi_ratio", "x_phys_poa_pu"]

    if variant in {"G2", "G2-PhysGate"} and "climate_class" in work.columns:
        dummies = pd.get_dummies(work["climate_class"].astype(str), prefix="kg", dtype="float64")
        work = pd.concat([work, dummies], axis=1)
        cols += list(dummies.columns)

    if variant == "G2-ContPhys":
        for c in ("clim_kt", "clim_temp_mean", "clim_diurnal", "clim_seasonality", "clim_cloud"):
            if c in work.columns:
                cols.append(c)

    out = work[[c for c in cols if c in work.columns]].apply(pd.to_numeric, errors="coerce")
    if tilt_col == "tilt_clipped_deg" and "tilt_clipped_deg" in out.columns:
        out = out.rename(columns={"tilt_clipped_deg": "tilt_deg"})
    return out
