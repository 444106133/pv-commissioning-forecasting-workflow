from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def select_causal_nwp_step(
    target_hour: pd.Timestamp,
    horizon_hours: int,
    available_cycles: pd.DataFrame,
    pub_delay_hours: int = 6,
    max_archive_step_hours: int = 12,
    daylight_cs_min_wm2: float = 50.0,
) -> dict[str, object]:
    t_issue = target_hour - pd.Timedelta(hours=horizon_hours - 1)
    legal = available_cycles[
        available_cycles["cycle_time"] + pd.Timedelta(hours=pub_delay_hours) <= t_issue
    ].copy()
    if legal.empty:
        raise ValueError("No causally available NWP cycles prior to issue time.")

    native = legal[
        (legal["valid_time"] == target_hour)
        & ((legal["valid_time"] - legal["cycle_time"]) <= pd.Timedelta(hours=max_archive_step_hours))
    ]
    if not native.empty:
        chosen = native.sort_values("cycle_time").iloc[-1]
        lead_h = (target_hour - chosen["cycle_time"]).total_seconds() / 3600.0
        return {"mode": "native", "row": chosen, "causal_lead_hours": lead_h, "t_issue": t_issue}

    if horizon_hours < 24:
        t_lag = target_hour - pd.Timedelta(hours=horizon_hours)
        same_day = legal[(legal["valid_time"] == t_lag) & (legal["cs_ghi_wm2"] >= daylight_cs_min_wm2)]
        if not same_day.empty:
            chosen = same_day.sort_values("cycle_time").iloc[-1]
            lead_h = (target_hour - chosen["cycle_time"]).total_seconds() / 3600.0
            return {"mode": "sameday_daylight_lag", "row": chosen, "causal_lead_hours": lead_h, "t_issue": t_issue}

    solar_back_h = 48 if horizon_hours >= 48 else 24
    t_diurnal = target_hour - pd.Timedelta(hours=solar_back_h)
    diurnal = legal[legal["valid_time"] == t_diurnal]
    chosen = (diurnal if not diurnal.empty else legal).sort_values("cycle_time").iloc[-1]
    lead_h = (target_hour - chosen["cycle_time"]).total_seconds() / 3600.0
    return {"mode": f"diurnal_{solar_back_h}h_fallback", "row": chosen, "causal_lead_hours": lead_h, "t_issue": t_issue}


@dataclass
class ShrinkageResidualCalibrator:
    tau_days: float = 7.0
    alpha_per_sample: float = 0.5
    _model: Ridge | None = None
    _days_fitted: float = 0.0

    def weight(self, days: float | None = None) -> float:
        d = self._days_fitted if days is None else float(days)
        if d <= 0.0:
            return 0.0
        return float(d / (d + self.tau_days))

    def fit(
        self,
        X_calib: np.ndarray,
        y_true_pu: np.ndarray,
        y_transfer_pu: np.ndarray,
        days_accumulated: float,
    ) -> ShrinkageResidualCalibrator:
        self._days_fitted = float(days_accumulated)
        if days_accumulated <= 0.0 or len(y_true_pu) == 0:
            self._model = None
            return self
        residuals = np.asarray(y_true_pu, dtype="float64") - np.asarray(y_transfer_pu, dtype="float64")
        alpha = max(self.alpha_per_sample * len(residuals), 1e-3)
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(np.asarray(X_calib, dtype="float64"), residuals)
        self._model = reg
        return self

    def predict(self, X_test: np.ndarray, y_transfer_pu: np.ndarray) -> np.ndarray:
        base = np.asarray(y_transfer_pu, dtype="float64")
        w = self.weight()
        if self._model is None or w <= 0.0:
            return np.clip(base, 0.0, None)
        r_hat = self._model.predict(np.asarray(X_test, dtype="float64"))
        return np.clip(base + w * r_hat, 0.0, None)
