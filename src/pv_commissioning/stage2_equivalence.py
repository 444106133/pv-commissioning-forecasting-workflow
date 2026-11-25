from __future__ import annotations

import numpy as np
import pandas as pd


def _mbb_weights(n_hours: int, block_hours: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    if n_hours <= block_hours:
        idx = rng.integers(0, n_hours, size=(replicates, n_hours))
        w = np.zeros((replicates, n_hours), dtype="float64")
        for b in range(replicates):
            np.add.at(w[b], idx[b], 1.0)
        return w
    n_blocks = int(np.ceil(n_hours / block_hours))
    max_start = n_hours - block_hours
    starts = rng.integers(0, max_start + 1, size=(replicates, n_blocks))
    offsets = np.arange(block_hours)[None, None, :]
    seq = (starts[:, :, None] + offsets).reshape(replicates, -1)[:, :n_hours]
    w = np.zeros((replicates, n_hours), dtype="float64")
    for b in range(replicates):
        np.add.at(w[b], seq[b], 1.0)
    return w


def moving_block_bootstrap_contrasts(
    df_eval: pd.DataFrame,
    y_true_pu: np.ndarray,
    preds_treat_pu: np.ndarray,
    preds_ref_pu: np.ndarray,
    block_hours: int = 168,
    replicates: int = 2000,
    seed: int = 20261005,
) -> dict[str, object]:
    df = df_eval.reset_index(drop=True)
    units = sorted(df["unit"].astype(str).unique())
    hours = sorted(df["timestamp"].unique())
    u_idx = pd.Index(units).get_indexer(df["unit"].astype(str))
    h_idx = pd.Index(hours).get_indexer(df["timestamp"])
    n_u, n_h = len(units), len(hours)

    ae_t = 100.0 * np.abs(np.asarray(preds_treat_pu, dtype="float64") - y_true_pu)
    ae_r = 100.0 * np.abs(np.asarray(preds_ref_pu, dtype="float64") - y_true_pu)
    diff_row = ae_t - ae_r

    sum_mat = np.zeros((n_u, n_h), dtype="float64")
    cnt_mat = np.zeros((n_u, n_h), dtype="float64")
    np.add.at(sum_mat, (u_idx, h_idx), diff_row)
    np.add.at(cnt_mat, (u_idx, h_idx), 1.0)

    unit_deltas = np.sum(sum_mat, axis=1) / np.maximum(np.sum(cnt_mat, axis=1), 1.0)
    delta_pp = float(np.mean(unit_deltas))

    rng = np.random.default_rng(seed)
    w = _mbb_weights(n_h, block_hours=block_hours, replicates=replicates, rng=rng)
    num_b = w @ sum_mat.T
    den_b = np.maximum(w @ cnt_mat.T, 1.0)
    unit_boot = num_b / den_b
    mbb_boot = np.mean(unit_boot, axis=1)

    u_draw = rng.integers(0, n_u, size=(replicates, n_u))
    two_level_boot = np.mean(np.take_along_axis(unit_boot, u_draw, axis=1), axis=1)

    return {
        "delta_nmae_pp": delta_pp,
        "per_unit_delta_pp": {u: float(unit_deltas[i]) for i, u in enumerate(units)},
        "mbb_ci90_pp": [float(np.quantile(mbb_boot, 0.05)), float(np.quantile(mbb_boot, 0.95))],
        "mbb_ci95_pp": [float(np.quantile(mbb_boot, 0.025)), float(np.quantile(mbb_boot, 0.975))],
        "two_level_ci90_pp": [float(np.quantile(two_level_boot, 0.05)), float(np.quantile(two_level_boot, 0.95))],
        "two_level_ci95_pp": [float(np.quantile(two_level_boot, 0.025)), float(np.quantile(two_level_boot, 0.975))],
    }


def evaluate_paired_tost(ci90_pp: list[float], ci95_pp: list[float], delta_bound_pp: float = 0.20) -> str:
    lo90, hi90 = ci90_pp
    lo95, hi95 = ci95_pp
    if lo95 > 0.0:
        if lo90 > -delta_bound_pp and hi90 < delta_bound_pp:
            return "Harm (within equivalence bounds)"
        return "Harm"
    if hi90 < -delta_bound_pp:
        return "Meaningful improvement"
    if lo90 >= -delta_bound_pp and hi90 <= delta_bound_pp:
        return "Equivalence"
    if hi95 < 0.0:
        return "Detectable but small"
    return "Inconclusive"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype="float64")[order]
    q_raw = ranked * m / np.arange(1, m + 1)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_clipped = np.clip(q_monotone, 0.0, 1.0)
    out = np.empty(m, dtype="float64")
    out[order] = q_clipped
    return out.tolist()
