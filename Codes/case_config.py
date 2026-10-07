"""Shared constants and feasibility checks for the registered case study."""

from __future__ import annotations

import numpy as np


N_HOURS = 92 * 24
HORIZON_HOURS = 24
TEMPERATURE_MIN_C = 22.0
TEMPERATURE_MAX_C = 26.0
REFERENCE_TEMPERATURE_C = 24.0
INITIAL_TEMPERATURE_C = 24.0


def cooling_reachability_audit(
    disturbance_c: np.ndarray,
    a: np.ndarray,
    b_negative_c_per_kw: np.ndarray,
    power_max_kw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Audit daily reachable intervals for cooling-only scalar TCL models."""

    disturbance = np.asarray(disturbance_c, dtype=np.float64)
    n_users, n_periods = disturbance.shape
    if n_periods % HORIZON_HOURS:
        raise ValueError("The disturbance cannot be divided into 24-hour horizons.")
    if np.any(b_negative_c_per_kw >= 0.0):
        raise ValueError("Cooling-only reachability requires b < 0.")

    below_mask = np.zeros_like(disturbance, dtype=bool)
    above_mask = np.zeros_like(disturbance, dtype=bool)
    projected = disturbance.copy()
    infeasible_user_day = np.zeros(
        (n_users, n_periods // HORIZON_HOURS), dtype=bool
    )
    for day in range(n_periods // HORIZON_HOURS):
        lower = np.full(n_users, INITIAL_TEMPERATURE_C, dtype=np.float64)
        upper = lower.copy()
        start = day * HORIZON_HOURS
        for hour in range(HORIZON_HOURS):
            column = start + hour
            disturbance_lower = TEMPERATURE_MIN_C - a * upper
            disturbance_upper = (
                TEMPERATURE_MAX_C
                - a * lower
                - b_negative_c_per_kw * power_max_kw
            )
            raw = disturbance[:, column]
            below = raw < disturbance_lower - 1.0e-8
            above = raw > disturbance_upper + 1.0e-8
            below_mask[:, column] = below
            above_mask[:, column] = above
            infeasible_user_day[:, day] |= below | above
            value = np.clip(raw, disturbance_lower, disturbance_upper)
            projected[:, column] = value
            lower = np.maximum(
                a * lower + b_negative_c_per_kw * power_max_kw + value,
                TEMPERATURE_MIN_C,
            )
            upper = np.minimum(a * upper + value, TEMPERATURE_MAX_C)
            if np.any(lower > upper + 1.0e-8):
                raise AssertionError("Cooling-only disturbance projection failed.")

    return {
        "below_mask": below_mask,
        "above_mask": above_mask,
        "projected_disturbance_c": projected,
        "infeasible_user_day": infeasible_user_day,
    }
