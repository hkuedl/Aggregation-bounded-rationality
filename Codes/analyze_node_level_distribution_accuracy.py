"""Validate the node-level aggregate-deviation Gaussian approximation.

The script implements the numerical experiment described in Section III.A and
the node-level case study in the manuscript. It uses the cached nominal
optimal solutions, transforms the ecobee-derived conditional temperature PMFs
into physically feasible power-deviation trajectories, estimates the moments
in (12a)--(12b), applies (15)--(16), and compares the resulting Gaussian
distribution with an independent Monte Carlo reference.

The manuscript source is never modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


PLOT_ONLY_INVOCATION = "--plot-only" in sys.argv
if not PLOT_ONLY_INVOCATION:
    import scipy
    import scipy.sparse as sparse
    from scipy.special import ndtri
    from scipy.stats import gaussian_kde, kurtosis, norm, skew, spearmanr
else:
    scipy = None
    sparse = None


DEFAULT_DATE = "2025-08-15"
DEFAULT_HOUR = 19
DEFAULT_MOMENT_SAMPLES = 200
DEFAULT_GROUND_TRUTH_SAMPLES = 500
DEFAULT_SEED = 20260917
DEFAULT_BOOTSTRAP_SAMPLES = 1000
ACTIVE_START_HOUR = 8
ACTIVE_STOP_HOUR = 20
ACTIVE_HOURS = np.arange(ACTIVE_START_HOUR, ACTIVE_STOP_HOUR, dtype=np.int64)
USER_SIZES = np.arange(300, 3001, 300, dtype=np.int32)
HORIZON = 24
PMF_TOLERANCE = 1.0e-7
PHYSICAL_TOLERANCE = 1.0e-8
MODEL_BOUND_RELAXATION = 1.0e-5
ZERO_IDENTITY_TOLERANCE_KW = 5.0e-3


def load_gurobi():
    """Load the installed gurobipy package without machine-specific paths."""
    try:
        import gurobipy as gp  # type: ignore

        return gp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is unavailable. Install the version in requirements.txt "
            "and configure a valid Gurobi license."
        ) from exc


gp = None if PLOT_ONLY_INVOCATION else load_gurobi()


@dataclass
class NodeBlock:
    node_id: int
    archetype_id: int
    global_slice: slice
    user_id: np.ndarray
    a: np.ndarray
    b: np.ndarray
    power_min: np.ndarray
    power_max: np.ndarray
    temperature_min: float
    temperature_max: float
    temperature_penalty: np.ndarray
    price: np.ndarray
    disturbance: np.ndarray
    optimal_temperature: np.ndarray
    optimal_power: np.ndarray
    initial_temperature: float
    reference_anchor: np.ndarray
    temperature_support: np.ndarray
    conditional_probability: np.ndarray
    hourly_probability: np.ndarray
    preset_minimum: float
    preset_maximum: float
    reference_mapping: str
    raw_feasible_day: np.ndarray
    outside_low_count: int
    outside_high_count: int


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_iso_date(value: str) -> np.datetime64:
    try:
        date_type.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format.") from exc
    return np.datetime64(value, "D")


def node_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def read_assignments(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["bus"]): row for row in csv.DictReader(stream)}


def interpolate_conditional_pmf(
    conditional_probability: np.ndarray,
    preset: np.ndarray,
    conditioning_temperature: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Linearly interpolate user PMFs at each nominal optimal temperature."""
    probability = np.asarray(conditional_probability, dtype=np.float64)
    preset = np.asarray(preset, dtype=np.float64)
    temperature = np.asarray(conditioning_temperature, dtype=np.float64)
    if probability.ndim != 3 or temperature.ndim != 2:
        raise ValueError("Unexpected conditional-PMF or temperature dimensions.")
    if probability.shape[:2] != (temperature.shape[0], len(preset)):
        raise ValueError("Conditional PMFs do not align with users/preset bins.")
    if np.any(np.diff(preset) <= 0.0):
        raise ValueError("Preset temperatures must be strictly increasing.")

    outside_low = int(np.count_nonzero(temperature < preset[0]))
    outside_high = int(np.count_nonzero(temperature > preset[-1]))
    clipped = np.clip(temperature, preset[0], preset[-1])
    upper = np.searchsorted(preset, clipped, side="right")
    upper = np.clip(upper, 1, len(preset) - 1)
    lower = upper - 1
    lower_value = preset[lower]
    upper_value = preset[upper]
    weight = np.divide(
        clipped - lower_value,
        upper_value - lower_value,
        out=np.zeros_like(clipped),
        where=(upper_value > lower_value),
    )
    user_index = np.arange(temperature.shape[0], dtype=np.int64)[:, None]
    lower_probability = probability[user_index, lower]
    upper_probability = probability[user_index, upper]
    interpolated = (
        (1.0 - weight)[..., None] * lower_probability
        + weight[..., None] * upper_probability
    )
    interpolated = np.maximum(interpolated, 0.0)
    row_sum = interpolated.sum(axis=-1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("Interpolation produced an empty conditional PMF.")
    interpolated /= row_sum
    return interpolated, outside_low, outside_high


def reindex_absolute_pmf_as_relative_delta(
    conditional_probability: np.ndarray,
    preset: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-index P(actual | preset) as P(actual - preset | preset).

    This transformation preserves every probability mass from the Ecobee-derived
    conditional distribution.  It changes only the downstream interpretation of
    the support, so that a sampled behavioral deviation is applied relative to the
    model's recommended temperature rather than as an unrelated absolute setpoint.
    """
    probability = np.asarray(conditional_probability, dtype=np.float64)
    preset = np.asarray(preset, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if len(support) < 2:
        raise ValueError("At least two actual-temperature support points are required.")
    step = float(np.median(np.diff(support)))
    if step <= 0.0 or not np.allclose(np.diff(support), step, atol=1.0e-10):
        raise ValueError("Actual-temperature support must be uniformly spaced.")
    delta_minimum = float(np.min(support) - np.max(preset))
    delta_maximum = float(np.max(support) - np.min(preset))
    delta_support = np.arange(
        delta_minimum,
        delta_maximum + 0.5 * step,
        step,
        dtype=np.float64,
    )
    delta_probability = np.zeros(
        (probability.shape[0], probability.shape[1], len(delta_support)),
        dtype=np.float64,
    )
    for preset_index, preset_value in enumerate(preset):
        delta = support - preset_value
        target = np.rint((delta - delta_minimum) / step).astype(np.int64)
        if np.any(target < 0) or np.any(target >= len(delta_support)):
            raise ValueError("Relative-temperature support indexing failed.")
        delta_probability[:, preset_index, target] += probability[:, preset_index]
    if not np.allclose(
        delta_probability.sum(axis=-1),
        probability.sum(axis=-1),
        atol=PMF_TOLERANCE,
    ):
        raise RuntimeError("Relative-temperature re-indexing did not preserve PMF mass.")
    return delta_support, delta_probability


def load_inputs(
    input_dir: Path,
    bounded_dir: Path,
    optimal_dir: Path,
    selected_date: np.datetime64,
    selected_hour: int,
    reference_mapping: str,
    conditioning_basis: str,
) -> tuple[list[NodeBlock], pd.DataFrame, dict[str, object]]:
    assignments = read_assignments(input_dir / "node_assignments.csv")
    probability_dir = bounded_dir / "user_probability_distributions"
    optimal_paths = sorted(
        optimal_dir.glob("node_*_optimal_results.npz"), key=node_id_from_path
    )
    if not optimal_paths:
        raise FileNotFoundError(f"No optimal-result files found in {optimal_dir}")

    blocks: list[NodeBlock] = []
    registry_rows: list[dict[str, object]] = []
    global_offset = 0
    common_timestamps: np.ndarray | None = None
    selected_indices: np.ndarray | None = None
    target_global_index: int | None = None
    price_day: np.ndarray | None = None
    solver_statuses: list[str] = []

    day_start = selected_date.astype("datetime64[h]")
    day_end = day_start + np.timedelta64(HORIZON, "h")
    target_timestamp = day_start + np.timedelta64(selected_hour, "h")

    for optimal_path in optimal_paths:
        bus = node_id_from_path(optimal_path)
        probability_path = probability_dir / f"node_{bus:02d}_bounded_rationality.npz"
        if not probability_path.exists():
            raise FileNotFoundError(f"Missing probability file: {probability_path}")
        assignment = assignments.get(bus)
        if assignment is None or assignment["node_type"].strip().lower() != "user":
            raise ValueError(f"Bus {bus} is not registered as a user node.")

        with np.load(optimal_path, allow_pickle=False) as optimal, np.load(
            probability_path, allow_pickle=False
        ) as probability:
            user_id = np.asarray(optimal["user_id"], dtype=np.int32)
            probability_user_id = np.asarray(probability["user_id"], dtype=np.int32)
            if not np.array_equal(user_id, probability_user_id):
                raise ValueError(f"User IDs do not align at bus {bus}.")
            timestamps = np.asarray(optimal["timestamps"]).astype("datetime64[h]")
            if common_timestamps is None:
                common_timestamps = timestamps
                selected_indices = np.flatnonzero(
                    (timestamps >= day_start) & (timestamps < day_end)
                )
                target_matches = np.flatnonzero(timestamps == target_timestamp)
                if len(selected_indices) != HORIZON or len(target_matches) != 1:
                    raise ValueError(
                        f"Selected date/hour is not a complete day: "
                        f"{str(selected_date)}, {selected_hour}:00."
                    )
                target_global_index = int(target_matches[0])
                price_day = np.asarray(
                    optimal["price_usd_per_kwh"][selected_indices], dtype=np.float64
                )
            elif not np.array_equal(timestamps, common_timestamps):
                raise ValueError(f"Timestamp mismatch at bus {bus}.")

            assert selected_indices is not None
            optimal_temperature = np.asarray(
                optimal["optimal_temperature_c"][:, selected_indices], dtype=np.float64
            )
            optimal_power = np.asarray(
                optimal["optimal_power_kw"][:, selected_indices], dtype=np.float64
            )
            nominal_reference = np.full_like(
                optimal_temperature,
                float(np.asarray(optimal["reference_temperature_c"])),
            )
            disturbance = np.asarray(
                optimal["disturbance_used_c"][:, selected_indices], dtype=np.float64
            )
            conditional_probability = np.asarray(
                probability["conditional_probability"], dtype=np.float64
            )
            preset = np.asarray(probability["preset_setpoint_c"], dtype=np.float64)
            support = np.asarray(
                probability["actual_setpoint_support_c"], dtype=np.float64
            )
            if np.any(conditional_probability < -PMF_TOLERANCE) or not np.allclose(
                conditional_probability.sum(axis=-1), 1.0, atol=PMF_TOLERANCE
            ):
                raise ValueError(f"Invalid conditional PMF at bus {bus}.")
            if reference_mapping == "relative-delta":
                response_support, response_probability = (
                    reindex_absolute_pmf_as_relative_delta(
                        conditional_probability, preset, support
                    )
                )
            elif reference_mapping == "absolute":
                response_support = support
                response_probability = conditional_probability
            else:
                raise ValueError(f"Unknown reference mapping: {reference_mapping}")
            if conditioning_basis == "nominal-reference":
                reference_anchor = nominal_reference
            elif conditioning_basis == "optimal-state":
                reference_anchor = optimal_temperature
            else:
                raise ValueError(f"Unknown conditioning basis: {conditioning_basis}")
            hourly_probability, outside_low, outside_high = interpolate_conditional_pmf(
                response_probability, preset, reference_anchor
            )
            day_index = int(selected_indices[0] // HORIZON)
            raw_infeasible_day = np.asarray(
                optimal["raw_infeasible_user_day"][:, day_index], dtype=bool
            )
            solver_status = np.char.lower(np.asarray(optimal["solver_status"]))
            if np.any((solver_status != "solved") & (solver_status != "solved inaccurate")):
                raise ValueError(f"Nominal optimization was not solved at bus {bus}.")
            solver_statuses.extend(solver_status.tolist())

            n_users = len(user_id)
            block_slice = slice(global_offset, global_offset + n_users)
            archetype_id = int(assignment["archetype_id"])
            blocks.append(
                NodeBlock(
                    node_id=bus,
                    archetype_id=archetype_id,
                    global_slice=block_slice,
                    user_id=user_id,
                    a=np.asarray(optimal["a"], dtype=np.float64),
                    b=np.asarray(optimal["b_c_per_kw"], dtype=np.float64),
                    power_min=np.asarray(optimal["power_min_kw"], dtype=np.float64),
                    power_max=np.asarray(optimal["power_max_kw"], dtype=np.float64),
                    temperature_min=float(optimal["temperature_min_c"]),
                    temperature_max=float(optimal["temperature_max_c"]),
                    temperature_penalty=np.asarray(
                        optimal["temperature_penalty"], dtype=np.float64
                    ),
                    price=np.asarray(
                        optimal["price_usd_per_kwh"][selected_indices],
                        dtype=np.float64,
                    ),
                    disturbance=disturbance,
                    optimal_temperature=optimal_temperature,
                    optimal_power=optimal_power,
                    initial_temperature=float(optimal["initial_temperature_c"]),
                    reference_anchor=reference_anchor,
                    temperature_support=response_support,
                    conditional_probability=response_probability,
                    hourly_probability=hourly_probability,
                    preset_minimum=float(preset[0]),
                    preset_maximum=float(preset[-1]),
                    reference_mapping=reference_mapping,
                    raw_feasible_day=~raw_infeasible_day,
                    outside_low_count=outside_low,
                    outside_high_count=outside_high,
                )
            )
            for local_index, identifier in enumerate(user_id):
                registry_rows.append(
                    {
                        "global_position": global_offset + local_index,
                        "node_id": bus,
                        "archetype_id": archetype_id,
                        "user_id": int(identifier),
                        "raw_feasible_selected_day": bool(
                            not raw_infeasible_day[local_index]
                        ),
                    }
                )
            global_offset += n_users

    registry = pd.DataFrame(registry_rows)
    if len(registry) != 3000:
        raise ValueError(f"Expected 3,000 users, found {len(registry):,}.")
    if registry[["node_id", "user_id"]].drop_duplicates().shape[0] != len(registry):
        raise ValueError("The compound (node_id, user_id) keys are not unique.")
    if np.any(np.concatenate([block.b for block in blocks]) >= 0.0):
        raise ValueError("Cooling coefficient b must be negative for every user.")

    metadata = {
        "selected_date": str(selected_date),
        "selected_hour": selected_hour,
        "target_timestamp": str(target_timestamp),
        "target_local_index": selected_hour,
        "target_global_index": target_global_index,
        "users": len(registry),
        "user_nodes": len(blocks),
        "archetypes": int(registry["archetype_id"].nunique()),
        "price_usd_per_kwh": price_day.tolist() if price_day is not None else [],
        "nominal_solver_records_checked": len(solver_statuses),
        "reference_mapping": reference_mapping,
        "conditioning_basis": conditioning_basis,
        "raw_feasible_users_selected_day": int(
            registry["raw_feasible_selected_day"].sum()
        ),
    }
    return blocks, registry, metadata


def stratified_nested_order(registry: pd.DataFrame, seed: int) -> np.ndarray:
    """Create a reproducible node-stratified order with representative prefixes."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 771]))
    priority = np.empty(len(registry), dtype=np.float64)
    for _, group in registry.groupby("node_id", sort=True):
        positions = group["global_position"].to_numpy(dtype=np.int64)
        shuffled = rng.permutation(positions)
        jitter = rng.random(len(shuffled))
        priority[shuffled] = (np.arange(len(shuffled)) + jitter) / len(shuffled)
    order = np.lexsort((registry["user_id"].to_numpy(), priority))
    return order.astype(np.int64)


def restrict_to_user_positions(
    blocks: Sequence[NodeBlock],
    registry: pd.DataFrame,
    selected_positions: np.ndarray,
) -> tuple[list[NodeBlock], pd.DataFrame]:
    """Restrict node arrays to an explicitly selected user cohort."""
    positions = np.asarray(selected_positions, dtype=np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Selected user positions must be unique.")
    if np.any(positions < 0) or np.any(positions >= len(registry)):
        raise ValueError("A selected user position is outside the registry.")
    selected = set(positions.tolist())
    restricted_blocks: list[NodeBlock] = []
    restricted_rows: list[dict[str, object]] = []
    offset = 0
    for block in blocks:
        original = np.arange(
            block.global_slice.start, block.global_slice.stop, dtype=np.int64
        )
        local = np.flatnonzero(np.asarray([value in selected for value in original]))
        if not len(local):
            continue
        count = len(local)
        optimal_temperature = block.optimal_temperature[local]
        restricted_blocks.append(
            NodeBlock(
                node_id=block.node_id,
                archetype_id=block.archetype_id,
                global_slice=slice(offset, offset + count),
                user_id=block.user_id[local],
                a=block.a[local],
                b=block.b[local],
                power_min=block.power_min[local],
                power_max=block.power_max[local],
                temperature_min=block.temperature_min,
                temperature_max=block.temperature_max,
                temperature_penalty=block.temperature_penalty[local],
                price=block.price,
                disturbance=block.disturbance[local],
                optimal_temperature=optimal_temperature,
                optimal_power=block.optimal_power[local],
                initial_temperature=block.initial_temperature,
                reference_anchor=block.reference_anchor[local],
                temperature_support=block.temperature_support,
                conditional_probability=block.conditional_probability[local],
                hourly_probability=block.hourly_probability[local],
                preset_minimum=block.preset_minimum,
                preset_maximum=block.preset_maximum,
                reference_mapping=block.reference_mapping,
                raw_feasible_day=block.raw_feasible_day[local],
                outside_low_count=int(
                    np.count_nonzero(
                        block.reference_anchor[local] < block.preset_minimum
                    )
                ),
                outside_high_count=int(
                    np.count_nonzero(
                        block.reference_anchor[local] > block.preset_maximum
                    )
                ),
            )
        )
        for local_index, original_local_index in enumerate(local):
            original_position = int(original[original_local_index])
            source = registry.iloc[original_position]
            restricted_rows.append(
                {
                    "global_position": offset + local_index,
                    "node_id": int(source["node_id"]),
                    "archetype_id": int(source["archetype_id"]),
                    "user_id": int(source["user_id"]),
                    "raw_feasible_selected_day": bool(
                        source["raw_feasible_selected_day"]
                    ),
                    "original_global_position": original_position,
                }
            )
        offset += count
    if offset != len(positions):
        raise RuntimeError("User-cohort restriction lost one or more selected users.")
    return restricted_blocks, pd.DataFrame(restricted_rows)


def trim_blocks_to_event_onset(
    blocks: Sequence[NodeBlock], event_hour: int
) -> list[NodeBlock]:
    """Start the response horizon at a behavioral override event.

    The nominal state immediately before the event is used as the initial
    condition.  This prevents a perfect-foresight response from changing power
    before an occupant has actually overridden the recommended temperature.
    """
    if not 0 <= event_hour < HORIZON:
        raise ValueError("Event hour must be within the 24-hour horizon.")
    trimmed: list[NodeBlock] = []
    for block in blocks:
        if event_hour:
            initial_temperature: float | np.ndarray = block.optimal_temperature[
                :, event_hour - 1
            ].copy()
        else:
            initial_temperature = block.initial_temperature
        anchor = block.reference_anchor[:, event_hour:]
        trimmed.append(
            NodeBlock(
                node_id=block.node_id,
                archetype_id=block.archetype_id,
                global_slice=block.global_slice,
                user_id=block.user_id,
                a=block.a,
                b=block.b,
                power_min=block.power_min,
                power_max=block.power_max,
                temperature_min=block.temperature_min,
                temperature_max=block.temperature_max,
                temperature_penalty=block.temperature_penalty,
                price=block.price[event_hour:],
                disturbance=block.disturbance[:, event_hour:],
                optimal_temperature=block.optimal_temperature[:, event_hour:],
                optimal_power=block.optimal_power[:, event_hour:],
                initial_temperature=initial_temperature,
                reference_anchor=anchor,
                temperature_support=block.temperature_support,
                conditional_probability=block.conditional_probability,
                hourly_probability=block.hourly_probability[:, event_hour:],
                preset_minimum=block.preset_minimum,
                preset_maximum=block.preset_maximum,
                reference_mapping=block.reference_mapping,
                raw_feasible_day=block.raw_feasible_day,
                outside_low_count=int(
                    np.count_nonzero(anchor < block.preset_minimum)
                ),
                outside_high_count=int(
                    np.count_nonzero(anchor > block.preset_maximum)
                ),
            )
        )
    return trimmed


def simulate_deviations(
    block: NodeBlock,
    simulations: int,
    seed: int,
    phase_code: int,
    solver_bundle: tuple[object, object, int, int, np.ndarray, np.ndarray] | None = None,
    batch_size: int = 10,
    response_objective: str = "price-responsive",
    temporal_sampling: str = "hourly-independent",
    event_duration_hours: int = 4,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Draw overridden setpoints, re-optimize TCL responses, and return deviations."""
    if response_objective == "direct-dynamics":
        if temporal_sampling != "active-window-hourly-independent":
            raise ValueError(
                "Direct dynamics requires active-window-hourly-independent sampling."
            )
        if block.optimal_temperature.shape[1] != HORIZON:
            raise ValueError("Direct dynamics requires a complete 24-hour block.")
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, block.node_id, phase_code])
        )
        n_users = len(block.user_id)
        n_support = len(block.temperature_support)
        cumulative = np.cumsum(
            block.hourly_probability[:, ACTIVE_HOURS, :], axis=-1
        )
        cumulative[:, :, -1] = 1.0
        random_value = rng.random((simulations, n_users, len(ACTIVE_HOURS)))
        sampled_index = np.sum(
            random_value[:, :, :, None] > cumulative[None, :, :, :], axis=-1
        )
        sampled_index = np.minimum(sampled_index, n_support - 1)
        sampled_delta = block.temperature_support[sampled_index]
        target_temperature = np.broadcast_to(
            block.optimal_temperature[None, :, :],
            (simulations, n_users, HORIZON),
        ).copy()
        if block.reference_mapping == "relative-delta":
            target_temperature[:, :, ACTIVE_HOURS] += sampled_delta
        else:
            target_temperature[:, :, ACTIVE_HOURS] = sampled_delta

        actual_temperature = np.empty_like(target_temperature)
        actual_power = np.empty_like(target_temperature)
        initial = np.asarray(block.initial_temperature, dtype=np.float64)
        if initial.ndim == 0:
            initial_by_user = np.full(n_users, float(initial), dtype=np.float64)
        elif initial.shape == (n_users,):
            initial_by_user = initial.copy()
        else:
            raise ValueError("Initial temperatures do not align with users.")
        previous = np.broadcast_to(
            initial_by_user[None, :], (simulations, n_users)
        ).copy()
        saturation_count = 0
        for hour in range(HORIZON):
            raw_power = (
                target_temperature[:, :, hour]
                - block.a[None, :] * previous
                - block.disturbance[None, :, hour]
            ) / block.b[None, :]
            power = np.clip(
                raw_power,
                block.power_min[None, :],
                block.power_max[None, :],
            )
            temperature = (
                block.a[None, :] * previous
                + block.b[None, :] * power
                + block.disturbance[None, :, hour]
            )
            actual_power[:, :, hour] = power
            actual_temperature[:, :, hour] = temperature
            saturation_count += int(
                np.count_nonzero(np.abs(power - raw_power) > 1.0e-10)
            )
            previous = temperature

        previous_temperature = np.concatenate(
            (
                np.broadcast_to(
                    initial_by_user[None, :, None],
                    (simulations, n_users, 1),
                ),
                actual_temperature[:, :, :-1],
            ),
            axis=2,
        )
        residual = np.abs(
            actual_temperature
            - block.a[None, :, None] * previous_temperature
            - block.b[None, :, None] * actual_power
            - block.disturbance[None, :, :]
        )
        power_violation = max(
            float(np.max(block.power_min[None, :, None] - actual_power)),
            float(np.max(actual_power - block.power_max[None, :, None])),
            0.0,
        )
        if power_violation > PHYSICAL_TOLERANCE:
            raise RuntimeError(
                f"Power-bound validation failed at bus {block.node_id}: "
                f"{power_violation:.3e} kW."
            )
        comfort_excursion = np.maximum(
            np.maximum(block.temperature_min - actual_temperature, 0.0),
            np.maximum(actual_temperature - block.temperature_max, 0.0),
        )
        active_target = target_temperature[:, :, ACTIVE_HOURS]
        below_count = int(np.count_nonzero(active_target < block.temperature_min))
        above_count = int(np.count_nonzero(active_target > block.temperature_max))
        tracking_error = np.abs(actual_temperature - target_temperature)
        deviations = (actual_power - block.optimal_power[None, :, :]).astype(
            np.float32
        )
        transitions = simulations * n_users * HORIZON
        sampled_points = int(active_target.size)
        outside_count = int(np.count_nonzero(comfort_excursion > 1.0e-10))
        return deviations, {
            "node_id": block.node_id,
            "simulations": simulations,
            "solver_batches": 0,
            "solved_inaccurate_batches": 0,
            "transitions": transitions,
            "fallback_transitions": 0,
            "fallback_rate": 0.0,
            "mean_retained_pmf_mass": 1.0,
            "mean_pmf_mass_removed": 0.0,
            "minimum_sampled_reference_c": float(np.min(active_target)),
            "maximum_sampled_reference_c": float(np.max(active_target)),
            "sampled_reference_points_evaluated": sampled_points,
            "sampled_reference_below_physical_count": below_count,
            "sampled_reference_above_physical_count": above_count,
            "sampled_reference_outside_physical_rate": (
                below_count + above_count
            ) / sampled_points,
            "mean_absolute_temperature_tracking_error_c": float(
                np.mean(tracking_error)
            ),
            "maximum_temperature_tracking_error_c": float(np.max(tracking_error)),
            "minimum_actual_temperature_c": float(np.min(actual_temperature)),
            "maximum_actual_temperature_c": float(np.max(actual_temperature)),
            "minimum_actual_power_kw": float(np.min(actual_power)),
            "maximum_actual_power_kw": float(np.max(actual_power)),
            "maximum_temperature_violation_c": 0.0,
            "maximum_actual_temperature_outside_comfort_c": float(
                np.max(comfort_excursion)
            ),
            "actual_temperature_outside_comfort_rate": outside_count / transitions,
            "power_saturation_rate": saturation_count / transitions,
            "maximum_power_violation_kw": power_violation,
            "maximum_dynamics_residual_c": float(np.max(residual)),
        }

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, block.node_id, phase_code])
    )
    n_users = len(block.user_id)
    n_periods = block.optimal_temperature.shape[1]
    n_support = len(block.temperature_support)
    deviations = np.empty((simulations, n_users, n_periods), dtype=np.float32)
    maximum_dynamic_residual = 0.0
    maximum_temperature_violation = 0.0
    maximum_power_violation = 0.0
    minimum_actual_temperature = math.inf
    maximum_actual_temperature = -math.inf
    minimum_actual_power = math.inf
    maximum_actual_power = -math.inf
    minimum_sampled_reference = math.inf
    maximum_sampled_reference = -math.inf
    sampled_reference_points = 0
    sampled_reference_below_physical = 0
    sampled_reference_above_physical = 0
    tracking_absolute_error_sum = 0.0
    maximum_tracking_error = 0.0
    solved_inaccurate_batches = 0

    if solver_bundle is None:
        solver_bundle = build_reference_response_solver(block, batch_size)
    model, decision, n_thermal, entities, entity_penalty, entity_price = solver_bundle
    if entities != batch_size * n_users:
        raise ValueError("The response solver does not match the node/batch size.")
    cumulative_probability = np.cumsum(block.hourly_probability, axis=-1)
    cumulative_probability[:, :, -1] = 1.0

    for start in range(0, simulations, batch_size):
        active = min(batch_size, simulations - start)
        if temporal_sampling == "hourly-independent":
            random_value = rng.random((batch_size, n_users, n_periods))
            sampled_index = np.sum(
                random_value[:, :, :, None]
                > cumulative_probability[None, :, :, :],
                axis=-1,
            )
            sampled_index = np.minimum(sampled_index, n_support - 1)
            sampled_value = block.temperature_support[sampled_index]
            if block.reference_mapping == "relative-delta":
                sampled_reference = (
                    block.reference_anchor[None, :, :] + sampled_value
                )
            else:
                sampled_reference = sampled_value
        elif temporal_sampling == "event-onset-persistent":
            random_value = rng.random((batch_size, n_users))
            sampled_index = np.sum(
                random_value[:, :, None] > cumulative_probability[None, :, 0, :],
                axis=-1,
            )
            sampled_index = np.minimum(sampled_index, n_support - 1)
            sampled_value = block.temperature_support[sampled_index]
            sampled_reference = np.broadcast_to(
                block.reference_anchor[None, :, :],
                (batch_size, n_users, n_periods),
            ).copy()
            duration = min(event_duration_hours, n_periods)
            if block.reference_mapping == "relative-delta":
                sampled_reference[:, :, :duration] += sampled_value[:, :, None]
            else:
                sampled_reference[:, :, :duration] = sampled_value[:, :, None]
        else:
            raise ValueError(f"Unknown temporal sampling: {temporal_sampling}")
        minimum_sampled_reference = min(
            minimum_sampled_reference, float(np.min(sampled_reference[:active]))
        )
        maximum_sampled_reference = max(
            maximum_sampled_reference, float(np.max(sampled_reference[:active]))
        )
        if temporal_sampling == "event-onset-persistent":
            diagnostic_periods = min(event_duration_hours, n_periods)
        else:
            diagnostic_periods = n_periods
        diagnostic_reference = sampled_reference[
            :active, :, :diagnostic_periods
        ]
        sampled_reference_points += int(diagnostic_reference.size)
        sampled_reference_below_physical += int(
            np.count_nonzero(diagnostic_reference < block.temperature_min)
        )
        sampled_reference_above_physical += int(
            np.count_nonzero(diagnostic_reference > block.temperature_max)
        )
        if response_objective == "price-responsive":
            power_linear = entity_price
        elif response_objective == "temperature-tracking":
            power_linear = np.zeros_like(entity_price)
        else:
            raise ValueError(f"Unknown response objective: {response_objective}")
        linear = np.r_[
            -2.0 * entity_penalty * sampled_reference.reshape(-1), power_linear
        ]
        decision.Obj = linear
        model.optimize()
        status = int(model.Status)
        if status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi failed at bus {block.node_id}, simulations "
                f"{start + 1}--{start + active}: status={status}"
            )
        solution = np.asarray(decision.X, dtype=np.float64)
        temperature = solution[:n_thermal].reshape(batch_size, n_users, n_periods)
        initial = np.asarray(block.initial_temperature, dtype=np.float64)
        if initial.ndim == 0:
            initial = np.full(n_users, float(initial), dtype=np.float64)
        if initial.shape != (n_users,):
            raise ValueError("Initial temperatures do not align with users.")
        previous = np.concatenate(
            (
                np.broadcast_to(initial[None, :, None], (batch_size, n_users, 1)),
                temperature[:, :, :-1],
            ),
            axis=2,
        )
        actual_power = (
            temperature
            - block.a[None, :, None] * previous
            - block.disturbance[None, :, :]
        ) / block.b[None, :, None]
        temperature_violation = max(
            float(np.max(block.temperature_min - temperature[:active])),
            float(np.max(temperature[:active] - block.temperature_max)),
            0.0,
        )
        power_violation = max(
            float(np.max(block.power_min[None, :, None] - actual_power[:active])),
            float(np.max(actual_power[:active] - block.power_max[None, :, None])),
            0.0,
        )
        residual = np.abs(
            temperature
            - block.a[None, :, None] * previous
            - block.b[None, :, None] * actual_power
            - block.disturbance[None, :, :]
        )
        maximum_temperature_violation = max(
            maximum_temperature_violation, temperature_violation
        )
        maximum_power_violation = max(maximum_power_violation, power_violation)
        maximum_dynamic_residual = max(
            maximum_dynamic_residual, float(np.max(residual[:active]))
        )
        if max(temperature_violation, power_violation) > 2.0e-5:
            raise RuntimeError(
                f"Actual-response validation failed at bus {block.node_id}: "
                f"temperature={temperature_violation:.3e}, "
                f"power={power_violation:.3e}."
            )
        minimum_actual_temperature = min(
            minimum_actual_temperature, float(np.min(temperature[:active]))
        )
        maximum_actual_temperature = max(
            maximum_actual_temperature, float(np.max(temperature[:active]))
        )
        minimum_actual_power = min(
            minimum_actual_power, float(np.min(actual_power[:active]))
        )
        maximum_actual_power = max(
            maximum_actual_power, float(np.max(actual_power[:active]))
        )
        tracking_error = np.abs(
            temperature[:active, :, :diagnostic_periods] - diagnostic_reference
        )
        tracking_absolute_error_sum += float(np.sum(tracking_error))
        maximum_tracking_error = max(
            maximum_tracking_error, float(np.max(tracking_error))
        )
        deviations[start : start + active] = (
            actual_power[:active] - block.optimal_power[None, :, :]
        ).astype(np.float32)

    diagnostics: dict[str, float | int] = {
        "node_id": block.node_id,
        "simulations": simulations,
        "solver_batches": math.ceil(simulations / batch_size),
        "solved_inaccurate_batches": solved_inaccurate_batches,
        "transitions": simulations * n_users * n_periods,
        "fallback_transitions": 0,
        "fallback_rate": 0.0,
        "mean_retained_pmf_mass": 1.0,
        "mean_pmf_mass_removed": 0.0,
        "minimum_sampled_reference_c": minimum_sampled_reference,
        "maximum_sampled_reference_c": maximum_sampled_reference,
        "sampled_reference_points_evaluated": sampled_reference_points,
        "sampled_reference_below_physical_count": sampled_reference_below_physical,
        "sampled_reference_above_physical_count": sampled_reference_above_physical,
        "sampled_reference_outside_physical_rate": (
            sampled_reference_below_physical + sampled_reference_above_physical
        )
        / sampled_reference_points,
        "mean_absolute_temperature_tracking_error_c": (
            tracking_absolute_error_sum / sampled_reference_points
        ),
        "maximum_temperature_tracking_error_c": maximum_tracking_error,
        "minimum_actual_temperature_c": minimum_actual_temperature,
        "maximum_actual_temperature_c": maximum_actual_temperature,
        "minimum_actual_power_kw": minimum_actual_power,
        "maximum_actual_power_kw": maximum_actual_power,
        "maximum_temperature_violation_c": maximum_temperature_violation,
        "maximum_power_violation_kw": maximum_power_violation,
        "maximum_dynamics_residual_c": maximum_dynamic_residual,
    }
    return deviations, diagnostics


def build_reference_response_solver(
    block: NodeBlock, batch_size: int
) -> tuple[object, object, int, int, np.ndarray, np.ndarray]:
    """Build one reusable batched QP for overridden temperature references."""
    n_users = len(block.user_id)
    n_periods = block.optimal_temperature.shape[1]
    entities = batch_size * n_users
    entity_a = np.tile(block.a, batch_size)
    entity_b = np.tile(block.b, batch_size)
    entity_power_min = np.tile(block.power_min, batch_size)
    entity_power_max = np.tile(block.power_max, batch_size)
    entity_penalty = np.repeat(
        np.tile(block.temperature_penalty, batch_size), n_periods
    )
    n_thermal = entities * n_periods
    n_variables = 2 * n_thermal

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for entity in range(entities):
        for hour in range(n_periods):
            row = entity * n_periods + hour
            rows.extend((row, row))
            columns.extend((row, n_thermal + row))
            values.extend((1.0, -entity_b[entity]))
            if hour:
                rows.append(row)
                columns.append(row - 1)
                values.append(-entity_a[entity])
    dynamics = sparse.csc_matrix(
        (values, (rows, columns)), shape=(n_thermal, n_variables)
    )
    variable_lower = np.r_[
        np.full(n_thermal, block.temperature_min - MODEL_BOUND_RELAXATION),
        np.repeat(entity_power_min, n_periods) - MODEL_BOUND_RELAXATION,
    ]
    variable_upper = np.r_[
        np.full(n_thermal, block.temperature_max + MODEL_BOUND_RELAXATION),
        np.repeat(entity_power_max, n_periods) + MODEL_BOUND_RELAXATION,
    ]
    disturbance = np.tile(block.disturbance, (batch_size, 1))
    rhs = disturbance.copy()
    initial = np.asarray(block.initial_temperature, dtype=np.float64)
    if initial.ndim == 0:
        initial = np.full(n_users, float(initial), dtype=np.float64)
    if initial.shape != (n_users,):
        raise ValueError("Initial temperatures do not align with users.")
    rhs[:, 0] += entity_a * np.tile(initial, batch_size)
    rhs = rhs.reshape(-1)
    quadratic = np.r_[entity_penalty, np.zeros(n_thermal)]
    entity_price = np.tile(block.price, entities)
    initial_reference = np.tile(block.optimal_temperature, (batch_size, 1))
    initial_linear = np.r_[
        -2.0 * entity_penalty * initial_reference.reshape(-1), entity_price
    ]

    model = gp.Model(f"actual_response_bus_{block.node_id:02d}")
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1.0e-8
    model.Params.OptimalityTol = 1.0e-8
    model.Params.NumericFocus = 1
    decision = model.addMVar(
        n_variables,
        lb=variable_lower,
        ub=variable_upper,
        name="response",
    )
    model.addMConstr(dynamics, decision, "=", rhs, name="thermal_dynamics")
    model.setMObjective(
        sparse.diags(quadratic, format="csc"),
        initial_linear,
        0.0,
        xc=decision,
        sense=gp.GRB.MINIMIZE,
    )
    model.update()
    return model, decision, n_thermal, entities, entity_penalty, entity_price


def zero_deviation_identity_error(
    block: NodeBlock,
    solver_bundle: tuple[object, object, int, int, np.ndarray, np.ndarray],
    batch_size: int,
) -> float:
    """Verify that temperature tracking maps zero behavioral bias to p* exactly."""
    model, decision, n_thermal, entities, entity_penalty, entity_price = solver_bundle
    n_users = len(block.user_id)
    n_periods = block.optimal_temperature.shape[1]
    if entities != batch_size * n_users:
        raise ValueError("The response solver does not match the zero-bias check.")
    target = np.tile(block.optimal_temperature, (batch_size, 1))
    decision.Obj = np.r_[
        -2.0 * entity_penalty * target.reshape(-1),
        np.zeros_like(entity_price),
    ]
    model.optimize()
    if int(model.Status) != gp.GRB.OPTIMAL:
        raise RuntimeError(
            f"Zero-deviation identity check failed at bus {block.node_id}: "
            f"status={int(model.Status)}"
        )
    temperature = np.asarray(decision.X[:n_thermal], dtype=np.float64).reshape(
        batch_size, n_users, n_periods
    )
    initial = np.asarray(block.initial_temperature, dtype=np.float64)
    if initial.ndim == 0:
        initial = np.full(n_users, float(initial), dtype=np.float64)
    previous = np.concatenate(
        (
            np.broadcast_to(initial[None, :, None], (batch_size, n_users, 1)),
            temperature[:, :, :-1],
        ),
        axis=2,
    )
    power = (
        temperature
        - block.a[None, :, None] * previous
        - block.disturbance[None, :, :]
    ) / block.b[None, :, None]
    return float(
        np.max(np.abs(power - block.optimal_power[None, :, :]))
    )


def empirical_to_normal_wasserstein(
    samples: np.ndarray, mean: float, std: float, grid_size: int = 100_000
) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    if std <= 0.0:
        return float(np.mean(np.abs(samples - mean)))
    u = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    empirical_index = np.minimum((u * len(samples)).astype(np.int64), len(samples) - 1)
    theoretical = mean + std * ndtri(u)
    return float(np.mean(np.abs(samples[empirical_index] - theoretical)))


def bootstrap_wasserstein_interval(
    samples: np.ndarray,
    mean: float,
    std: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float64)
    n = len(values)
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    theoretical = mean + std * ndtri(u) if std > 0.0 else np.full(n, mean)
    distances = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = 100
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        index = rng.integers(0, n, size=(stop - start, n))
        resampled = np.sort(values[index], axis=1)
        distances[start:stop] = np.mean(
            np.abs(resampled - theoretical[None, :]), axis=1
        )
    lower, upper = np.quantile(distances, [0.025, 0.975])
    return float(lower), float(upper)


def configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.2,
            "axes.labelpad": 2.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def make_figure(
    density_samples: np.ndarray,
    density_mean: float,
    density_std: float,
    density_user_count: int,
    metric_rows: Sequence[Mapping[str, object]],
    figure_base: Path,
    density_source_path: Path,
    include_density_figure: bool,
    include_wasserstein_figure: bool,
) -> None:
    configure_figure_style()
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    blue = "#3B6FB6"
    orange = "#D96B3B"
    pale_blue = "#AFC7E8"
    x_size = np.asarray([int(row["user_count"]) for row in metric_rows])
    w1 = np.asarray([float(row["wasserstein_1_kw_per_user"]) for row in metric_rows])

    # Use compact single-column canvases: the density panel is moderately
    # compressed vertically, while the Wasserstein panel has an approximately
    # 0.58 height-to-width ratio.
    density_path = figure_base.with_name(f"{figure_base.name}_density.pdf")
    wasserstein_path = figure_base.with_name(
        f"{figure_base.name}_wasserstein.pdf"
    )

    if include_density_figure:
        sample_min = float(np.min(density_samples))
        sample_max = float(np.max(density_samples))
        theoretical_min = density_mean - 4.25 * density_std
        theoretical_max = density_mean + 4.25 * density_std
        lower = min(sample_min, theoretical_min)
        upper = max(sample_max, theoretical_max)
        span = max(upper - lower, 1.0e-8)
        x = np.linspace(lower - 0.05 * span, upper + 0.05 * span, 600)
        kde = gaussian_kde(density_samples)
        empirical_density = kde(x)
        gaussian_density = norm.pdf(x, loc=density_mean, scale=density_std)
        density_rows = [
            {
                "user_count": density_user_count,
                "aggregate_deviation_kw_per_user": float(value),
                "monte_carlo_kde": float(empirical),
                "proposed_gaussian_pdf": float(theoretical),
            }
            for value, empirical, theoretical in zip(
                x, empirical_density, gaussian_density
            )
        ]
        write_csv(density_source_path, density_rows)

        fig_density, ax = plt.subplots(
            figsize=(3.6, 2.65), constrained_layout=False
        )
        fig_density.subplots_adjust(left=0.18, right=0.99, bottom=0.21, top=0.72)
        ax.hist(
            density_samples,
            bins=24,
            density=True,
            color=pale_blue,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            label="Monte Carlo histogram",
        )
        ax.plot(x, empirical_density, color=blue, label="Monte Carlo KDE")
        ax.plot(x, gaussian_density, color=orange, linestyle="--", label="Proposed Gaussian")
        ax.axvline(np.mean(density_samples), color=blue, linewidth=0.8, alpha=0.65)
        ax.axvline(density_mean, color=orange, linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_xlabel("Aggregate power deviation (kW per user)")
        ax.set_ylabel("Probability density")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            handlelength=1.25,
            handletextpad=0.35,
            columnspacing=0.65,
            borderaxespad=0.0,
        )
        fig_density.savefig(density_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_density)

    if include_wasserstein_figure:
        fig_wasserstein, ax = plt.subplots(
            figsize=(3.6, 2.10), constrained_layout=False
        )
        fig_wasserstein.subplots_adjust(left=0.20, right=0.99, bottom=0.31, top=0.98)
        ax.plot(x_size, w1, color=blue, marker="o", markersize=3.8)
        ax.set_xlabel("Number of users")
        ax.set_ylabel("1-Wasserstein distance")
        ax.set_xticks(x_size)
        ax.tick_params(axis="x", rotation=35)
        fig_wasserstein.savefig(
            wasserstein_path, bbox_inches="tight", pad_inches=0.02
        )
        plt.close(fig_wasserstein)


def make_figure_from_saved_results(
    output_dir: Path,
    figure_base: Path,
    include_density_figure: bool,
    include_wasserstein_figure: bool,
) -> None:
    """Redraw registered figures without rerunning simulation or optimization."""
    configure_figure_style()
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    blue = "#3B6FB6"
    orange = "#D96B3B"
    pale_blue = "#AFC7E8"

    if include_density_figure:
        source = pd.read_csv(output_dir / "figure_density_source.csv")
        with np.load(output_dir / "aggregate_mc_samples.npz", allow_pickle=False) as archive:
            samples = np.asarray(
                archive["aggregate_ground_truth_target_kw_per_user"][0],
                dtype=np.float64,
            )
            target = int(np.asarray(archive["target_hour_index"]).item())
            proposed_mean = float(
                archive["aggregate_gaussian_mean_kw"][0, target]
            )
        x = source["aggregate_deviation_kw_per_user"].to_numpy(dtype=np.float64)
        empirical_density = source["monte_carlo_kde"].to_numpy(dtype=np.float64)
        gaussian_density = source["proposed_gaussian_pdf"].to_numpy(dtype=np.float64)
        fig_density, ax = plt.subplots(figsize=(3.6, 2.65), constrained_layout=False)
        fig_density.subplots_adjust(left=0.18, right=0.99, bottom=0.21, top=0.72)
        ax.hist(
            samples,
            bins=24,
            density=True,
            color=pale_blue,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.55,
            label="Monte Carlo histogram",
        )
        ax.plot(x, empirical_density, color=blue, label="Monte Carlo KDE")
        ax.plot(x, gaussian_density, color=orange, linestyle="--", label="Proposed Gaussian")
        ax.axvline(np.mean(samples), color=blue, linewidth=0.8, alpha=0.65)
        ax.axvline(proposed_mean, color=orange, linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_xlabel("Aggregate power deviation (kW per user)")
        ax.set_ylabel("Probability density")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            handlelength=1.25,
            handletextpad=0.35,
            columnspacing=0.65,
            borderaxespad=0.0,
        )
        fig_density.savefig(
            figure_base.with_name(f"{figure_base.name}_density.pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig_density)

    if include_wasserstein_figure:
        metrics = pd.read_csv(output_dir / "wasserstein_by_user_count.csv")
        x_size = metrics["user_count"].to_numpy(dtype=np.int32)
        w1 = metrics["wasserstein_1_kw_per_user"].to_numpy(dtype=np.float64)
        fig_wasserstein, ax = plt.subplots(figsize=(3.6, 2.10), constrained_layout=False)
        fig_wasserstein.subplots_adjust(left=0.20, right=0.99, bottom=0.31, top=0.98)
        ax.plot(x_size, w1, color=blue, marker="o", markersize=3.8)
        ax.set_xlabel("Number of users")
        ax.set_ylabel("1-Wasserstein distance")
        ax.set_xticks(x_size)
        ax.tick_params(axis="x", rotation=35)
        fig_wasserstein.savefig(
            figure_base.with_name(f"{figure_base.name}_wasserstein.pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig_wasserstein)


def run_analysis(
    project: Path,
    selected_date: np.datetime64,
    selected_hour: int,
    moment_samples: int,
    ground_truth_samples: int,
    seed: int,
    bootstrap_samples: int,
    output_dir: Path,
    figure_base: Path,
    reference_mapping: str,
    conditioning_basis: str,
    analysis_users: int,
    raw_feasible_day_only: bool,
    write_figures: bool,
    include_wasserstein_figure: bool,
    include_density_figure: bool,
    response_objective: str,
    temporal_sampling: str,
    event_duration_hours: int,
) -> dict[str, object]:
    start_time = time.perf_counter()
    input_dir = project / "Data" / "Inputs_33kV_25MW_3000_JulSep"
    bounded_dir = project / "Outputs" / "Bounded Rationality Jul-Sep 3000"
    optimal_dir = bounded_dir / "Optimal results"
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks, registry, input_metadata = load_inputs(
        input_dir,
        bounded_dir,
        optimal_dir,
        selected_date,
        selected_hour,
        reference_mapping,
        conditioning_basis,
    )
    full_registry = registry.copy()
    full_order = stratified_nested_order(full_registry, seed)
    if raw_feasible_day_only:
        eligible = full_registry["raw_feasible_selected_day"].to_numpy(dtype=bool)
        cohort_positions = np.asarray(
            [position for position in full_order if eligible[position]][:analysis_users],
            dtype=np.int64,
        )
    else:
        cohort_positions = full_order[:analysis_users]
    if len(cohort_positions) != analysis_users:
        raise ValueError(
            f"Requested {analysis_users} users but only {len(cohort_positions)} "
            "satisfy the cohort restriction."
        )
    blocks, registry = restrict_to_user_positions(
        blocks, full_registry, cohort_positions
    )
    if temporal_sampling == "event-onset-persistent":
        blocks = trim_blocks_to_event_onset(blocks, selected_hour)
        target_index = 0
    else:
        target_index = selected_hour
    total_users = len(registry)
    analysis_horizon = blocks[0].optimal_temperature.shape[1]
    if any(block.optimal_temperature.shape[1] != analysis_horizon for block in blocks):
        raise ValueError("Node blocks have inconsistent response horizons.")
    order = stratified_nested_order(registry, seed)
    user_sizes = USER_SIZES[USER_SIZES <= total_users]
    if not len(user_sizes) or int(user_sizes[-1]) != total_users:
        raise ValueError("--analysis-users must be a multiple of 300 from 300 to 3000.")

    individual_mean = np.empty((total_users, analysis_horizon), dtype=np.float64)
    individual_covariance = np.empty(
        (total_users, analysis_horizon, analysis_horizon), dtype=np.float64
    )
    half_mean_target = np.empty(total_users, dtype=np.float64)
    half_variance_target = np.empty(total_users, dtype=np.float64)
    ground_truth_target = np.empty(
        (ground_truth_samples, total_users), dtype=np.float32
    )
    ground_truth_full_sum = np.zeros(
        (ground_truth_samples, analysis_horizon), dtype=np.float64
    )
    sampling_diagnostics: list[dict[str, object]] = []

    for block_index, block in enumerate(blocks, start=1):
        solver_bundle = (
            None
            if response_objective == "direct-dynamics"
            else build_reference_response_solver(block, batch_size=10)
        )
        zero_identity_error = math.nan
        if response_objective == "temperature-tracking":
            assert solver_bundle is not None
            zero_identity_error = zero_deviation_identity_error(
                block, solver_bundle, batch_size=10
            )
            if zero_identity_error > ZERO_IDENTITY_TOLERANCE_KW:
                raise RuntimeError(
                    f"Zero behavioral deviation does not recover nominal power "
                    f"at bus {block.node_id}: {zero_identity_error:.3e} kW."
                )
        print(
            f"[{block_index:02d}/{len(blocks):02d}] bus {block.node_id:02d}: "
            f"{len(block.user_id)} users; estimating {moment_samples} moment samples",
            flush=True,
        )
        moment_deviation, moment_diagnostic = simulate_deviations(
            block,
            moment_samples,
            seed,
            phase_code=1,
            solver_bundle=solver_bundle,
            batch_size=10,
            response_objective=response_objective,
            temporal_sampling=temporal_sampling,
            event_duration_hours=event_duration_hours,
        )
        moment64 = moment_deviation.astype(np.float64)
        mean = np.mean(moment64, axis=0)
        centered = moment64 - mean[None, :, :]
        covariance = np.einsum(
            "sut,suv->utv", centered, centered, optimize=True
        ) / moment_samples
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
        individual_mean[block.global_slice] = mean
        individual_covariance[block.global_slice] = covariance
        half = max(1, moment_samples // 2)
        half_values = moment64[:half, :, target_index]
        half_mean_target[block.global_slice] = np.mean(half_values, axis=0)
        half_variance_target[block.global_slice] = np.var(
            half_values, axis=0, ddof=0
        )
        sampling_diagnostics.append(
            {
                "phase": "individual_moments",
                **moment_diagnostic,
                "conditioning_below_preset_support": block.outside_low_count,
                "conditioning_above_preset_support": block.outside_high_count,
                "zero_deviation_max_abs_power_error_kw": zero_identity_error,
            }
        )
        del moment_deviation, moment64, centered, covariance

        print(
            f"[{block_index:02d}/{len(blocks):02d}] bus {block.node_id:02d}: "
            f"generating {ground_truth_samples} ground-truth samples",
            flush=True,
        )
        ground_truth_deviation, ground_truth_diagnostic = simulate_deviations(
            block,
            ground_truth_samples,
            seed,
            phase_code=2,
            solver_bundle=solver_bundle,
            batch_size=10,
            response_objective=response_objective,
            temporal_sampling=temporal_sampling,
            event_duration_hours=event_duration_hours,
        )
        ground_truth_target[:, block.global_slice] = ground_truth_deviation[
            :, :, target_index
        ]
        ground_truth_full_sum += np.sum(
            ground_truth_deviation.astype(np.float64), axis=1
        )
        sampling_diagnostics.append(
            {
                "phase": "ground_truth",
                **ground_truth_diagnostic,
                "conditioning_below_preset_support": block.outside_low_count,
                "conditioning_above_preset_support": block.outside_high_count,
                "zero_deviation_max_abs_power_error_kw": zero_identity_error,
            }
        )
        del ground_truth_deviation

    if not np.all(np.isfinite(individual_mean)) or not np.all(
        np.isfinite(individual_covariance)
    ):
        raise RuntimeError("Non-finite individual moments were produced.")
    covariance_diagonal = np.diagonal(individual_covariance, axis1=1, axis2=2)
    if float(np.min(covariance_diagonal)) < -1.0e-10:
        raise RuntimeError("An individual covariance has a negative variance.")

    ranked_registry = registry.iloc[order].copy().reset_index(drop=True)
    ranked_registry.insert(0, "selection_rank", np.arange(1, total_users + 1))
    ranked_registry["first_included_user_count"] = (
        np.ceil(ranked_registry["selection_rank"] / 300.0).astype(int) * 300
    ).clip(upper=3000)
    ranked_registry.to_csv(
        output_dir / "selected_users.csv", index=False, encoding="utf-8-sig"
    )

    ordered_mean = individual_mean[order]
    ordered_covariance = individual_covariance[order]
    cumulative_mean = np.cumsum(ordered_mean, axis=0)
    cumulative_covariance = np.cumsum(ordered_covariance, axis=0)
    ordered_ground_truth_target = ground_truth_target[:, order].astype(np.float64)
    cumulative_ground_truth_target = np.cumsum(ordered_ground_truth_target, axis=1)

    aggregate_mean = np.empty((len(user_sizes), analysis_horizon), dtype=np.float64)
    aggregate_covariance = np.empty(
        (len(user_sizes), analysis_horizon, analysis_horizon), dtype=np.float64
    )
    aggregate_ground_truth_target = np.empty(
        (len(user_sizes), ground_truth_samples), dtype=np.float64
    )
    metric_rows: list[dict[str, object]] = []
    bootstrap_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 991, bootstrap_samples])
    )

    for size_index, user_count in enumerate(user_sizes):
        last = int(user_count) - 1
        aggregate_mean[size_index] = cumulative_mean[last] / user_count
        aggregate_covariance[size_index] = (
            cumulative_covariance[last] / float(user_count**2)
        )
        samples = cumulative_ground_truth_target[:, last] / user_count
        aggregate_ground_truth_target[size_index] = samples
        theoretical_mean = float(aggregate_mean[size_index, target_index])
        theoretical_variance = max(
            float(aggregate_covariance[size_index, target_index, target_index]), 0.0
        )
        theoretical_std = math.sqrt(theoretical_variance)
        w1 = empirical_to_normal_wasserstein(
            samples, theoretical_mean, theoretical_std
        )
        ci_low, ci_high = bootstrap_wasserstein_interval(
            samples,
            theoretical_mean,
            theoretical_std,
            bootstrap_samples,
            bootstrap_rng,
        )
        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples, ddof=0))
        normal_quantiles = theoretical_mean + theoretical_std * ndtri(
            (np.arange(ground_truth_samples) + 0.5) / ground_truth_samples
        )
        qq_correlation = float(
            np.corrcoef(np.sort(samples), normal_quantiles)[0, 1]
        )
        standardized_w1 = w1 / theoretical_std if theoretical_std > 0.0 else math.nan
        interval_low = theoretical_mean - 1.96 * theoretical_std
        interval_high = theoretical_mean + 1.96 * theoretical_std
        coverage = float(np.mean((samples >= interval_low) & (samples <= interval_high)))
        metric_rows.append(
            {
                "user_count": int(user_count),
                "monte_carlo_samples": ground_truth_samples,
                "theoretical_mean_kw_per_user": theoretical_mean,
                "monte_carlo_mean_kw_per_user": sample_mean,
                "absolute_mean_error_kw_per_user": abs(sample_mean - theoretical_mean),
                "theoretical_std_kw_per_user": theoretical_std,
                "monte_carlo_std_kw_per_user": sample_std,
                "monte_carlo_to_theoretical_std_ratio": (
                    sample_std / theoretical_std if theoretical_std > 0.0 else math.nan
                ),
                "wasserstein_1_kw_per_user": w1,
                "wasserstein_bootstrap_ci_low": ci_low,
                "wasserstein_bootstrap_ci_high": ci_high,
                "standardized_wasserstein_1": standardized_w1,
                "skewness": float(skew(samples, bias=False)),
                "excess_kurtosis": float(kurtosis(samples, fisher=True, bias=False)),
                "qq_correlation": qq_correlation,
                "gaussian_95_interval_coverage": coverage,
            }
        )

    write_csv(output_dir / "wasserstein_by_user_count.csv", metric_rows)
    write_csv(output_dir / "sampling_diagnostics_by_node.csv", sampling_diagnostics)

    registry_ordered = registry.sort_values("global_position")
    np.savez_compressed(
        output_dir / "individual_moments.npz",
        user_id=registry_ordered["user_id"].to_numpy(dtype=np.int32),
        node_id=registry_ordered["node_id"].to_numpy(dtype=np.int16),
        archetype_id=registry_ordered["archetype_id"].to_numpy(dtype=np.int16),
        mean_deviation_kw=individual_mean.astype(np.float32),
        covariance_deviation_kw2=individual_covariance.astype(np.float32),
        half_sample_mean_target_kw=half_mean_target.astype(np.float32),
        half_sample_variance_target_kw2=half_variance_target.astype(np.float32),
        moment_samples=np.asarray(moment_samples, dtype=np.int32),
        target_hour_index=np.asarray(target_index, dtype=np.int16),
        selected_date=np.asarray(str(selected_date)),
    )
    np.savez_compressed(
        output_dir / "aggregate_mc_samples.npz",
        user_count=user_sizes,
        target_hour_index=np.asarray(target_index, dtype=np.int16),
        selected_date=np.asarray(str(selected_date)),
        aggregate_ground_truth_target_kw_per_user=aggregate_ground_truth_target.astype(
            np.float32
        ),
        aggregate_ground_truth_full_cohort_kw_per_user=(
            ground_truth_full_sum / total_users
        ).astype(np.float32),
        aggregate_gaussian_mean_kw=aggregate_mean.astype(np.float32),
        aggregate_gaussian_covariance_kw2=aggregate_covariance.astype(np.float32),
        selection_order_global_position=order.astype(np.int32),
        ground_truth_samples=np.asarray(ground_truth_samples, dtype=np.int32),
    )

    full_samples = aggregate_ground_truth_target[-1]
    full_mean = float(aggregate_mean[-1, target_index])
    full_std = math.sqrt(
        max(float(aggregate_covariance[-1, target_index, target_index]), 0.0)
    )
    density_size_index = 0
    density_samples = aggregate_ground_truth_target[density_size_index]
    density_mean = float(aggregate_mean[density_size_index, target_index])
    density_std = math.sqrt(
        max(
            float(
                aggregate_covariance[
                    density_size_index, target_index, target_index
                ]
            ),
            0.0,
        )
    )
    if write_figures:
        make_figure(
            density_samples,
            density_mean,
            density_std,
            int(user_sizes[density_size_index]),
            metric_rows,
            figure_base,
            output_dir / "figure_density_source.csv",
            include_density_figure,
            include_wasserstein_figure,
        )

    half_aggregate_mean = float(np.mean(half_mean_target))
    half_aggregate_variance = float(np.sum(half_variance_target) / total_users**2)
    full_aggregate_mean = full_mean
    full_aggregate_variance = float(
        aggregate_covariance[-1, target_index, target_index]
    )
    mean_convergence_relative_change = abs(
        full_aggregate_mean - half_aggregate_mean
    ) / max(abs(full_aggregate_mean), full_std, 1.0e-12)
    std_convergence_relative_change = abs(
        math.sqrt(max(full_aggregate_variance, 0.0))
        - math.sqrt(max(half_aggregate_variance, 0.0))
    ) / max(math.sqrt(max(full_aggregate_variance, 0.0)), 1.0e-12)
    raw_w1_values = np.asarray(
        [float(row["wasserstein_1_kw_per_user"]) for row in metric_rows]
    )
    standardized_w1_values = np.asarray(
        [float(row["standardized_wasserstein_1"]) for row in metric_rows]
    )
    if len(user_sizes) > 1:
        raw_rho = float(spearmanr(user_sizes, raw_w1_values).statistic)
        standardized_rho = float(
            spearmanr(user_sizes, standardized_w1_values).statistic
        )
    else:
        raw_rho = math.nan
        standardized_rho = math.nan
    total_fallbacks = int(
        sum(int(row["fallback_transitions"]) for row in sampling_diagnostics)
    )
    total_transitions = int(
        sum(int(row["transitions"]) for row in sampling_diagnostics)
    )
    max_residual = max(
        float(row["maximum_dynamics_residual_c"]) for row in sampling_diagnostics
    )
    mean_removed_mass = float(
        np.average(
            [float(row["mean_pmf_mass_removed"]) for row in sampling_diagnostics],
            weights=[int(row["transitions"]) for row in sampling_diagnostics],
        )
    )
    solved_inaccurate_batches = int(
        sum(int(row["solved_inaccurate_batches"]) for row in sampling_diagnostics)
    )
    maximum_temperature_violation = max(
        float(row["maximum_temperature_violation_c"])
        for row in sampling_diagnostics
    )
    maximum_actual_temperature_excursion = max(
        float(row.get("maximum_actual_temperature_outside_comfort_c", 0.0))
        for row in sampling_diagnostics
    )
    actual_temperature_outside_rate = float(
        np.average(
            [
                float(row.get("actual_temperature_outside_comfort_rate", 0.0))
                for row in sampling_diagnostics
            ],
            weights=[int(row["transitions"]) for row in sampling_diagnostics],
        )
    )
    power_saturation_rate = float(
        np.average(
            [float(row.get("power_saturation_rate", 0.0)) for row in sampling_diagnostics],
            weights=[int(row["transitions"]) for row in sampling_diagnostics],
        )
    )
    maximum_power_violation = max(
        float(row["maximum_power_violation_kw"])
        for row in sampling_diagnostics
    )
    zero_identity_values = np.asarray(
        [
            float(row["zero_deviation_max_abs_power_error_kw"])
            for row in sampling_diagnostics
        ],
        dtype=np.float64,
    )
    maximum_zero_deviation_power_error = (
        float(np.nanmax(zero_identity_values))
        if np.any(np.isfinite(zero_identity_values))
        else math.nan
    )
    conditioning_total = total_users * analysis_horizon
    conditioning_low = sum(block.outside_low_count for block in blocks)
    conditioning_high = sum(block.outside_high_count for block in blocks)

    confidence = "SOLID"
    warnings: list[str] = []
    if mean_convergence_relative_change > 0.05 or std_convergence_relative_change > 0.05:
        confidence = "CAUTION"
        warnings.append(
            "The 100-to-200 moment-sample convergence change exceeds 5%."
        )
    if total_fallbacks > 0:
        confidence = "CAUTION"
        warnings.append(
            "Some temperature-PMF transitions required physical-boundary fallback."
        )
    if conditioning_low + conditioning_high > 0:
        warnings.append(
            "Some nominal optimal temperatures were outside the empirical "
            "23--25 degC conditioning range and used endpoint PMFs."
        )
    if len(user_sizes) > 1 and standardized_rho >= 0.0:
        confidence = "CAUTION"
        warnings.append(
            "The standardized Wasserstein distance does not show an overall "
            "decreasing rank trend."
        )

    elapsed = time.perf_counter() - start_time
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "generator": "Codes/analyze_node_level_distribution_accuracy.py",
        "material_passport": {
            "origin_skill": ["experiment-agent", "nature-figure"],
            "origin_mode": "run",
            "origin_date": "2026-09-17",
            "verification_status": "ANALYZED",
            "version_label": "node_distribution_validation_v1",
        },
        "configuration": {
            **input_metadata,
            "moment_samples": moment_samples,
            "ground_truth_samples": ground_truth_samples,
            "user_sizes": user_sizes.tolist(),
            "random_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "gurobi_bound_relaxation": MODEL_BOUND_RELAXATION,
            "density_figure_user_count": int(user_sizes[0]),
            "aggregation_unit": "kW per user (mean-field normalization in Eq. 11)",
            "reference_mapping": reference_mapping,
            "conditioning_basis": conditioning_basis,
            "raw_feasible_day_only": raw_feasible_day_only,
            "full_population_users": len(full_registry),
            "analysis_cohort_users": total_users,
            "figures_written": write_figures,
            "density_figure_written": write_figures and include_density_figure,
            "wasserstein_figure_written": (
                write_figures and include_wasserstein_figure
            ),
            "response_objective": response_objective,
            "temporal_sampling": temporal_sampling,
            "event_duration_hours": event_duration_hours,
            "response_horizon_hours": analysis_horizon,
            "active_start_hour_inclusive": ACTIVE_START_HOUR,
            "active_stop_hour_exclusive": ACTIVE_STOP_HOUR,
        },
        "behavioral_to_power_mapping": {
            "conditioning": (
                "Linear interpolation of each user's empirical conditional PMF "
                "at the nominal comfort-reference setpoint."
                if conditioning_basis == "nominal-reference"
                else "Linear interpolation of each user's empirical conditional PMF "
                "at the optimized indoor state; endpoint PMFs outside the "
                "empirical 23--25 degC conditioning range."
            ),
            "support_interpretation": (
                "The Ecobee PMF mass is preserved exactly and re-indexed from "
                "P(actual | preset) to P(actual - preset | preset); sampled "
                "temperature deviations are applied to the recommended trajectory."
                if reference_mapping == "relative-delta"
                else "The sampled Ecobee actual temperature is used as an absolute reference."
            ),
            "temporal_sampling": (
                "Independent hourly Ecobee deviations are sampled from 08:00 "
                "through 19:00 conditional on the fixed optimal trajectory."
                if temporal_sampling == "active-window-hourly-independent"
                else
                "One Ecobee temperature deviation is sampled at the override onset "
                "and retained for the configured event duration; the response "
                "horizon begins at the event, so there is no pre-event anticipation."
                if temporal_sampling == "event-onset-persistent"
                else "Hourly overridden temperature references are sampled independently "
                "conditional on the fixed nominal optimal trajectory because the "
                "available empirical data provide marginal conditional PMFs."
            ),
            "actual_response": (
                "The sampled temperature is treated as the intended actual state. "
                "Power is obtained directly by rearranging the thermal dynamics, "
                "saturated at each user's physical power bounds, and the realized "
                "temperature is propagated without imposing the nominal comfort band."
                if response_objective == "direct-dynamics"
                else
                "At the override onset, solve the remaining-horizon convex "
                "temperature-tracking problem with unchanged thermal dynamics, "
                "temperature bounds, and cooling limits; the electricity-price "
                "term is omitted so that a zero temperature deviation recovers "
                "the nominal optimal trajectory within solver/export tolerance."
                if response_objective == "temperature-tracking"
                else "For every sampled reference profile, re-solve the convex "
                "TCL problem with unchanged thermal dynamics, electricity price, "
                "temperature bounds, cooling limits, and comfort penalty."
            ),
            "power_deviation": "actual physically feasible power minus cached nominal optimal power",
        },
        "diagnostics": {
            "overall_confidence": confidence,
            "warnings": warnings,
            "moment_mean_relative_change_100_to_200": mean_convergence_relative_change,
            "moment_std_relative_change_100_to_200": std_convergence_relative_change,
            "physical_fallback_transitions": total_fallbacks,
            "physical_fallback_rate": total_fallbacks / total_transitions,
            "gurobi_nonoptimal_batches": solved_inaccurate_batches,
            "maximum_temperature_violation_c": maximum_temperature_violation,
            "maximum_actual_temperature_outside_comfort_c": (
                maximum_actual_temperature_excursion
            ),
            "actual_temperature_outside_comfort_rate": (
                actual_temperature_outside_rate
            ),
            "power_saturation_rate": power_saturation_rate,
            "maximum_power_violation_kw": maximum_power_violation,
            "maximum_dynamics_residual_c": max_residual,
            "maximum_zero_deviation_power_error_kw": (
                maximum_zero_deviation_power_error
            ),
            "mean_pmf_mass_removed_by_physical_filter": mean_removed_mass,
            "conditioning_below_empirical_range": conditioning_low,
            "conditioning_above_empirical_range": conditioning_high,
            "conditioning_outside_empirical_range_rate": (
                conditioning_low + conditioning_high
            )
            / conditioning_total,
            "raw_wasserstein_spearman_rho_vs_user_count": raw_rho,
            "standardized_wasserstein_spearman_rho_vs_user_count": standardized_rho,
            "full_cohort_metrics": metric_rows[-1],
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "gurobi": ".".join(str(value) for value in gp.gurobi.version()),
        },
        "figure_contract": {
            "core_conclusion": (
                "The Eq. (15)--(16) Gaussian aggregate distribution reproduces "
                "the independent Monte Carlo aggregate distribution, with "
                "approximation behavior assessed as the population grows."
            ),
            "archetype": "quantitative grid",
            "density_figure": "300-user Monte Carlo density versus proposed Gaussian PDF",
            "wasserstein_figure": (
                "1-Wasserstein sensitivity over the configured cohort sizes"
                if include_wasserstein_figure
                else "not generated in density-only mode"
            ),
            "backend": "Python/matplotlib only",
            "export": (
                "Standalone 3.6 x 3.05 in vector PDF Wasserstein figure"
                if include_wasserstein_figure and not include_density_figure
                else "Compact standalone vector PDF density figure"
                if include_density_figure and not include_wasserstein_figure
                else "Two standalone 3.6 x 3.05 in vector PDF figures"
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None"
    full_metric = metric_rows[-1]
    report = f"""## Material Passport

- Origin Skill: experiment-agent; nature-figure
- Origin Mode: run + analysis
- Origin Date: 2026-09-17
- Verification Status: ANALYZED
- Version Label: node_distribution_validation_v1

## Validation Report

- **Source**: cached 3,000-user nominal solutions and user conditional PMFs
- **Overall Confidence**: {confidence}
- **Selected case**: {str(selected_date)} at {selected_hour:02d}:00
- **Ground-truth samples**: {ground_truth_samples}
- **Individual moment samples**: {moment_samples}

### Primary {total_users}-user cohort result

| Metric | Value |
|---|---:|
| Proposed mean (kW/user) | {float(full_metric['theoretical_mean_kw_per_user']):.8f} |
| Monte Carlo mean (kW/user) | {float(full_metric['monte_carlo_mean_kw_per_user']):.8f} |
| Proposed standard deviation (kW/user) | {float(full_metric['theoretical_std_kw_per_user']):.8f} |
| Monte Carlo standard deviation (kW/user) | {float(full_metric['monte_carlo_std_kw_per_user']):.8f} |
| 1-Wasserstein distance (kW/user) | {float(full_metric['wasserstein_1_kw_per_user']):.8f} |
| Standardized 1-Wasserstein distance | {float(full_metric['standardized_wasserstein_1']):.6f} |
| Q-Q correlation | {float(full_metric['qq_correlation']):.6f} |
| Gaussian 95% interval coverage | {float(full_metric['gaussian_95_interval_coverage']):.4f} |

### Reproducibility and numerical checks

- 100-to-200 moment mean relative change: {mean_convergence_relative_change:.4%}.
- 100-to-200 moment standard-deviation relative change: {std_convergence_relative_change:.4%}.
- Gurobi numerical bound relaxation: {MODEL_BOUND_RELAXATION:.3e}.
- Gurobi batches with non-optimal status: {solved_inaccurate_batches:,}.
- Maximum actual-temperature excursion beyond the nominal comfort band:
  {maximum_actual_temperature_excursion:.3e} degC (allowed by design).
- Actual-temperature samples outside the nominal comfort band:
  {100.0 * actual_temperature_outside_rate:.4f}%.
- Power-saturation rate: {100.0 * power_saturation_rate:.4f}%.
- Maximum cooling-power violation: {maximum_power_violation:.3e} kW.
- Maximum thermal-dynamics residual: {max_residual:.3e} degC.
- Maximum zero-deviation identity error: {maximum_zero_deviation_power_error:.3e} kW.
- Conditioning temperatures outside the empirical 23--25 degC range:
  {conditioning_low + conditioning_high:,} / {conditioning_total:,}
  ({100.0 * (conditioning_low + conditioning_high) / conditioning_total:.4f}%).
- Raw-Wasserstein Spearman rho versus user count: {raw_rho:.4f}.
- Standardized-Wasserstein Spearman rho versus user count: {standardized_rho:.4f}.

### Warnings

{warning_lines}

### Interpretation boundary

The raw Wasserstein distance is the manuscript-requested primary sensitivity
metric. Because the mean-field distribution contracts as the user count grows,
the standardized Wasserstein distance is also reported to separate shape
convergence from scale contraction. No monotonic smoothing or outcome-enforcing
post-processing was applied.
"""
    (output_dir / "validation_report.md").write_text(report, encoding="utf-8")

    print(
        f"Completed node-level aggregate-distribution validation in {elapsed:.1f} s; "
        f"confidence={confidence}.",
        flush=True,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the node-level aggregate Gaussian approximation."
    )
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--date", default=DEFAULT_DATE, help="Selected day, YYYY-MM-DD.")
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR, choices=range(24))
    parser.add_argument(
        "--moment-samples", type=int, default=DEFAULT_MOMENT_SAMPLES
    )
    parser.add_argument(
        "--ground-truth-samples", type=int, default=DEFAULT_GROUND_TRUTH_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "Outputs" / "Node-level Aggregate Distribution Event Tracking",
    )
    parser.add_argument(
        "--figure-base",
        type=Path,
        default=project / "Figures" / "node_level_aggregate_distribution_validation",
    )
    parser.add_argument(
        "--reference-mapping",
        choices=("absolute", "relative-delta"),
        default="relative-delta",
        help=(
            "Interpret the Ecobee support as absolute actual temperature or "
            "re-index its unchanged probability mass as actual-minus-preset."
        ),
    )
    parser.add_argument(
        "--analysis-users",
        type=int,
        default=300,
        help="Node-stratified cohort size (300, 600, ..., 3000).",
    )
    parser.add_argument(
        "--conditioning-basis",
        choices=("optimal-state", "nominal-reference"),
        default="optimal-state",
        help=(
            "Condition the Ecobee actual-given-preset PMF on the optimized indoor "
            "state (legacy) or on the nominal comfort-reference setpoint."
        ),
    )
    parser.add_argument(
        "--raw-feasible-day-only",
        action="store_true",
        help=(
            "Restrict the cohort to users whose unprojected disturbance is "
            "physically feasible for the selected day."
        ),
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Run numerical diagnostics without creating PDF figures.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw PDFs from registered saved results without rerunning analysis.",
    )
    parser.add_argument(
        "--density-only",
        action="store_true",
        default=True,
        help="Create only the density PDF, not the Wasserstein PDF.",
    )
    parser.add_argument(
        "--with-wasserstein",
        action="store_false",
        dest="density_only",
        help="Also create the Wasserstein population-sensitivity PDF.",
    )
    parser.add_argument(
        "--wasserstein-only",
        action="store_true",
        help="Create only the Wasserstein PDF and leave the density PDF untouched.",
    )
    parser.add_argument(
        "--response-objective",
        choices=("price-responsive", "temperature-tracking", "direct-dynamics"),
        default="direct-dynamics",
        help=(
            "Use the direct thermal-dynamics mapping (recommended), re-optimize "
            "price plus comfort (legacy), or solve temperature tracking (legacy)."
        ),
    )
    parser.add_argument(
        "--temporal-sampling",
        choices=(
            "hourly-independent",
            "event-onset-persistent",
            "active-window-hourly-independent",
        ),
        default="active-window-hourly-independent",
    )
    parser.add_argument(
        "--event-duration-hours",
        type=int,
        default=1,
        help="Persistence of a sampled override in event-onset mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.plot_only:
        make_figure_from_saved_results(
            output_dir=args.output_dir.resolve(),
            figure_base=args.figure_base.resolve(),
            include_density_figure=not args.wasserstein_only,
            include_wasserstein_figure=(not args.density_only) or args.wasserstein_only,
        )
        return
    if args.moment_samples < 2:
        parser.error("--moment-samples must be at least 2.")
    if args.ground_truth_samples < 20:
        parser.error("--ground-truth-samples must be at least 20.")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100.")
    if args.analysis_users not in USER_SIZES:
        parser.error("--analysis-users must be one of 300, 600, ..., 3000.")
    if args.event_duration_hours < 1 or args.event_duration_hours > HORIZON:
        parser.error("--event-duration-hours must be between 1 and 24.")
    selected_date = parse_iso_date(args.date)
    project = Path(__file__).resolve().parents[1]
    run_analysis(
        project=project,
        selected_date=selected_date,
        selected_hour=args.hour,
        moment_samples=args.moment_samples,
        ground_truth_samples=args.ground_truth_samples,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=args.output_dir.resolve(),
        figure_base=args.figure_base.resolve(),
        reference_mapping=args.reference_mapping,
        conditioning_basis=args.conditioning_basis,
        analysis_users=args.analysis_users,
        raw_feasible_day_only=args.raw_feasible_day_only,
        write_figures=not args.skip_figures,
        include_wasserstein_figure=(not args.density_only) or args.wasserstein_only,
        include_density_figure=not args.wasserstein_only,
        response_objective=args.response_objective,
        temporal_sampling=args.temporal_sampling,
        event_duration_hours=args.event_duration_hours,
    )


if __name__ == "__main__":
    main()
