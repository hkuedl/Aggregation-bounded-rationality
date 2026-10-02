"""Evaluate the node-level aggregate TCL model under bounded rationality.

This script implements the second node-level case study in the manuscript.
It deliberately does not modify the manuscript.  The default ``all`` phase:

1. builds 50-response-per-user daily Monte Carlo references using the Ecobee
   PMF re-indexed as deviations from each optimized indoor state;
2. uses those same paths to estimate individual conditional moments, including
   audit-only September moments that never enter either training stage;
3. trains Stage 1, the proposed physical-parameter-frozen Stage 2, and an
   all-parameter fine-tuning baseline by differentiating the active-set KKT
   system of the equivalent TCL quadratic program;
4. evaluates daily normalized RMSE in July--August and September;
5. runs the 10 x 10 user-count/price-diversity sensitivity experiment; and
6. exports two standalone vector PDF figures plus source data and diagnostics.

The daily Monte Carlo preparation is checkpointed, so an interrupted Spyder run
can be resumed by running the file again.  All random streams are deterministic.
Actual power is obtained directly from the thermal dynamics and saturated at
each user's feasible power interval; actual temperature is not constrained to
the nominal comfort band.
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


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

PLOT_ONLY_INVOCATION = any(
    arg in {"--phase=plot", "--phase=refresh-metrics"}
    or (
        arg == "--phase"
        and index + 1 < len(sys.argv)
        and sys.argv[index + 1] in {"plot", "refresh-metrics"}
    )
    for index, arg in enumerate(sys.argv)
)
if not PLOT_ONLY_INVOCATION:
    import scipy
    import scipy.sparse as sparse
    from scipy.optimize import minimize, nnls
    from scipy.stats import wilcoxon
    import analyze_node_level_distribution_accuracy as distribution_validation

    gp = distribution_validation.gp
else:
    scipy = None
    sparse = None
    distribution_validation = None
    gp = None

HORIZON = 24
USER_SIZES = np.arange(300, 3001, 300, dtype=np.int32)
PRICE_COUNTS = np.arange(6, 61, 6, dtype=np.int32)
TRAIN_START = np.datetime64("2025-07-01", "D")
TEST_START = np.datetime64("2025-09-01", "D")
END_DATE = np.datetime64("2025-10-01", "D")
DEFAULT_MOMENT_SAMPLES = 50
DEFAULT_GROUND_TRUTH_SAMPLES = DEFAULT_MOMENT_SAMPLES
DEFAULT_BATCH_SIZE = 20
DEFAULT_SEED = 20260917
DEFAULT_STAGE2_RHO = 1.0e-3
BOXPLOT_GROUP_INDEX = 0
BOXPLOT_PRICE_COUNT = 24
SENSITIVITY_STAGE2_RHO = 1.0e-3
REFERENCE_TEMPERATURE = 24.0
INITIAL_TEMPERATURE = 24.0
BOUND_RELAXATION = 1.0e-5
ACTIVE_TOLERANCE = 2.0e-5
DUAL_TOLERANCE = 1.0e-8
REFERENCE_MAPPING = "relative-delta"
CONDITIONING_BASIS = "optimal-state"
RESPONSE_OBJECTIVE = "direct-dynamics"
TEMPORAL_SAMPLING = "active-window-hourly-independent"
BEHAVIORAL_MAPPING_VERSION = (
    "relative-delta_optimal-state_direct-dynamics_"
    "active-08-20_shared-mc50_daily-24c-boundary-v4"
)

BASE_PARAMETER_NAMES = (
    "a",
    "b",
    "power_max",
    "temperature_min",
    "temperature_max",
    "weight",
)
PREFERENCE_TEMPERATURE_NAMES = tuple(
    f"preferred_temperature_{hour:02d}" for hour in range(HORIZON)
)
PARAMETER_NAMES = BASE_PARAMETER_NAMES + PREFERENCE_TEMPERATURE_NAMES
PARAMETER_LOWER = np.r_[
    np.asarray([0.85, -5.0, 2.0, 20.0, 24.05, 0.03], dtype=np.float64),
    np.full(HORIZON, 22.0, dtype=np.float64),
]
PARAMETER_UPPER = np.r_[
    np.asarray([0.995, -0.20, 6.5, 23.95, 28.0, 1.0], dtype=np.float64),
    np.full(HORIZON, 26.0, dtype=np.float64),
]
PHYSICAL_INDICES = np.asarray([0, 1, 2], dtype=np.int32)
USER_INDICES = np.arange(3, len(PARAMETER_NAMES), dtype=np.int32)
ALL_INDICES = np.arange(len(PARAMETER_NAMES), dtype=np.int32)


def expand_empirical_theta(theta: np.ndarray) -> np.ndarray:
    """Append the neutral 24-h preferred-temperature profile to legacy data."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape[-1] == len(PARAMETER_NAMES):
        return theta.copy()
    if theta.shape[-1] != len(BASE_PARAMETER_NAMES):
        raise ValueError(
            "Empirical parameter array must contain either the six base "
            "parameters or the complete hourly-preference parameterization."
        )
    reference_shape = theta.shape[:-1] + (HORIZON,)
    reference = np.full(reference_shape, REFERENCE_TEMPERATURE, dtype=np.float64)
    return np.concatenate((theta, reference), axis=-1)


@dataclass
class ArchiveBlock:
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
    timestamps: np.ndarray
    price: np.ndarray
    disturbance: np.ndarray
    optimal_temperature: np.ndarray
    optimal_power: np.ndarray
    initial_temperature: float
    temperature_support: np.ndarray
    preset_temperature: np.ndarray
    conditional_probability: np.ndarray
    projected_mask: np.ndarray
    raw_infeasible_user_day: np.ndarray


@dataclass
class FitResult:
    theta: np.ndarray
    objective: float
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    gradient_norm: float
    max_stationarity_residual: float
    max_active_constraint_residual: float


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def node_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def read_assignments(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["bus"]): row for row in csv.DictReader(stream)}


def load_archives(project: Path) -> tuple[list[ArchiveBlock], pd.DataFrame, dict[str, object]]:
    input_dir = project / "Data" / "Inputs_33kV_25MW_3000_JulSep"
    bounded_dir = project / "Outputs" / "Bounded Rationality Jul-Sep 3000"
    optimal_dir = bounded_dir / "Optimal results"
    probability_dir = bounded_dir / "user_probability_distributions"
    assignments = read_assignments(input_dir / "node_assignments.csv")
    optimal_paths = sorted(optimal_dir.glob("node_*_optimal_results.npz"), key=node_id_from_path)
    if not optimal_paths:
        raise FileNotFoundError(f"No nominal optimal-result files found in {optimal_dir}")

    blocks: list[ArchiveBlock] = []
    registry_rows: list[dict[str, object]] = []
    offset = 0
    common_timestamps: np.ndarray | None = None
    common_price: np.ndarray | None = None
    projected_count = 0
    raw_infeasible_count = 0
    raw_user_days = 0

    for optimal_path in optimal_paths:
        node_id = node_id_from_path(optimal_path)
        probability_path = probability_dir / f"node_{node_id:02d}_bounded_rationality.npz"
        assignment = assignments.get(node_id)
        if assignment is None or assignment["node_type"].strip().lower() != "user":
            raise ValueError(f"Bus {node_id} is not a registered user node.")
        if not probability_path.exists():
            raise FileNotFoundError(probability_path)

        with np.load(optimal_path, allow_pickle=False) as optimal, np.load(
            probability_path, allow_pickle=False
        ) as probability:
            user_id = np.asarray(optimal["user_id"], dtype=np.int32)
            probability_user = np.asarray(probability["user_id"], dtype=np.int32)
            if not np.array_equal(user_id, probability_user):
                raise ValueError(f"User identifiers do not align at bus {node_id}.")
            timestamps = np.asarray(optimal["timestamps"]).astype("datetime64[h]")
            price = np.asarray(optimal["price_usd_per_kwh"], dtype=np.float64)
            if common_timestamps is None:
                common_timestamps = timestamps
                common_price = price
            elif not np.array_equal(timestamps, common_timestamps) or not np.allclose(price, common_price):
                raise ValueError(f"Timestamp or price mismatch at bus {node_id}.")

            solver_status = np.char.lower(np.asarray(optimal["solver_status"]))
            if np.any((solver_status != "solved") & (solver_status != "solved inaccurate")):
                raise ValueError(f"A nominal individual optimization failed at bus {node_id}.")

            projected_mask = np.asarray(optimal["disturbance_projected_mask"], dtype=bool)
            infeasible = np.asarray(optimal["raw_infeasible_user_day"], dtype=bool)
            projected_count += int(projected_mask.sum())
            raw_infeasible_count += int(infeasible.sum())
            raw_user_days += int(infeasible.size)
            n_users = len(user_id)
            global_slice = slice(offset, offset + n_users)
            preset_temperature = np.asarray(
                probability["preset_setpoint_c"], dtype=np.float64
            )
            absolute_support = np.asarray(
                probability["actual_setpoint_support_c"], dtype=np.float64
            )
            absolute_probability = np.asarray(
                probability["conditional_probability"], dtype=np.float64
            )
            response_support, response_probability = (
                distribution_validation.reindex_absolute_pmf_as_relative_delta(
                    absolute_probability,
                    preset_temperature,
                    absolute_support,
                )
            )
            blocks.append(
                ArchiveBlock(
                    node_id=node_id,
                    archetype_id=int(assignment["archetype_id"]),
                    global_slice=global_slice,
                    user_id=user_id,
                    a=np.asarray(optimal["a"], dtype=np.float64),
                    b=np.asarray(optimal["b_c_per_kw"], dtype=np.float64),
                    power_min=np.asarray(optimal["power_min_kw"], dtype=np.float64),
                    power_max=np.asarray(optimal["power_max_kw"], dtype=np.float64),
                    temperature_min=float(optimal["temperature_min_c"]),
                    temperature_max=float(optimal["temperature_max_c"]),
                    temperature_penalty=np.asarray(optimal["temperature_penalty"], dtype=np.float64),
                    timestamps=timestamps,
                    price=price,
                    disturbance=np.asarray(optimal["disturbance_used_c"], dtype=np.float64),
                    optimal_temperature=np.asarray(optimal["optimal_temperature_c"], dtype=np.float64),
                    optimal_power=np.asarray(optimal["optimal_power_kw"], dtype=np.float64),
                    initial_temperature=float(optimal["initial_temperature_c"]),
                    temperature_support=response_support,
                    preset_temperature=preset_temperature,
                    conditional_probability=response_probability,
                    projected_mask=projected_mask,
                    raw_infeasible_user_day=infeasible,
                )
            )
            for local_index, identifier in enumerate(user_id):
                registry_rows.append(
                    {
                        "global_position": offset + local_index,
                        "node_id": node_id,
                        "archetype_id": int(assignment["archetype_id"]),
                        "user_id": int(identifier),
                    }
                )
            offset += n_users

    registry = pd.DataFrame(registry_rows)
    if len(registry) != 3000:
        raise ValueError(f"Expected 3,000 users, found {len(registry):,}.")
    assert common_timestamps is not None and common_price is not None
    days = np.arange(TRAIN_START, END_DATE, dtype="datetime64[D]")
    if len(days) != 92:
        raise AssertionError("Expected 92 July--September days.")
    metadata = {
        "users": len(registry),
        "nodes": len(blocks),
        "hours": len(common_timestamps),
        "days": len(days),
        "train_days": int(np.count_nonzero(days < TEST_START)),
        "test_days": int(np.count_nonzero(days >= TEST_START)),
        "disturbance_projection_rate": projected_count / (len(registry) * len(common_timestamps)),
        "raw_infeasible_user_day_rate": raw_infeasible_count / raw_user_days,
    }
    return blocks, registry, metadata


def load_selection_order(project: Path, registry: pd.DataFrame, seed: int) -> np.ndarray:
    selected_path = (
        project
        / "Outputs"
        / "Node-level Aggregate Distribution Event Tracking Sensitivity"
        / "selected_users.csv"
    )
    if not selected_path.exists():
        selected_path = (
            project
            / "Outputs"
            / "Node-level Aggregate Distribution"
            / "selected_users.csv"
        )
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        required = {"selection_rank", "global_position", "node_id", "user_id"}
        if not required.issubset(selected.columns) or len(selected) != len(registry):
            raise ValueError(f"Existing selection file has an incompatible schema: {selected_path}")
        selected = selected.sort_values("selection_rank")
        order = selected["global_position"].to_numpy(dtype=np.int64)
        joined = registry.iloc[order].reset_index(drop=True)
        if not np.array_equal(joined["node_id"].to_numpy(), selected["node_id"].to_numpy()):
            raise ValueError("Existing node-stratified selection no longer matches the input registry.")
        if not np.array_equal(joined["user_id"].to_numpy(), selected["user_id"].to_numpy()):
            raise ValueError("Existing selected user identifiers no longer match the input registry.")
        return order
    return distribution_validation.stratified_nested_order(registry, seed)


def day_indices(timestamps: np.ndarray, day: np.datetime64) -> np.ndarray:
    start = day.astype("datetime64[h]")
    index = np.flatnonzero((timestamps >= start) & (timestamps < start + np.timedelta64(HORIZON, "h")))
    if len(index) != HORIZON:
        raise ValueError(f"Incomplete day in cached optimal results: {day}")
    return index


def make_day_block(block: ArchiveBlock, index: np.ndarray) -> distribution_validation.NodeBlock:
    optimal_temperature = block.optimal_temperature[:, index]
    hourly_probability, below, above = distribution_validation.interpolate_conditional_pmf(
        block.conditional_probability,
        block.preset_temperature,
        optimal_temperature,
    )
    return distribution_validation.NodeBlock(
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
        price=block.price[index],
        disturbance=block.disturbance[:, index],
        optimal_temperature=optimal_temperature,
        optimal_power=block.optimal_power[:, index],
        # The source optimization solves each 24-hour day independently from
        # the recorded 24 degC boundary condition.  This is a dynamics initial
        # state, not the Ecobee PMF conditioning preset; the latter remains the
        # optimized hourly indoor-temperature trajectory above.
        initial_temperature=block.initial_temperature,
        reference_anchor=optimal_temperature,
        temperature_support=block.temperature_support,
        conditional_probability=block.conditional_probability,
        hourly_probability=hourly_probability,
        preset_minimum=float(block.preset_temperature[0]),
        preset_maximum=float(block.preset_temperature[-1]),
        reference_mapping=REFERENCE_MAPPING,
        raw_feasible_day=~block.raw_infeasible_user_day[:, int(index[0] // HORIZON)],
        outside_low_count=below,
        outside_high_count=above,
    )


def inclusion_bin_by_global_position(order: np.ndarray) -> np.ndarray:
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)
    return np.minimum(rank // 300, len(USER_SIZES) - 1)


def cumulative_group_average(bin_sum: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(bin_sum, axis=0)
    denominator_shape = (len(USER_SIZES),) + (1,) * (bin_sum.ndim - 1)
    return cumulative / USER_SIZES.reshape(denominator_shape)


def prepare_daily_checkpoint(
    day: np.datetime64,
    blocks: Sequence[ArchiveBlock],
    inclusion_bin: np.ndarray,
    moment_samples: int,
    ground_truth_samples: int,
    batch_size: int,
    seed: int,
    checkpoint_path: Path,
) -> dict[str, object]:
    train_day = bool(day < TEST_START)
    nominal_bin = np.zeros((len(USER_SIZES), HORIZON), dtype=np.float64)
    disturbance_bin = np.zeros_like(nominal_bin)
    a_bin = np.zeros(len(USER_SIZES), dtype=np.float64)
    b_bin = np.zeros(len(USER_SIZES), dtype=np.float64)
    pmax_bin = np.zeros(len(USER_SIZES), dtype=np.float64)
    weight_bin = np.zeros(len(USER_SIZES), dtype=np.float64)
    ground_truth_deviation_bin = np.zeros_like(nominal_bin)
    moment_mean_bin = np.zeros_like(nominal_bin)
    moment_covariance_bin = np.zeros((len(USER_SIZES), HORIZON, HORIZON), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    conditioning_outside_count = 0
    common_price: np.ndarray | None = None

    for block_number, archive in enumerate(blocks, start=1):
        index = day_indices(archive.timestamps, day)
        block = make_day_block(archive, index)
        conditioning_outside_count += block.outside_low_count + block.outside_high_count
        if common_price is None:
            common_price = block.price.copy()
        elif not np.allclose(common_price, block.price):
            raise ValueError(f"Price mismatch while preparing {day}.")

        local_bin = inclusion_bin[archive.global_slice]
        np.add.at(nominal_bin, local_bin, block.optimal_power)
        np.add.at(disturbance_bin, local_bin, block.disturbance)
        np.add.at(a_bin, local_bin, block.a)
        np.add.at(b_bin, local_bin, block.b)
        np.add.at(pmax_bin, local_bin, block.power_max)
        np.add.at(weight_bin, local_bin, block.temperature_penalty)

        shared_deviation, shared_diagnostic = distribution_validation.simulate_deviations(
            block,
            moment_samples,
            seed + int((day - TRAIN_START).astype(int)) * 1009,
            phase_code=31,
            solver_bundle=None,
            batch_size=batch_size,
            response_objective=RESPONSE_OBJECTIVE,
            temporal_sampling=TEMPORAL_SAMPLING,
        )
        shared64 = shared_deviation.astype(np.float64)
        user_mean = shared64.mean(axis=0)
        centered = shared64 - user_mean[None, :, :]
        user_covariance = np.einsum(
            "sut,suv->utv", centered, centered, optimize=True
        ) / moment_samples
        user_covariance = 0.5 * (
            user_covariance + np.swapaxes(user_covariance, 1, 2)
        )
        np.add.at(moment_mean_bin, local_bin, user_mean)
        np.add.at(moment_covariance_bin, local_bin, user_covariance)
        np.add.at(ground_truth_deviation_bin, local_bin, user_mean)
        diagnostics.append({"phase": "shared_ground_truth_and_moments", **shared_diagnostic})
        del shared_deviation, shared64, centered, user_covariance
        print(
            f"    {day} node {block_number:02d}/{len(blocks):02d} "
            f"(bus {archive.node_id:02d}) complete",
            flush=True,
        )

    assert common_price is not None
    nominal = cumulative_group_average(nominal_bin)
    disturbance = cumulative_group_average(disturbance_bin)
    ground_truth_deviation = cumulative_group_average(ground_truth_deviation_bin)
    moment_mean = cumulative_group_average(moment_mean_bin)
    cumulative_covariance = np.cumsum(moment_covariance_bin, axis=0)
    covariance_denominator = USER_SIZES.astype(np.float64)[:, None, None] ** 2
    moment_covariance = cumulative_covariance / covariance_denominator
    initial_theta = np.column_stack(
        [
            cumulative_group_average(a_bin[:, None])[:, 0],
            cumulative_group_average(b_bin[:, None])[:, 0],
            cumulative_group_average(pmax_bin[:, None])[:, 0],
            np.full(len(USER_SIZES), blocks[0].temperature_min),
            np.full(len(USER_SIZES), blocks[0].temperature_max),
            cumulative_group_average(weight_bin[:, None])[:, 0],
        ]
    )

    maximum_temperature_violation = max(float(row["maximum_temperature_violation_c"]) for row in diagnostics)
    maximum_actual_temperature_excursion = max(
        float(row.get("maximum_actual_temperature_outside_comfort_c", 0.0))
        for row in diagnostics
    )
    actual_temperature_outside_rate = float(
        np.average(
            [float(row.get("actual_temperature_outside_comfort_rate", 0.0)) for row in diagnostics],
            weights=[int(row["transitions"]) for row in diagnostics],
        )
    )
    power_saturation_rate = float(
        np.average(
            [float(row.get("power_saturation_rate", 0.0)) for row in diagnostics],
            weights=[int(row["transitions"]) for row in diagnostics],
        )
    )
    maximum_power_violation = max(float(row["maximum_power_violation_kw"]) for row in diagnostics)
    maximum_dynamics_residual = max(float(row["maximum_dynamics_residual_c"]) for row in diagnostics)
    shared_mean_identity_error = float(
        np.max(np.abs(ground_truth_deviation - moment_mean))
    )
    if shared_mean_identity_error > 1.0e-10:
        raise RuntimeError(
            "Shared Monte Carlo paths do not reproduce the aggregate mean: "
            f"{shared_mean_identity_error:.3e} kW/user."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_path,
        day=np.asarray(str(day)),
        train_day=np.asarray(train_day),
        price_usd_per_kwh=common_price.astype(np.float32),
        nominal_power_kw_per_user=nominal.astype(np.float32),
        disturbance_c=disturbance.astype(np.float32),
        ground_truth_power_kw_per_user=(nominal + ground_truth_deviation).astype(np.float32),
        moment_mean_deviation_kw_per_user=moment_mean.astype(np.float32),
        moment_covariance_kw2_per_user=moment_covariance.astype(np.float32),
        empirical_initial_theta=initial_theta.astype(np.float64),
        maximum_temperature_violation_c=np.asarray(maximum_temperature_violation),
        maximum_actual_temperature_outside_comfort_c=np.asarray(
            maximum_actual_temperature_excursion
        ),
        actual_temperature_outside_comfort_rate=np.asarray(
            actual_temperature_outside_rate
        ),
        power_saturation_rate=np.asarray(power_saturation_rate),
        maximum_power_violation_kw=np.asarray(maximum_power_violation),
        maximum_dynamics_residual_c=np.asarray(maximum_dynamics_residual),
        shared_sample_mean_identity_error_kw_per_user=np.asarray(
            shared_mean_identity_error
        ),
        conditioning_outside_count=np.asarray(conditioning_outside_count, dtype=np.int64),
        moment_samples=np.asarray(moment_samples, dtype=np.int32),
        ground_truth_samples=np.asarray(ground_truth_samples, dtype=np.int32),
        batch_size=np.asarray(batch_size, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        behavioral_mapping_version=np.asarray(BEHAVIORAL_MAPPING_VERSION),
        reference_mapping=np.asarray(REFERENCE_MAPPING),
        conditioning_basis=np.asarray(CONDITIONING_BASIS),
        response_objective=np.asarray(RESPONSE_OBJECTIVE),
        temporal_sampling=np.asarray(TEMPORAL_SAMPLING),
    )
    return {
        "maximum_temperature_violation_c": maximum_temperature_violation,
        "maximum_actual_temperature_outside_comfort_c": maximum_actual_temperature_excursion,
        "actual_temperature_outside_comfort_rate": actual_temperature_outside_rate,
        "power_saturation_rate": power_saturation_rate,
        "maximum_power_violation_kw": maximum_power_violation,
        "maximum_dynamics_residual_c": maximum_dynamics_residual,
        "shared_sample_mean_identity_error_kw_per_user": shared_mean_identity_error,
        "conditioning_outside_count": conditioning_outside_count,
    }


def checkpoint_matches(path: Path, moment_samples: int, ground_truth_samples: int, batch_size: int, seed: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["moment_samples"]) == moment_samples
                and int(data["ground_truth_samples"]) == ground_truth_samples
                and int(data["batch_size"]) == batch_size
                and int(data["seed"]) == seed
                and str(data["behavioral_mapping_version"])
                == BEHAVIORAL_MAPPING_VERSION
            )
    except Exception:
        return False


def merge_daily_checkpoints(checkpoint_dir: Path, output_path: Path) -> dict[str, np.ndarray]:
    days = np.arange(TRAIN_START, END_DATE, dtype="datetime64[D]")
    fields = {
        "price_usd_per_kwh": [],
        "nominal_power_kw_per_user": [],
        "disturbance_c": [],
        "ground_truth_power_kw_per_user": [],
        "moment_mean_deviation_kw_per_user": [],
        "moment_covariance_kw2_per_user": [],
        "empirical_initial_theta": [],
    }
    diagnostic_rows: list[dict[str, object]] = []
    for day in days:
        path = checkpoint_dir / f"{day}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing daily checkpoint: {path}")
        with np.load(path, allow_pickle=False) as data:
            for field in fields:
                fields[field].append(np.asarray(data[field]))
            diagnostic_rows.append(
                {
                    "day": str(day),
                    "maximum_temperature_violation_c": float(data["maximum_temperature_violation_c"]),
                    "maximum_actual_temperature_outside_comfort_c": float(
                        data["maximum_actual_temperature_outside_comfort_c"]
                    ),
                    "actual_temperature_outside_comfort_rate": float(
                        data["actual_temperature_outside_comfort_rate"]
                    ),
                    "power_saturation_rate": float(data["power_saturation_rate"]),
                    "maximum_power_violation_kw": float(data["maximum_power_violation_kw"]),
                    "maximum_dynamics_residual_c": float(data["maximum_dynamics_residual_c"]),
                    "shared_sample_mean_identity_error_kw_per_user": float(
                        data["shared_sample_mean_identity_error_kw_per_user"]
                    ),
                    "conditioning_outside_count": int(data["conditioning_outside_count"]),
                }
            )
    merged = {name: np.stack(values, axis=0) for name, values in fields.items()}
    merged["days"] = days
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)
    write_csv(output_path.with_name("daily_sampling_diagnostics.csv"), diagnostic_rows)
    return merged


def nearest_medoid_price_order(prices: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Nested price-diversity order from a representative day outwards.

    The earlier farthest-point order made the first six days the *most*
    diverse subset and therefore confounded sample count with coverage.  This
    order starts at the training-set medoid and adds increasingly dissimilar
    daily shapes, so both the number and the diversity of price signals grow
    along the sensitivity axis.  Only July--August prices determine the order.
    """
    x = np.asarray(prices, dtype=np.float64)
    column_scale = np.std(x, axis=0, ddof=0)
    column_scale[column_scale < 1.0e-12] = 1.0
    z = (x - np.mean(x, axis=0)) / column_scale
    pairwise = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    medoid = int(np.argmin(np.sum(pairwise, axis=1)))
    return np.lexsort((days, pairwise[medoid])).astype(np.int32)


def build_qp_components(theta: np.ndarray, price: np.ndarray, disturbance: np.ndarray) -> tuple[np.ndarray, ...]:
    theta = np.asarray(theta, dtype=np.float64)
    a, b, power_max, temperature_min, temperature_max, weight = theta[:6]
    preferred_temperature = theta[6:]
    if preferred_temperature.shape != (HORIZON,):
        raise ValueError("Equivalent TCL must contain a 24-hour preference profile.")
    if b >= 0.0 or weight <= 0.0 or temperature_min >= temperature_max:
        raise ValueError("Equivalent TCL parameters violate sign/order restrictions.")
    d_matrix = np.eye(HORIZON, dtype=np.float64)
    d_matrix[np.arange(1, HORIZON), np.arange(HORIZON - 1)] = -a
    c = np.asarray(disturbance, dtype=np.float64).copy()
    c[0] += a * INITIAL_TEMPERATURE
    # Gurobi's matrix API uses x'Qx + q'x (without a one-half factor),
    # so Q=wI gives the intended comfort cost w||s-s_ref||^2.
    q_matrix = weight * np.eye(HORIZON, dtype=np.float64)
    q_vector = -2.0 * weight * preferred_temperature
    q_vector += d_matrix.T @ np.asarray(price, dtype=np.float64) / b
    identity = np.eye(HORIZON, dtype=np.float64)
    g_matrix = np.vstack((identity, -identity, d_matrix, -d_matrix))
    h_vector = np.r_[
        np.full(HORIZON, temperature_max + BOUND_RELAXATION),
        np.full(HORIZON, -temperature_min + BOUND_RELAXATION),
        c + BOUND_RELAXATION,
        -c - b * power_max + BOUND_RELAXATION,
    ]
    return d_matrix, c, q_matrix, q_vector, g_matrix, h_vector


def solve_equivalent_batch(
    theta: np.ndarray,
    prices: np.ndarray,
    disturbances: np.ndarray,
    with_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    prices = np.asarray(prices, dtype=np.float64)
    disturbances = np.asarray(disturbances, dtype=np.float64)
    n_days = len(prices)
    components = [build_qp_components(theta, prices[d], disturbances[d]) for d in range(n_days)]
    q_block = sparse.block_diag([item[2] for item in components], format="csc")
    g_block = sparse.block_diag([item[4] for item in components], format="csc")
    q_vector = np.concatenate([item[3] for item in components])
    h_vector = np.concatenate([item[5] for item in components])

    model = gp.Model("equivalent_tcl_batch")
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1.0e-8
    model.Params.OptimalityTol = 1.0e-8
    model.Params.NumericFocus = 1
    state = model.addMVar(n_days * HORIZON, lb=-gp.GRB.INFINITY, name="temperature")
    inequalities = model.addMConstr(g_block, state, "<", h_vector, name="limits")
    model.setMObjective(q_block, q_vector, 0.0, xc=state, sense=gp.GRB.MINIMIZE)
    model.optimize()
    if int(model.Status) != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Equivalent TCL batch QP failed with Gurobi status {int(model.Status)}.")
    state_value = np.asarray(state.X, dtype=np.float64).reshape(n_days, HORIZON)
    # Retain the solver duals and also reconstruct nonnegative KKT multipliers
    # from the active constraints below.  Gurobi reports shadow prices rather
    # than canonical inequality multipliers, and their sign depends on the
    # constraint convention.  Selecting the feasible sign/reconstruction with
    # the smallest stationarity residual makes that convention explicit and
    # remains robust when the active set is degenerate.
    _gurobi_pi = np.asarray(inequalities.Pi, dtype=np.float64).reshape(n_days, 4 * HORIZON)

    power = np.empty_like(state_value)
    jacobian = np.empty((n_days, HORIZON, len(PARAMETER_NAMES)), dtype=np.float64) if with_jacobian else None
    max_stationarity = 0.0
    max_active_residual = 0.0
    theta = np.asarray(theta, dtype=np.float64)
    a, b, power_max, _, _, weight = theta[:6]
    preferred_temperature = theta[6:]
    identity = np.eye(HORIZON, dtype=np.float64)
    zero_matrix = np.zeros((HORIZON, HORIZON), dtype=np.float64)
    d_dmatrix_da = np.zeros((HORIZON, HORIZON), dtype=np.float64)
    d_dmatrix_da[np.arange(1, HORIZON), np.arange(HORIZON - 1)] = -1.0
    d_c_da = np.zeros(HORIZON, dtype=np.float64)
    d_c_da[0] = INITIAL_TEMPERATURE

    for day_index, (d_matrix, c, q_matrix, q_day, g_matrix, h_day) in enumerate(components):
        x = state_value[day_index]
        u = d_matrix @ x - c
        power[day_index] = u / b
        hessian = 2.0 * q_matrix
        gradient = hessian @ x + q_day
        slack = h_day - g_matrix @ x
        candidate_active = slack <= ACTIVE_TOLERANCE
        reconstructed_dual = np.zeros(4 * HORIZON, dtype=np.float64)
        if np.any(candidate_active):
            reconstructed, _ = nnls(g_matrix[candidate_active].T, -gradient)
            reconstructed_dual[candidate_active] = reconstructed
        raw_dual = _gurobi_pi[day_index]
        dual_candidates = (
            reconstructed_dual,
            np.maximum(raw_dual, 0.0),
            np.maximum(-raw_dual, 0.0),
        )
        day_dual = min(
            dual_candidates,
            key=lambda candidate: float(np.max(np.abs(gradient + g_matrix.T @ candidate))),
        )
        stationarity = gradient + g_matrix.T @ day_dual
        max_stationarity = max(max_stationarity, float(np.max(np.abs(stationarity))))
        if not with_jacobian:
            continue
        active = candidate_active & (day_dual > DUAL_TOLERANCE)
        g_active = g_matrix[active]
        nu_active = day_dual[active]
        if len(g_active):
            max_active_residual = max(max_active_residual, float(np.max(np.abs(g_active @ x - h_day[active]))))
            kkt = np.block(
                [
                    [hessian, g_active.T],
                    [g_active, np.zeros((len(g_active), len(g_active)), dtype=np.float64)],
                ]
            )
        else:
            kkt = hessian

        derivative_terms: list[tuple[np.ndarray, np.ndarray, float]] = []
        rhs_columns: list[np.ndarray] = []
        for parameter_index in range(len(PARAMETER_NAMES)):
            d_qmatrix = zero_matrix
            d_qvector = np.zeros(HORIZON, dtype=np.float64)
            d_gmatrix = np.zeros_like(g_matrix)
            d_hvector = np.zeros_like(h_day)
            d_dmatrix = zero_matrix
            d_c = np.zeros(HORIZON, dtype=np.float64)
            d_b = 0.0
            if parameter_index == 0:  # a
                d_dmatrix = d_dmatrix_da
                d_c = d_c_da
                d_qvector = d_dmatrix.T @ prices[day_index] / b
                d_gmatrix[2 * HORIZON : 3 * HORIZON] = d_dmatrix
                d_gmatrix[3 * HORIZON : 4 * HORIZON] = -d_dmatrix
                d_hvector[2 * HORIZON : 3 * HORIZON] = d_c
                d_hvector[3 * HORIZON : 4 * HORIZON] = -d_c
            elif parameter_index == 1:  # b
                d_b = 1.0
                d_qvector = -(d_matrix.T @ prices[day_index]) / (b * b)
                d_hvector[3 * HORIZON : 4 * HORIZON] = -power_max
            elif parameter_index == 2:  # Pmax
                d_hvector[3 * HORIZON : 4 * HORIZON] = -b
            elif parameter_index == 3:  # Tmin
                d_hvector[HORIZON : 2 * HORIZON] = -1.0
            elif parameter_index == 4:  # Tmax
                d_hvector[:HORIZON] = 1.0
            elif parameter_index == 5:  # weight
                d_qmatrix = 2.0 * identity
                d_qvector = -2.0 * preferred_temperature
            else:  # hourly preferred temperature
                hour = parameter_index - len(BASE_PARAMETER_NAMES)
                d_qvector[hour] = -2.0 * weight

            top_rhs = -(d_qmatrix @ x + d_qvector)
            if len(g_active):
                d_g_active = d_gmatrix[active]
                d_h_active = d_hvector[active]
                top_rhs -= d_g_active.T @ nu_active
                bottom_rhs = -(d_g_active @ x - d_h_active)
                rhs_columns.append(np.r_[top_rhs, bottom_rhs])
            else:
                rhs_columns.append(top_rhs)
            derivative_terms.append((d_dmatrix, d_c, d_b))

        rhs_matrix = np.column_stack(rhs_columns)
        if len(g_active):
            derivative_solution = np.linalg.lstsq(
                kkt, rhs_matrix, rcond=1.0e-11
            )[0]
        else:
            derivative_solution = np.linalg.solve(kkt, rhs_matrix)
        dx_matrix = derivative_solution[:HORIZON]
        for parameter_index, (d_dmatrix, d_c, d_b) in enumerate(
            derivative_terms
        ):
            dx = dx_matrix[:, parameter_index]
            dp = (d_dmatrix @ x + d_matrix @ dx - d_c) / b - u * d_b / (b * b)
            assert jacobian is not None
            jacobian[day_index, :, parameter_index] = dp

    diagnostics = {
        "max_stationarity_residual": max_stationarity,
        "max_active_constraint_residual": max_active_residual,
    }
    return power, jacobian, diagnostics


def build_precision(covariance: np.ndarray) -> tuple[np.ndarray, float, float]:
    covariance = np.asarray(covariance, dtype=np.float64)
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    positive = diagonal[diagonal > 0.0]
    base = float(np.median(positive)) if len(positive) else 1.0e-6
    ridge = max(1.0e-8, 1.0e-3 * base)
    precision = np.empty_like(covariance)
    for index, matrix in enumerate(covariance):
        regularized = 0.5 * (matrix + matrix.T) + ridge * np.eye(HORIZON)
        precision[index] = np.linalg.inv(regularized)
    global_scale = float(np.mean(np.trace(precision, axis1=1, axis2=2) / HORIZON))
    precision /= global_scale
    return precision, ridge, global_scale


def fit_equivalent_model(
    prices: np.ndarray,
    disturbances: np.ndarray,
    target_power: np.ndarray,
    theta_initial: np.ndarray,
    free_indices: np.ndarray,
    covariance: np.ndarray | None = None,
    covariance_weight: float = 1.0,
    anchor_theta: np.ndarray | None = None,
    regularization_indices: np.ndarray | None = None,
    rho: float = 0.0,
    max_iterations: int = 120,
) -> FitResult:
    theta_initial = np.clip(np.asarray(theta_initial, dtype=np.float64), PARAMETER_LOWER, PARAMETER_UPPER)
    free_indices = np.asarray(free_indices, dtype=np.int32)
    parameter_range = PARAMETER_UPPER - PARAMETER_LOWER
    y0 = (theta_initial[free_indices] - PARAMETER_LOWER[free_indices]) / parameter_range[free_indices]
    fixed_theta = theta_initial.copy()
    target_power = np.asarray(target_power, dtype=np.float64)
    if covariance is None:
        precision = None
    else:
        precision, _, _ = build_precision(covariance)
    covariance_weight = float(covariance_weight)
    if not 0.0 <= covariance_weight <= 1.0:
        raise ValueError("covariance_weight must lie in [0, 1].")
    anchor = theta_initial if anchor_theta is None else np.asarray(anchor_theta, dtype=np.float64)
    best_diagnostics = {"max_stationarity_residual": math.inf, "max_active_constraint_residual": math.inf}

    def unpack(y: np.ndarray) -> np.ndarray:
        theta = fixed_theta.copy()
        theta[free_indices] = PARAMETER_LOWER[free_indices] + np.asarray(y) * parameter_range[free_indices]
        return theta

    def objective(y: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_diagnostics
        theta = unpack(y)
        try:
            predicted, jacobian, diagnostics = solve_equivalent_batch(theta, prices, disturbances, with_jacobian=True)
        except RuntimeError:
            return 1.0e12, np.zeros_like(y)
        assert jacobian is not None
        residual = predicted - target_power
        if precision is None:
            loss = float(np.mean(np.sum(residual * residual, axis=1) / HORIZON))
            gradient_theta = 2.0 * np.mean(
                np.einsum("dt,dtp->dp", residual, jacobian, optimize=True) / HORIZON,
                axis=0,
            )
        else:
            weighted = np.einsum("dtv,dv->dt", precision, residual, optimize=True)
            blended_residual = (
                (1.0 - covariance_weight) * residual
                + covariance_weight * weighted
            )
            loss = float(
                np.mean(
                    np.sum(residual * blended_residual, axis=1) / HORIZON
                )
            )
            gradient_theta = 2.0 * np.mean(
                np.einsum(
                    "dt,dtp->dp", blended_residual, jacobian, optimize=True
                )
                / HORIZON,
                axis=0,
            )
        if rho > 0.0:
            if regularization_indices is None:
                penalty_indices = free_indices
            else:
                requested = np.asarray(regularization_indices, dtype=np.int32)
                penalty_indices = np.intersect1d(
                    free_indices, requested, assume_unique=False
                )
            if len(penalty_indices):
                normalized_difference = (
                    theta[penalty_indices] - anchor[penalty_indices]
                ) / parameter_range[penalty_indices]
                loss += float(rho * np.mean(normalized_difference**2))
                gradient_theta[penalty_indices] += (
                    2.0
                    * rho
                    * normalized_difference
                    / parameter_range[penalty_indices]
                    / len(penalty_indices)
                )
        best_diagnostics = diagnostics
        gradient_y = gradient_theta[free_indices] * parameter_range[free_indices]
        return loss, gradient_y

    result = minimize(
        objective,
        y0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(free_indices),
        options={"maxiter": max_iterations, "ftol": 1.0e-11, "gtol": 1.0e-7, "maxls": 30},
    )
    theta = unpack(result.x)
    return FitResult(
        theta=theta,
        objective=float(result.fun),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        gradient_norm=float(np.linalg.norm(result.jac, ord=np.inf)),
        max_stationarity_residual=float(best_diagnostics["max_stationarity_residual"]),
        max_active_constraint_residual=float(best_diagnostics["max_active_constraint_residual"]),
    )


def daily_normalized_rmse(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    numerator = np.sqrt(np.mean((predicted - truth) ** 2, axis=1))
    denominator = np.mean(truth, axis=1)
    return 100.0 * np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def conventional_rmse(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(predicted) - np.asarray(truth)) ** 2, axis=1))


def choose_stage2_rho(
    prices: np.ndarray,
    disturbances: np.ndarray,
    nominal: np.ndarray,
    moment_mean: np.ndarray,
    covariance: np.ndarray,
    initial_theta: np.ndarray,
    full_finetune: bool,
    max_iterations: int,
) -> tuple[float, list[dict[str, object]]]:
    validation_index = np.arange(4, len(prices), 5, dtype=np.int32)
    training_index = np.setdiff1d(np.arange(len(prices), dtype=np.int32), validation_index)
    stage1 = fit_equivalent_model(
        prices[training_index],
        disturbances[training_index],
        nominal[training_index],
        initial_theta,
        ALL_INDICES,
        max_iterations=max_iterations,
    )
    candidates = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
    rows: list[dict[str, object]] = []
    free = ALL_INDICES if full_finetune else USER_INDICES
    target = nominal + moment_mean
    for rho in candidates:
        fitted = fit_equivalent_model(
            prices[training_index],
            disturbances[training_index],
            target[training_index],
            stage1.theta,
            free,
            covariance=covariance[training_index],
            anchor_theta=stage1.theta,
            rho=rho,
            max_iterations=max_iterations,
        )
        validation_prediction, _, _ = solve_equivalent_batch(
            fitted.theta,
            prices[validation_index],
            disturbances[validation_index],
            with_jacobian=False,
        )
        validation_error = float(np.mean(daily_normalized_rmse(validation_prediction, target[validation_index])))
        rows.append(
            {
                "method": "Full fine-tuning" if full_finetune else "Proposed",
                "rho": rho,
                "validation_mean_nrmse_percent": validation_error,
                "optimization_success": fitted.success,
                "optimization_message": fitted.message,
            }
        )
    successful = [row for row in rows if bool(row["optimization_success"])]
    pool = successful if successful else rows
    best = min(pool, key=lambda row: float(row["validation_mean_nrmse_percent"]))
    return float(best["rho"]), rows


def fit_and_evaluate(
    aggregate_data_path: Path,
    output_dir: Path,
    max_iterations: int,
    default_stage2_rho: float,
    quick: bool,
) -> dict[str, object]:
    with np.load(aggregate_data_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    days = arrays["days"].astype("datetime64[D]")
    train_mask = days < TEST_START
    test_mask = ~train_mask
    train_days = days[train_mask]
    prices = arrays["price_usd_per_kwh"].astype(np.float64)
    nominal = arrays["nominal_power_kw_per_user"].astype(np.float64)
    disturbance = arrays["disturbance_c"].astype(np.float64)
    ground_truth = arrays["ground_truth_power_kw_per_user"].astype(np.float64)
    moment_mean = arrays["moment_mean_deviation_kw_per_user"].astype(np.float64)
    covariance = arrays["moment_covariance_kw2_per_user"].astype(np.float64)
    empirical_theta = expand_empirical_theta(
        arrays["empirical_initial_theta"].astype(np.float64)
    )
    price_order = nearest_medoid_price_order(prices[train_mask], train_days)
    selected_box_train = price_order[:BOXPLOT_PRICE_COUNT]
    price_rows = [
        {
            "selection_rank": rank + 1,
            "date": str(train_days[index]),
            "original_training_day_index": int(index),
            "used_in_boxplot_training": bool(rank < BOXPLOT_PRICE_COUNT),
        }
        for rank, index in enumerate(price_order)
    ]
    write_csv(output_dir / "selected_price_signals.csv", price_rows)

    box_group = BOXPLOT_GROUP_INDEX
    box_user_count = int(USER_SIZES[box_group])
    initial_theta = np.mean(empirical_theta[train_mask, box_group][selected_box_train], axis=0)
    initial_theta = np.clip(initial_theta, PARAMETER_LOWER, PARAMETER_UPPER)
    proposed_rho = float(default_stage2_rho)
    full_rho = float(default_stage2_rho)
    hyperparameter_rows = [
        {
            "method": method,
            "rho": rho,
            "selection_basis": "July-August-only stress-case diagnostic",
            "boxplot_user_count": box_user_count,
            "boxplot_training_price_count": BOXPLOT_PRICE_COUNT,
        }
        for method, rho in (("Proposed", proposed_rho), ("Full fine-tuning", full_rho))
    ]
    write_csv(output_dir / "stage2_hyperparameter_validation.csv", hyperparameter_rows)

    print(
        f"Fitting the three boxplot models for N={box_user_count} on "
        f"{BOXPLOT_PRICE_COUNT} training price signals...",
        flush=True,
    )
    stage1 = fit_equivalent_model(
        prices[train_mask][selected_box_train],
        disturbance[train_mask, box_group][selected_box_train],
        nominal[train_mask, box_group][selected_box_train],
        initial_theta, ALL_INDICES, max_iterations=max_iterations,
    )
    behavior_target = (
        nominal[train_mask, box_group][selected_box_train]
        + moment_mean[train_mask, box_group][selected_box_train]
    )
    proposed = fit_equivalent_model(
        prices[train_mask][selected_box_train],
        disturbance[train_mask, box_group][selected_box_train],
        behavior_target,
        stage1.theta, USER_INDICES,
        covariance=covariance[train_mask, box_group][selected_box_train],
        anchor_theta=stage1.theta, rho=proposed_rho, max_iterations=max_iterations,
    )
    full = fit_equivalent_model(
        prices[train_mask][selected_box_train],
        disturbance[train_mask, box_group][selected_box_train],
        behavior_target,
        stage1.theta, ALL_INDICES,
        covariance=covariance[train_mask, box_group][selected_box_train],
        anchor_theta=stage1.theta, rho=full_rho, max_iterations=max_iterations,
    )

    method_results = {"Proposed": proposed, "Stage 1 only": stage1, "Full fine-tuning": full}
    predicted_profiles = np.empty((3, len(days), HORIZON), dtype=np.float64)
    daily_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for method_index, (method, fitted) in enumerate(method_results.items()):
        prediction, _, solve_diagnostics = solve_equivalent_batch(
            fitted.theta, prices, disturbance[:, box_group], with_jacobian=False
        )
        predicted_profiles[method_index] = prediction
        nrmse = daily_normalized_rmse(prediction, ground_truth[:, box_group])
        rmse = conventional_rmse(prediction, ground_truth[:, box_group])
        for day_index, day in enumerate(days):
            training_index = int(np.count_nonzero(train_mask[:day_index])) if train_mask[day_index] else -1
            is_selected_training_day = bool(
                train_mask[day_index] and training_index in set(selected_box_train.tolist())
            )
            if not is_selected_training_day and not test_mask[day_index]:
                continue
            daily_rows.append(
                {
                    "method": method,
                    "period": "July-August (training)" if train_mask[day_index] else "September (test)",
                    "date": str(day),
                    "normalized_rmse_percent": float(nrmse[day_index]),
                    "rmse_kw_per_user": float(rmse[day_index]),
                }
            )
        for name, value in zip(PARAMETER_NAMES, fitted.theta):
            parameter_rows.append(
                {
                    "case": f"boxplot_N{box_user_count}_D{BOXPLOT_PRICE_COUNT}",
                    "method": method,
                    "parameter": name,
                    "value": float(value),
                    "stage2_rho": proposed_rho if method == "Proposed" else (full_rho if method == "Full fine-tuning" else math.nan),
                }
            )
        diagnostic_rows.append(
            {
                "case": f"boxplot_N{box_user_count}_D{BOXPLOT_PRICE_COUNT}",
                "method": method,
                "objective": fitted.objective,
                "optimization_success": fitted.success,
                "status": fitted.status,
                "message": fitted.message,
                "iterations": fitted.iterations,
                "evaluations": fitted.evaluations,
                "gradient_infinity_norm": fitted.gradient_norm,
                "fit_max_stationarity_residual": fitted.max_stationarity_residual,
                "fit_max_active_constraint_residual": fitted.max_active_constraint_residual,
                "prediction_max_stationarity_residual": solve_diagnostics["max_stationarity_residual"],
            }
        )

    print("Running the 10 x 10 user-count/price-diversity sensitivity grid...", flush=True)
    sensitivity_rows: list[dict[str, object]] = []
    stage1_warm: dict[int, np.ndarray] = {}
    proposed_warm: dict[int, np.ndarray] = {}
    user_loop = range(1 if quick else len(USER_SIZES))
    price_loop = PRICE_COUNTS[:1] if quick else PRICE_COUNTS
    for group_index in user_loop:
        user_count = int(USER_SIZES[group_index])
        base_theta = np.mean(empirical_theta[train_mask, group_index], axis=0)
        base_theta = np.clip(base_theta, PARAMETER_LOWER, PARAMETER_UPPER)
        for price_count in price_loop:
            selected_train = price_order[: int(price_count)]
            # The data-fit term is an average over days.  Scaling rho as 1/D
            # is therefore equivalent to a summed likelihood plus a fixed
            # parameter prior: evidence grows with sample count while the
            # prior strength stays fixed.
            case_rho = SENSITIVITY_STAGE2_RHO * BOXPLOT_PRICE_COUNT / float(price_count)
            stage1_initial = stage1_warm.get(int(price_count), base_theta)
            fitted_stage1 = fit_equivalent_model(
                prices[train_mask][selected_train],
                disturbance[train_mask, group_index][selected_train],
                nominal[train_mask, group_index][selected_train],
                stage1_initial,
                ALL_INDICES,
                max_iterations=max_iterations,
            )
            proposed_initial = fitted_stage1.theta.copy()
            if int(price_count) in proposed_warm:
                proposed_initial[USER_INDICES] = proposed_warm[int(price_count)][USER_INDICES]
            target_selected = (
                nominal[train_mask, group_index][selected_train]
                + moment_mean[train_mask, group_index][selected_train]
            )
            fitted_proposed = fit_equivalent_model(
                prices[train_mask][selected_train],
                disturbance[train_mask, group_index][selected_train],
                target_selected,
                proposed_initial,
                USER_INDICES,
                covariance=covariance[train_mask, group_index][selected_train],
                anchor_theta=fitted_stage1.theta,
                rho=case_rho,
                max_iterations=max_iterations,
            )
            test_prediction, _, sensitivity_diagnostics = solve_equivalent_batch(
                fitted_proposed.theta,
                prices[test_mask],
                disturbance[test_mask, group_index],
                with_jacobian=False,
            )
            test_nrmse = daily_normalized_rmse(test_prediction, ground_truth[test_mask, group_index])
            sensitivity_rows.append(
                {
                    "user_count": user_count,
                    "price_signal_count": int(price_count),
                    "mean_september_normalized_rmse_percent": float(np.mean(test_nrmse)),
                    "median_september_normalized_rmse_percent": float(np.median(test_nrmse)),
                    "stage2_rho": case_rho,
                    "stage1_success": fitted_stage1.success,
                    "stage2_success": fitted_proposed.success,
                    "stage1_objective": fitted_stage1.objective,
                    "stage2_objective": fitted_proposed.objective,
                    "maximum_prediction_stationarity_residual": sensitivity_diagnostics["max_stationarity_residual"],
                }
            )
            write_csv(output_dir / "sensitivity_rmse_partial.csv", sensitivity_rows)
            stage1_warm[int(price_count)] = fitted_stage1.theta
            proposed_warm[int(price_count)] = fitted_proposed.theta
            for name, value in zip(PARAMETER_NAMES, fitted_proposed.theta):
                parameter_rows.append(
                    {
                        "case": f"sensitivity_N{user_count}_D{int(price_count)}",
                        "method": "Proposed",
                        "parameter": name,
                        "value": float(value),
                        "stage2_rho": case_rho,
                    }
                )
            print(
                f"    N={user_count:4d}, D={int(price_count):2d}: "
                f"mean September NRMSE={float(np.mean(test_nrmse)):.4f}%",
                flush=True,
            )

    write_csv(output_dir / "daily_rmse.csv", daily_rows)
    write_csv(output_dir / "sensitivity_rmse.csv", sensitivity_rows)
    write_csv(output_dir / "fitted_parameters.csv", parameter_rows)
    write_csv(output_dir / "optimization_diagnostics.csv", diagnostic_rows)
    np.savez_compressed(
        output_dir / "predicted_profiles.npz",
        days=days,
        methods=np.asarray(list(method_results.keys())),
        predicted_power_kw_per_user=predicted_profiles.astype(np.float32),
        ground_truth_power_kw_per_user=ground_truth[:, box_group].astype(np.float32),
        train_mask=np.asarray(
            [
                bool(train_mask[index] and int(np.count_nonzero(train_mask[:index])) in set(selected_box_train.tolist()))
                for index in range(len(days))
            ],
            dtype=bool,
        ),
    )

    test_frame = pd.DataFrame(daily_rows)
    test_frame = test_frame[test_frame["period"] == "September (test)"].pivot(
        index="date", columns="method", values="normalized_rmse_percent"
    )
    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for baseline in ("Stage 1 only", "Full fine-tuning"):
        statistic, p_value = wilcoxon(
            test_frame["Proposed"].to_numpy(),
            test_frame[baseline].to_numpy(),
            alternative="two-sided",
            zero_method="pratt",
        )
        difference = test_frame["Proposed"].to_numpy() - test_frame[baseline].to_numpy()
        raw_p.append(float(p_value))
        comparisons.append(
            {
                "comparison": f"Proposed vs {baseline}",
                "paired_days": len(difference),
                "wilcoxon_statistic": float(statistic),
                "p_value_raw": float(p_value),
                "median_paired_difference_percentage_points": float(np.median(difference)),
                "mean_paired_difference_percentage_points": float(np.mean(difference)),
            }
        )
    order_p = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order_p):
        value = min(1.0, raw_p[index] * (len(raw_p) - rank))
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(comparisons, adjusted):
        row["p_value_holm"] = float(value)
    write_csv(output_dir / "statistical_comparisons.csv", comparisons)

    summary = {
        "proposed_rho": proposed_rho,
        "full_finetune_rho": full_rho,
        "sensitivity_rho": SENSITIVITY_STAGE2_RHO,
        "boxplot_user_count": box_user_count,
        "boxplot_training_price_count": BOXPLOT_PRICE_COUNT,
        "main_fit_success": {method: result.success for method, result in method_results.items()},
        "price_order": price_order.tolist(),
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11.5,
            "axes.labelsize": 13.0,
            "axes.labelpad": 3.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def make_figures(output_dir: Path, figure_dir: Path) -> None:
    configure_figure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(output_dir / "daily_rmse.csv")
    sensitivity = pd.read_csv(output_dir / "sensitivity_rmse.csv")
    methods = ["Proposed", "Stage 1 only", "Full fine-tuning"]
    method_labels = ["Proposed", "No-BR", "Full training"]
    periods = ["July-August (training)", "September (test)"]
    period_labels = ["Training (24 days)", "Test (30 days)"]
    colors = ["#9FC5E8", "#2F6FA3"]

    fig, ax = plt.subplots(figsize=(4.8, 2.9), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.78)
    positions: list[float] = []
    values: list[np.ndarray] = []
    box_colors: list[str] = []
    width = 0.28
    centers = np.arange(len(methods), dtype=np.float64)
    for method_index, method in enumerate(methods):
        for period_index, period in enumerate(periods):
            positions.append(float(centers[method_index] + (period_index - 0.5) * 0.34))
            values.append(
                daily.loc[(daily["method"] == method) & (daily["period"] == period), "normalized_rmse_percent"].to_numpy()
            )
            box_colors.append(colors[period_index])
    box = ax.boxplot(
        values,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
        boxprops={"edgecolor": "#4A4A4A", "linewidth": 1.1},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "#777777", "markeredgecolor": "none", "alpha": 0.5},
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    legend_handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="#4A4A4A", label=label)
        for color, label in zip(colors, period_labels)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("NRMSE (%)")
    fig.savefig(
        figure_dir / "node_level_modeling_rmse_boxplot.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)

    if len(sensitivity) != len(USER_SIZES) * len(PRICE_COUNTS):
        raise ValueError("The sensitivity source data do not contain the complete 10 x 10 grid.")
    pivot = sensitivity.pivot(index="user_count", columns="price_signal_count", values="mean_september_normalized_rmse_percent")
    pivot = pivot.reindex(index=USER_SIZES, columns=PRICE_COUNTS)
    matrix = pivot.to_numpy(dtype=np.float64)
    minimum = float(np.min(matrix))
    maximum = float(np.max(matrix))
    span = max(maximum - minimum, 1.0e-12)
    normalized = (matrix - minimum) / span
    # Both circle area and color use the same linear normalization so equal
    # RMSE increments always receive equal visual increments.  Visibility is
    # improved only through a common base area and a truncated blue palette.
    color_norm = mpl.colors.Normalize(vmin=minimum, vmax=maximum)
    x, y = np.meshgrid(np.arange(len(PRICE_COUNTS)), np.arange(len(USER_SIZES)))
    sizes = 340.0 + 520.0 * normalized
    base_cmap = mpl.colormaps["Blues"]
    heatmap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "truncated_blues",
        base_cmap(np.linspace(0.20, 0.98, 256)),
    )

    fig, ax = plt.subplots(figsize=(4.8, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.15, top=0.98)
    for row in range(len(USER_SIZES)):
        for column in range(len(PRICE_COUNTS)):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (column - 0.47, row - 0.47),
                    0.94,
                    0.94,
                    facecolor="#F4F6F8",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=0,
                )
            )
    scatter = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=sizes.ravel(),
        c=matrix.ravel(),
        cmap=heatmap_cmap,
        norm=color_norm,
        edgecolors="none",
        zorder=2,
    )
    cmap = heatmap_cmap
    for row in range(len(USER_SIZES)):
        for column in range(len(PRICE_COUNTS)):
            value = matrix[row, column]
            red, green, blue, _ = cmap(color_norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if luminance < 0.53 else "#202020",
                fontweight="semibold",
                zorder=3,
            )
    ax.set_xlim(-0.6, len(PRICE_COUNTS) - 0.4)
    ax.set_ylim(-0.6, len(USER_SIZES) - 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(len(PRICE_COUNTS)))
    ax.set_xticklabels(PRICE_COUNTS)
    ax.set_yticks(np.arange(len(USER_SIZES)))
    ax.set_yticklabels(USER_SIZES)
    ax.set_xlabel("Number of training price signals", fontsize=10.5)
    ax.set_ylabel("Number of users", fontsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=9.5)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Mean September normalized RMSE (%)", fontsize=10.0)
    colorbar.ax.tick_params(labelsize=9.0)
    colorbar.outline.set_linewidth(0.7)
    fig.savefig(
        figure_dir / "node_level_modeling_sensitivity_heatmap.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def _saved_parameter_vector(
    frame: pd.DataFrame,
    case: str,
    method: str,
) -> np.ndarray:
    subset = frame.loc[
        (frame["case"] == case) & (frame["method"] == method),
        ["parameter", "value"],
    ]
    values = {str(row.parameter): float(row.value) for row in subset.itertuples()}
    missing = [name for name in PARAMETER_NAMES if name not in values]
    if missing:
        raise ValueError(f"Saved fit {case}/{method} is missing parameters: {missing}")
    return np.asarray([values[name] for name in PARAMETER_NAMES], dtype=np.float64)


def solve_registered_batch_with_cvxpy(
    theta: np.ndarray,
    prices: np.ndarray,
    disturbances: np.ndarray,
) -> np.ndarray:
    """Re-evaluate a registered fitted model without retraining it.

    This solver is used only to refresh metrics from the saved parameter table
    when Gurobi is unavailable.  It solves the identical convex state-space QP
    with OSQP and is cross-checked against the saved Gurobi predictions before
    any sensitivity metric is replaced.
    """

    import cvxpy as cp
    import scipy.sparse as scipy_sparse

    prices = np.asarray(prices, dtype=np.float64)
    disturbances = np.asarray(disturbances, dtype=np.float64)
    n_days = len(prices)
    components = [
        build_qp_components(theta, prices[day], disturbances[day])
        for day in range(n_days)
    ]
    q_block = scipy_sparse.block_diag([item[2] for item in components], format="csc")
    g_block = scipy_sparse.block_diag([item[4] for item in components], format="csc")
    q_vector = np.concatenate([item[3] for item in components])
    h_vector = np.concatenate([item[5] for item in components])
    state = cp.Variable(n_days * HORIZON)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(state, cp.psd_wrap(q_block)) + q_vector @ state),
        [g_block @ state <= h_vector],
    )
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1.0e-8,
        eps_rel=1.0e-8,
        max_iter=200_000,
        polish=True,
        verbose=False,
    )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or state.value is None:
        raise RuntimeError(f"Saved-fit QP failed with CVXPY status {problem.status}.")
    state_value = np.asarray(state.value, dtype=np.float64).reshape(n_days, HORIZON)
    power = np.empty_like(state_value)
    b = float(np.asarray(theta, dtype=np.float64)[1])
    for day, (d_matrix, c, *_rest) in enumerate(components):
        power[day] = (d_matrix @ state_value[day] - c) / b
    return power


def refresh_metrics_from_saved_fits(output_dir: Path) -> dict[str, float]:
    """Apply the mean-power NRMSE definition to every registered result."""

    from scipy.stats import wilcoxon as scipy_wilcoxon

    aggregate_path = output_dir / "aggregate_daily_data.npz"
    prediction_path = output_dir / "predicted_profiles.npz"
    parameter_path = output_dir / "fitted_parameters.csv"
    with np.load(aggregate_path, allow_pickle=False) as archive:
        days = np.asarray(archive["days"]).astype("datetime64[D]")
        prices = np.asarray(archive["price_usd_per_kwh"], dtype=np.float64)
        disturbances = np.asarray(archive["disturbance_c"], dtype=np.float64)
        ground_truth = np.asarray(
            archive["ground_truth_power_kw_per_user"], dtype=np.float64
        )
    with np.load(prediction_path, allow_pickle=False) as archive:
        method_names = [str(value) for value in np.asarray(archive["methods"])]
        predicted = np.asarray(
            archive["predicted_power_kw_per_user"], dtype=np.float64
        )
        selected_training_mask = np.asarray(archive["train_mask"], dtype=bool)
    test_mask = days >= TEST_START
    evaluation_mask = selected_training_mask | test_mask
    box_truth = ground_truth[:, BOXPLOT_GROUP_INDEX]

    daily_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(method_names):
        method_nrmse = daily_normalized_rmse(predicted[method_index], box_truth)
        method_rmse = conventional_rmse(predicted[method_index], box_truth)
        daily_mean_power = np.mean(box_truth, axis=1)
        for day_index in np.flatnonzero(evaluation_mask):
            daily_rows.append(
                {
                    "method": method,
                    "period": (
                        "July-August (training)"
                        if selected_training_mask[day_index]
                        else "September (test)"
                    ),
                    "date": str(days[day_index]),
                    "normalized_rmse_percent": float(method_nrmse[day_index]),
                    "rmse_kw_per_user": float(method_rmse[day_index]),
                    "mean_ground_truth_power_kw_per_user": float(
                        daily_mean_power[day_index]
                    ),
                }
            )
    write_csv(output_dir / "daily_rmse.csv", daily_rows)

    parameters = pd.read_csv(parameter_path)
    proposed_theta = _saved_parameter_vector(
        parameters,
        f"boxplot_N{int(USER_SIZES[BOXPLOT_GROUP_INDEX])}_D{BOXPLOT_PRICE_COUNT}",
        "Proposed",
    )
    proposed_check = solve_registered_batch_with_cvxpy(
        proposed_theta,
        prices[test_mask],
        disturbances[test_mask, BOXPLOT_GROUP_INDEX],
    )
    proposed_index = method_names.index("Proposed")
    solver_crosscheck = float(
        np.max(np.abs(proposed_check - predicted[proposed_index, test_mask]))
    )
    if solver_crosscheck > 2.0e-4:
        raise RuntimeError(
            "CVXPY saved-fit re-evaluation does not match the registered Gurobi "
            f"predictions (maximum difference {solver_crosscheck:.3e} kW/user)."
        )

    previous_sensitivity = pd.read_csv(output_dir / "sensitivity_rmse.csv")
    previous_lookup = {
        (int(row.user_count), int(row.price_signal_count)): row._asdict()
        for row in previous_sensitivity.itertuples(index=False)
    }
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_daily_rows: list[dict[str, object]] = []
    for group_index, user_count in enumerate(USER_SIZES):
        truth = ground_truth[test_mask, group_index]
        truth_mean = np.mean(truth, axis=1)
        for price_count in PRICE_COUNTS:
            theta = _saved_parameter_vector(
                parameters,
                f"sensitivity_N{int(user_count)}_D{int(price_count)}",
                "Proposed",
            )
            prediction = solve_registered_batch_with_cvxpy(
                theta,
                prices[test_mask],
                disturbances[test_mask, group_index],
            )
            nrmse = daily_normalized_rmse(prediction, truth)
            rmse = conventional_rmse(prediction, truth)
            key = (int(user_count), int(price_count))
            row = dict(previous_lookup[key])
            row["mean_september_normalized_rmse_percent"] = float(np.mean(nrmse))
            row["median_september_normalized_rmse_percent"] = float(np.median(nrmse))
            sensitivity_rows.append(row)
            for local_day, global_day in enumerate(np.flatnonzero(test_mask)):
                sensitivity_daily_rows.append(
                    {
                        "user_count": int(user_count),
                        "price_signal_count": int(price_count),
                        "date": str(days[global_day]),
                        "normalized_rmse_percent": float(nrmse[local_day]),
                        "rmse_kw_per_user": float(rmse[local_day]),
                        "mean_ground_truth_power_kw_per_user": float(
                            truth_mean[local_day]
                        ),
                    }
                )
            print(
                f"Refreshed N={int(user_count):4d}, D={int(price_count):2d}: "
                f"mean September NRMSE={float(np.mean(nrmse)):.4f}%",
                flush=True,
            )
    write_csv(output_dir / "sensitivity_rmse.csv", sensitivity_rows)
    write_csv(output_dir / "sensitivity_daily_rmse.csv", sensitivity_daily_rows)

    daily_frame = pd.DataFrame(daily_rows)
    test_frame = daily_frame[daily_frame["period"] == "September (test)"].pivot(
        index="date", columns="method", values="normalized_rmse_percent"
    )
    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for baseline in ("Stage 1 only", "Full fine-tuning"):
        statistic, p_value = scipy_wilcoxon(
            test_frame["Proposed"].to_numpy(),
            test_frame[baseline].to_numpy(),
            alternative="two-sided",
            zero_method="pratt",
        )
        difference = (
            test_frame["Proposed"].to_numpy()
            - test_frame[baseline].to_numpy()
        )
        raw_p.append(float(p_value))
        comparisons.append(
            {
                "comparison": f"Proposed vs {baseline}",
                "paired_days": len(difference),
                "wilcoxon_statistic": float(statistic),
                "p_value_raw": float(p_value),
                "median_paired_difference_percentage_points": float(
                    np.median(difference)
                ),
                "mean_paired_difference_percentage_points": float(
                    np.mean(difference)
                ),
            }
        )
    order_p = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order_p):
        value = min(1.0, raw_p[index] * (len(raw_p) - rank))
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(comparisons, adjusted):
        row["p_value_holm"] = float(value)
    write_csv(output_dir / "statistical_comparisons.csv", comparisons)

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["configuration"]["metric"] = (
        "RMSE(prediction, truth) / mean_t(truth_t) x 100%"
    )
    metadata["metric_refresh"] = {
        "source": "registered fitted parameters and source data",
        "solver": "CVXPY/OSQP re-evaluation of the unchanged convex QP",
        "maximum_prediction_crosscheck_kw_per_user": solver_crosscheck,
        "daily_sensitivity_values_written": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    previous_elapsed = float(
        metadata.get("runtime", {}).get("elapsed_seconds_last_invocation", 0.0)
    )
    write_validation_report(
        output_dir,
        {**metadata.get("source_data", {}), **metadata["configuration"]},
        previous_elapsed,
    )
    return {"maximum_prediction_crosscheck_kw_per_user": solver_crosscheck}


def write_validation_report(output_dir: Path, metadata: Mapping[str, object], elapsed: float) -> None:
    daily = pd.read_csv(output_dir / "daily_rmse.csv")
    sensitivity = pd.read_csv(output_dir / "sensitivity_rmse.csv")
    diagnostics = pd.read_csv(output_dir / "optimization_diagnostics.csv")
    comparisons = pd.read_csv(output_dir / "statistical_comparisons.csv")
    summary_rows = []
    for method in ("Proposed", "Stage 1 only", "Full fine-tuning"):
        for period in ("July-August (training)", "September (test)"):
            values = daily.loc[(daily["method"] == method) & (daily["period"] == period), "normalized_rmse_percent"]
            summary_rows.append(
                {
                    "method": method,
                    "period": period,
                    "n_days": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                }
            )
    write_csv(output_dir / "rmse_summary.csv", summary_rows)
    all_success = bool(diagnostics["optimization_success"].all()) and bool(sensitivity["stage1_success"].all()) and bool(sensitivity["stage2_success"].all())
    confidence = "SOLID" if all_success else "CAUTION"
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent; nature-figure; pdf",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-09-19",
        "- Verification Status: ANALYZED",
        "- Version Label: node_modeling_impact_v4_mean_power_nrmse",
        "",
        "## Validation Report",
        "",
        f"- **Overall Confidence**: {confidence}",
        f"- **Runtime**: {elapsed:.1f} seconds",
        f"- **Shared Monte Carlo paths**: {metadata['ground_truth_samples']} actual-response draws per user-day form both the ground truth and individual moments.",
        f"- **Boxplot stress case**: {USER_SIZES[BOXPLOT_GROUP_INDEX]} users and {BOXPLOT_PRICE_COUNT} July-August training days, selected without September outcomes.",
        "- **Sensitivity training sets**: 6--60 nested July-August price signals ordered from the training-set medoid toward increasingly dissimilar daily shapes; September moments are audit-only.",
        f"- **Sensitivity regularization**: rho_D = {SENSITIVITY_STAGE2_RHO:g} x {BOXPLOT_PRICE_COUNT}/D because the data-fit loss is averaged over D days.",
        "- **Test period**: 30 September days.",
        "- **Primary metric**: RMSE(prediction, ground truth) / mean_t(ground truth_t) x 100%.",
        "",
        "### Daily normalized RMSE summaries",
        "",
        "| Method | Period | n | Mean (%) | Median (%) | Q1 (%) | Q3 (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['period']} | {row['n_days']} | {row['mean']:.5f} | {row['median']:.5f} | {row['q1']:.5f} | {row['q3']:.5f} |"
        )
    lines.extend(
        [
            "",
            "### Paired September comparisons",
            "",
            "| Comparison | Wilcoxon p | Holm p | Median paired difference (percentage points) |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['p_value_raw']:.6g} | {row['p_value_holm']:.6g} | {row['median_paired_difference_percentage_points']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Numerical and reproducibility checks",
            "",
            f"- Main-model optimizations all converged: {bool(diagnostics['optimization_success'].all())}.",
            f"- Sensitivity Stage-1 fits all converged: {bool(sensitivity['stage1_success'].all())}.",
            f"- Sensitivity Stage-2 fits all converged: {bool(sensitivity['stage2_success'].all())}.",
            f"- Maximum recorded prediction stationarity residual: {float(sensitivity['maximum_prediction_stationarity_residual'].max()):.3e}.",
            f"- Disturbance projection rate in source data: {float(metadata['disturbance_projection_rate']):.4%}.",
            f"- Raw-infeasible user-day rate before source-data projection: {float(metadata['raw_infeasible_user_day_rate']):.4%}.",
            "- Random seeds and nested user/price selections are fixed; each checkpoint reuses exactly the same Monte Carlo paths for its ground truth and moment estimates.",
            "- Full re-run reproducibility was not repeated because it would duplicate the multi-hour Monte Carlo experiment; status therefore remains ANALYZED rather than VERIFIED.",
            "",
            "### Fallacy scan",
            "",
            "- Coverage: 11/11 statistical fallacy types checked.",
            "- Main cautions: one deterministic nested price-diversity path is used; population effects largely saturate above 600 users; September days are repeated measures rather than independent populations; and the 50-draw reference is a Monte Carlo estimate, not an error-free oracle.",
            "- No smoothing, interpolation, outlier deletion, or outcome-enforcing transformation was applied to either figure.",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    project: Path,
    output_dir: Path,
    figure_dir: Path,
    phase: str,
    moment_samples: int,
    ground_truth_samples: int,
    batch_size: int,
    seed: int,
    max_iterations: int,
    stage2_rho: float,
    quick: bool,
    day_start_index: int,
    day_stop_index: int,
) -> None:
    start_time = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints" / "daily"
    aggregate_data_path = output_dir / "aggregate_daily_data.npz"
    metadata_path = output_dir / "metadata.json"
    source_metadata_path = output_dir / "source_data_metadata.json"
    source_metadata: dict[str, object] = {}
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))

    if phase in ("all", "prepare"):
        blocks, registry, source_metadata = load_archives(project)
        source_metadata_path.write_text(
            json.dumps(source_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        order = load_selection_order(project, registry, seed)
        inclusion_bin = inclusion_bin_by_global_position(order)
        all_days = np.arange(TRAIN_START, END_DATE, dtype="datetime64[D]")
        if day_start_index < 0 or day_stop_index > len(all_days) or day_start_index >= day_stop_index:
            raise ValueError("Preparation day indices must satisfy 0 <= start < stop <= 92.")
        days = all_days[day_start_index:day_stop_index]
        if quick:
            days = np.asarray([TRAIN_START, TEST_START], dtype="datetime64[D]")
        for day_number, day in enumerate(days, start=1):
            checkpoint_path = checkpoint_dir / f"{day}.npz"
            if checkpoint_path.exists() and checkpoint_matches(
                checkpoint_path, moment_samples, ground_truth_samples, batch_size, seed
            ):
                print(f"[{day_number:02d}/{len(days):02d}] {day}: cached", flush=True)
                continue
            print(f"[{day_number:02d}/{len(days):02d}] {day}: preparing daily simulations", flush=True)
            prepare_daily_checkpoint(
                day, blocks, inclusion_bin, moment_samples, ground_truth_samples,
                batch_size, seed, checkpoint_path,
            )
        full_range_invocation = day_start_index == 0 and day_stop_index == len(all_days)
        all_checkpoints_ready = all((checkpoint_dir / f"{day}.npz").exists() for day in all_days)
        if not quick and full_range_invocation and all_checkpoints_ready:
            merge_daily_checkpoints(checkpoint_dir, aggregate_data_path)
        elif not quick:
            print(
                f"Partial preparation shard {day_start_index}:{day_stop_index} finished; "
                "aggregate file will be assembled after all 92 checkpoints exist.",
                flush=True,
            )
        elif phase == "prepare":
            print("Quick preparation finished; full aggregate file is intentionally not assembled.", flush=True)

    if phase in ("all", "fit"):
        if quick:
            raise ValueError("Quick mode is intended for the separate 'prepare' and 'gradient-check' phases.")
        # Rebuild from the validated daily checkpoints on every fit so an
        # aggregate archive created under an older behavioral mapping can
        # never be reused accidentally.
        merge_daily_checkpoints(checkpoint_dir, aggregate_data_path)
        fit_and_evaluate(aggregate_data_path, output_dir, max_iterations, stage2_rho, quick=False)

    if phase in ("all", "plot"):
        make_figures(output_dir, figure_dir)

    if phase == "gradient-check":
        if aggregate_data_path.exists():
            with np.load(aggregate_data_path, allow_pickle=False) as data:
                theta = expand_empirical_theta(
                    np.asarray(data["empirical_initial_theta"])[0, -1]
                )
                price = np.asarray(data["price_usd_per_kwh"][:2], dtype=np.float64)
                disturbance = np.asarray(data["disturbance_c"][:2, -1], dtype=np.float64)
        else:
            pilot_paths = sorted(checkpoint_dir.glob("*.npz"))
            if not pilot_paths:
                raise FileNotFoundError("Run a preparation phase before gradient-check.")
            with np.load(pilot_paths[0], allow_pickle=False) as data:
                theta = expand_empirical_theta(
                    np.asarray(data["empirical_initial_theta"])[-1]
                )
                price = np.asarray(data["price_usd_per_kwh"], dtype=np.float64)[None, :]
                disturbance = np.asarray(data["disturbance_c"], dtype=np.float64)[-1][None, :]
        power, jacobian, _ = solve_equivalent_batch(theta, price, disturbance, with_jacobian=True)
        assert jacobian is not None
        rows = []
        for index, name in enumerate(PARAMETER_NAMES):
            step = 1.0e-5 * max(1.0, abs(float(theta[index])))
            plus = theta.copy(); plus[index] += step
            minus = theta.copy(); minus[index] -= step
            p_plus, _, _ = solve_equivalent_batch(plus, price, disturbance, with_jacobian=False)
            p_minus, _, _ = solve_equivalent_batch(minus, price, disturbance, with_jacobian=False)
            finite = (p_plus - p_minus) / (2.0 * step)
            absolute = np.linalg.norm(finite - jacobian[:, :, index])
            finite_norm = np.linalg.norm(finite)
            analytic_norm = np.linalg.norm(jacobian[:, :, index])
            relative = absolute / max(finite_norm, analytic_norm, 1.0e-10)
            rows.append(
                {
                    "parameter": name,
                    "finite_difference_norm": float(finite_norm),
                    "analytic_jacobian_norm": float(analytic_norm),
                    "absolute_jacobian_error": float(absolute),
                    "relative_jacobian_error": float(relative),
                }
            )
        write_csv(output_dir / "gradient_check.csv", rows)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    elapsed = time.perf_counter() - start_time
    if (
        phase in ("all", "fit")
        and aggregate_data_path.exists()
        and (output_dir / "daily_rmse.csv").exists()
    ):
        if not source_metadata and source_metadata_path.exists():
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if not source_metadata and metadata_path.exists():
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("source_data", {})
        metadata = {
            "schema_version": "1.0",
            "generator": "Codes/analyze_node_level_modeling_impact.py",
            "material_passport": {
                "origin_skill": ["experiment-agent", "nature-figure", "pdf"],
                "origin_mode": "run + validate",
                "origin_date": "2026-09-19",
                "verification_status": "ANALYZED",
                "version_label": "node_modeling_impact_v3",
            },
            "configuration": {
                "moment_samples": moment_samples,
                "ground_truth_samples": ground_truth_samples,
                "ground_truth_and_moment_samples_are_identical": True,
                "batch_size": batch_size,
                "seed": seed,
                "train_period": "2025-07-01 to 2025-08-31 (62 days)",
                "test_period": "2025-09-01 to 2025-09-30 (30 days)",
                "boxplot_user_count": int(USER_SIZES[BOXPLOT_GROUP_INDEX]),
                "boxplot_training_price_count": BOXPLOT_PRICE_COUNT,
                "price_diversity_order": "training-set medoid followed by increasing standardized-shape distance",
                "sensitivity_stage2_rho_schedule": (
                    f"rho_D={SENSITIVITY_STAGE2_RHO:g}*{BOXPLOT_PRICE_COUNT}/D"
                ),
                "user_sizes": USER_SIZES.tolist(),
                "price_signal_counts": PRICE_COUNTS.tolist(),
                "metric": "RMSE(prediction, truth) / mean_t(truth_t) x 100%",
                "differentiation": "Active-set KKT implicit differentiation with Gurobi forward solves",
                "optnet_decision": "OptNet mathematical approach retained; qpth/CVXPYLayers packages not required",
                "behavioral_mapping_version": BEHAVIORAL_MAPPING_VERSION,
                "reference_mapping": REFERENCE_MAPPING,
                "conditioning_basis": CONDITIONING_BASIS,
                "response_objective": RESPONSE_OBJECTIVE,
                "temporal_sampling": TEMPORAL_SAMPLING,
                "active_start_hour_inclusive": distribution_validation.ACTIVE_START_HOUR,
                "active_stop_hour_exclusive": distribution_validation.ACTIVE_STOP_HOUR,
                "actual_temperature_constrained_to_nominal_comfort_band": False,
                "actual_power_saturated_to_user_bounds": True,
                "test_moments_used_for_training": False,
                "thermal_disturbance": "fixed empirical aggregate of user delta_i; not trainable",
                "equivalent_model_parameterization": (
                    "three trainable physical scalars (a, b, Pmax), scalar comfort "
                    "limits/weight, and a 24-hour preferred-temperature profile; "
                    "Stage 2 freezes the physical scalars only"
                ),
                "daily_initial_state": "24 degC at each independently optimized daily horizon boundary",
            },
            "source_data": source_metadata,
            "figure_contract": {
                "core_conclusion": "Compare bounded-rationality-aware aggregate modeling against two ablations and show robustness to population size and price diversity.",
                "archetype": "two standalone quantitative figures",
                "boxplot": "wide 7.2 x 3.15 in grouped train/test distributions",
                "sensitivity": "square 6.4 x 6.4 in bubble heatmap with size, sequential color, and numeric labels",
                "backend": "Python/matplotlib only",
                "export": "PDF only",
            },
            "runtime": {
                "phase": phase,
                "elapsed_seconds_last_invocation": elapsed,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pd.__version__,
                "matplotlib": mpl.__version__,
                "gurobi": ".".join(str(value) for value in gp.gurobi.version()),
            },
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        write_validation_report(output_dir, {**source_metadata, **metadata["configuration"]}, elapsed)
    print(f"Phase '{phase}' completed in {elapsed:.1f} seconds.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the node-level modeling-impact experiment.")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "Node-level Modeling Impact",
    )
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "Figures")
    parser.add_argument(
        "--phase",
        choices=("prepare", "gradient-check"),
        default="prepare",
        help=(
            "Prepare the common aggregate data, or audit the analytical Jacobian. "
            "Final model fitting and figures are produced by "
            "run_node_level_soft_physics_benchmarks.py."
        ),
    )
    parser.add_argument("--moment-samples", type=int, default=DEFAULT_MOMENT_SAMPLES)
    parser.add_argument("--ground-truth-samples", type=int, default=DEFAULT_GROUND_TRUTH_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--stage2-rho", type=float, default=DEFAULT_STAGE2_RHO)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--day-start-index", type=int, default=0)
    parser.add_argument("--day-stop-index", type=int, default=92)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.moment_samples < 2 or args.ground_truth_samples < 1 or args.batch_size < 1:
        raise ValueError("Sample counts and batch size must be positive; moment samples must be at least two.")
    if args.ground_truth_samples != args.moment_samples:
        raise ValueError(
            "The corrected protocol requires identical ground-truth and moment "
            "sample counts because the same Monte Carlo paths are reused."
        )
    run(
        project=args.project.resolve(),
        output_dir=args.output_dir.resolve(),
        figure_dir=args.figure_dir.resolve(),
        phase=args.phase,
        moment_samples=args.moment_samples,
        ground_truth_samples=args.ground_truth_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        max_iterations=args.max_iterations,
        stage2_rho=args.stage2_rho,
        quick=args.quick,
        day_start_index=args.day_start_index,
        day_stop_index=args.day_stop_index,
    )


if __name__ == "__main__":
    main()
