"""Validate network-level flexibility-boundary propagation under bounded rationality.

This script implements the network-level case study in the manuscript.
It does not edit the manuscript.  The workflow is deliberately staged and
checkpointed:

1. Fit one nominal (Stage-1-only) equivalent TCL for each of the 26 user buses.
2. Use the original reference date (August 16, 2025).
3. Solve the 48 nominal root-boundary directions for either the fully aligned
   market-clearing setting or its branch-capacity-relaxed counterpart.
4. Evaluate node-level aggregate Gaussian moments at each boundary command.
5. Compare the proposed first-order Gaussian propagation with 500 exact
   Monte-Carlo re-optimizations of the perturbed network problem.
6. Export source data, diagnostics, and a single-axis publication figure.

The network Monte-Carlo layer never re-solves individual TCLs.  It consumes
only node-level aggregate moments.  Those moments are evaluated by the frozen
node-level direct-dynamics mapping, using each bus's user-specific Ecobee PMFs
and the Stage-1 equivalent TCL trajectory.
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
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

PLOT_ONLY_INVOCATION = any(
    arg == "--phase=plot"
    or (arg == "--phase" and index + 1 < len(sys.argv) and sys.argv[index + 1] == "plot")
    for index, arg in enumerate(sys.argv)
)
if not PLOT_ONLY_INVOCATION:
    import scipy
    from scipy.stats import norm, wasserstein_distance
    import analyze_node_level_distribution_accuracy as distribution
    import analyze_node_level_modeling_impact as node_model

    gp = distribution.gp
    GRB = gp.GRB
else:
    scipy = None
    distribution = None
    node_model = None
    gp = None
    GRB = None

HORIZON = 24
N_BUSES = 33
ROOT_BUS = 0
TAN_GAMMA = 0.5
NOMINAL_VOLTAGE_KV = 33.0
VOLTAGE_MIN_PU = 0.95
VOLTAGE_MAX_PU = 1.05
BRANCH_LIMIT_MW = 7.5
INFLEXIBLE_LOAD_SCALE = 1.15
PV_OUTPUT_SCALE = 0.80
ACTIVE_HOURS = np.arange(8, 20, dtype=np.int32)
TRAIN_END = np.datetime64("2025-09-01", "D")
REFERENCE_DATE = np.datetime64("2025-08-16", "D")
DEFAULT_SEED = 20260920
DEFAULT_MC_SAMPLES = 500
DEFAULT_MOMENT_SAMPLES = 200
DEFAULT_STAGE1_ITERATIONS = 120
BOUND_RELAXATION = 1.0e-7
ACTIVE_SLACK_TOL = 2.0e-5

INPUT_DIR = PROJECT_ROOT / "Data" / "Inputs_33kV_25MW_3000_JulSep"
OPTIMAL_DIR = (
    PROJECT_ROOT
    / "Outputs"
    / "Bounded Rationality Jul-Sep 3000"
    / "Optimal results"
)
PROBABILITY_DIR = (
    PROJECT_ROOT
    / "Outputs"
    / "Bounded Rationality Jul-Sep 3000"
    / "user_probability_distributions"
)
OUTPUT_ROOT = PROJECT_ROOT / "Outputs" / "Network-level Flexibility Figure 1"
SCENARIO_NAME = "market"
SCENARIO_LABEL = "market-clearing aligned"
OUTPUT_DIR = OUTPUT_ROOT / "market_clearing"
CHECKPOINT_ROOT = OUTPUT_DIR / "checkpoints"
STAGE1_CACHE_DIR = OUTPUT_ROOT / "stage1_node_cache"
FIGURE_BASE = PROJECT_ROOT / "Figures" / "network_level_flexibility_bounds_market_clearing"
STAGE1_CACHE_SCHEMA = 2

USER_NODE_IDS = np.asarray(
    [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 32],
    dtype=np.int32,
)
PV_NODE_IDS = np.asarray([5, 11, 18, 22, 27, 31], dtype=np.int32)


def configure_scenario(name: str) -> None:
    """Register one of the two fixed, directly comparable Figure-1 scenarios."""
    global SCENARIO_NAME, SCENARIO_LABEL
    global BRANCH_LIMIT_MW, INFLEXIBLE_LOAD_SCALE, PV_OUTPUT_SCALE
    global OUTPUT_DIR, CHECKPOINT_ROOT, FIGURE_BASE

    INFLEXIBLE_LOAD_SCALE = 1.15
    PV_OUTPUT_SCALE = 0.80
    SCENARIO_NAME = name
    if name == "market":
        SCENARIO_LABEL = "market-clearing aligned"
        BRANCH_LIMIT_MW = 7.5
        output_stem = "market_clearing"
        figure_stem = "network_level_flexibility_bounds_market_clearing"
    elif name == "relaxed":
        SCENARIO_LABEL = "branch-capacity relaxed"
        BRANCH_LIMIT_MW = 15.0
        output_stem = "relaxed_15mw"
        figure_stem = "network_level_flexibility_bounds_15MW"
    else:
        raise ValueError(f"Unknown scenario: {name}")

    OUTPUT_DIR = OUTPUT_ROOT / output_stem
    CHECKPOINT_ROOT = OUTPUT_DIR / "checkpoints"
    FIGURE_BASE = PROJECT_ROOT / "Figures" / figure_stem


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def direction_name(hour: int, sign: int) -> str:
    return f"{'upper' if sign > 0 else 'lower'}_{hour:02d}"


@dataclass
class NodeArchive:
    node_id: int
    user_count: int
    timestamps: np.ndarray
    days: np.ndarray
    price: np.ndarray
    a: np.ndarray
    b: np.ndarray
    power_max: np.ndarray
    temperature_min: float
    temperature_max: float
    temperature_penalty: np.ndarray
    disturbance: np.ndarray
    optimal_power: np.ndarray
    preset: np.ndarray
    delta_support: np.ndarray
    delta_probability: np.ndarray


@dataclass
class Stage1Bundle:
    node_ids: np.ndarray
    user_counts: np.ndarray
    theta: np.ndarray
    disturbances: np.ndarray
    nominal_power: np.ndarray
    prices: np.ndarray
    days: np.ndarray


def load_node_archives() -> list[NodeArchive]:
    archives: list[NodeArchive] = []
    reference_timestamps: np.ndarray | None = None
    reference_price: np.ndarray | None = None
    for node_id in USER_NODE_IDS:
        optimal_path = OPTIMAL_DIR / f"node_{node_id:02d}_optimal_results.npz"
        probability_path = PROBABILITY_DIR / f"node_{node_id:02d}_bounded_rationality.npz"
        if not optimal_path.exists() or not probability_path.exists():
            raise FileNotFoundError(f"Missing node input for bus {node_id:02d}.")
        with np.load(optimal_path, allow_pickle=False) as z:
            timestamps = np.asarray(z["timestamps"]).astype("datetime64[h]")
            price = np.asarray(z["price_usd_per_kwh"], dtype=np.float64)
            a = np.asarray(z["a"], dtype=np.float64)
            b = np.asarray(z["b_c_per_kw"], dtype=np.float64)
            pmax = np.asarray(z["power_max_kw"], dtype=np.float64)
            penalty = np.asarray(z["temperature_penalty"], dtype=np.float64)
            disturbance = np.asarray(z["disturbance_used_c"], dtype=np.float64)
            optimal_power = np.asarray(z["optimal_power_kw"], dtype=np.float64)
            tmin = float(z["temperature_min_c"])
            tmax = float(z["temperature_max_c"])
        with np.load(probability_path, allow_pickle=False) as z:
            preset = np.asarray(z["preset_setpoint_c"], dtype=np.float64)
            support = np.asarray(z["actual_setpoint_support_c"], dtype=np.float64)
            probability = np.asarray(z["conditional_probability"], dtype=np.float64)
        if probability.shape[0] != len(a):
            raise ValueError(f"Probability/user mismatch at bus {node_id}.")
        delta_support, delta_probability = distribution.reindex_absolute_pmf_as_relative_delta(
            probability, preset, support
        )
        if delta_probability.shape[:2] != (len(a), len(preset)) or delta_probability.shape[2] != len(delta_support):
            raise ValueError(f"Relative-delta PMF dimensions are inconsistent at bus {node_id}.")
        if reference_timestamps is None:
            reference_timestamps = timestamps.copy()
            reference_price = price.copy()
        elif not np.array_equal(reference_timestamps, timestamps) or not np.allclose(reference_price, price):
            raise ValueError(f"Timestamp or price mismatch at bus {node_id}.")
        days = timestamps.astype("datetime64[D]").reshape(-1, HORIZON)[:, 0]
        archives.append(
            NodeArchive(
                node_id=int(node_id),
                user_count=len(a),
                timestamps=timestamps,
                days=days,
                price=price.reshape(-1, HORIZON),
                a=a,
                b=b,
                power_max=pmax,
                temperature_min=tmin,
                temperature_max=tmax,
                temperature_penalty=penalty,
                disturbance=disturbance.reshape(len(a), -1, HORIZON),
                optimal_power=optimal_power.reshape(len(a), -1, HORIZON),
                preset=preset,
                delta_support=delta_support,
                delta_probability=delta_probability,
            )
        )
    if int(sum(item.user_count for item in archives)) != 3000:
        raise ValueError("The node archives do not contain exactly 3,000 users.")
    return archives


def build_empirical_theta(archive: NodeArchive) -> np.ndarray:
    base = np.asarray(
        [
            np.mean(archive.a),
            np.mean(archive.b),
            np.mean(archive.power_max),
            archive.temperature_min,
            archive.temperature_max,
            np.mean(archive.temperature_penalty),
        ],
        dtype=np.float64,
    )
    theta = node_model.expand_empirical_theta(base)
    return np.clip(theta, node_model.PARAMETER_LOWER, node_model.PARAMETER_UPPER)


def equivalent_dynamics_feasible(theta: np.ndarray, disturbances: np.ndarray) -> bool:
    """Check daily TCL feasibility by propagating the exact reachable interval."""
    a, b, power_max, temperature_min, temperature_max = map(float, theta[:5])
    for disturbance in np.asarray(disturbances, dtype=np.float64):
        reachable_min = node_model.INITIAL_TEMPERATURE
        reachable_max = node_model.INITIAL_TEMPERATURE
        for value in disturbance:
            next_min = a * reachable_min + float(value) + b * power_max
            next_max = a * reachable_max + float(value)
            reachable_min = max(temperature_min, next_min)
            reachable_max = min(temperature_max, next_max)
            if reachable_min > reachable_max + 1.0e-10:
                return False
    return True


def build_feasible_stage1_seed(
    archive: NodeArchive,
    training_disturbances: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the empirical seed or its smallest feasible comfort-band expansion.

    Averaging heterogeneous TCL dynamics can make the empirical equivalent
    model infeasible even when every underlying user trajectory is feasible.
    The fit objective cannot recover from such a seed because its infeasibility
    penalty is flat.  We therefore search only the two comfort bounds, in
    0.05-degree increments, and choose the feasible seed closest to the
    empirical one in normalized parameter space.  All parameters remain free
    during the subsequent Stage-1 fit.
    """
    empirical = build_empirical_theta(archive)
    if equivalent_dynamics_feasible(empirical, training_disturbances):
        return empirical, {
            "empirical_seed_feasible": True,
            "seed_temperature_min_c": float(empirical[3]),
            "seed_temperature_max_c": float(empirical[4]),
            "seed_comfort_expansion_c": 0.0,
        }

    lower_span = float(empirical[3] - node_model.PARAMETER_LOWER[3])
    upper_span = float(node_model.PARAMETER_UPPER[4] - empirical[4])
    lower_count = max(1, int(math.ceil(lower_span / 0.05)))
    upper_count = max(1, int(math.ceil(upper_span / 0.05)))
    lower_candidates = np.linspace(empirical[3], node_model.PARAMETER_LOWER[3], lower_count + 1)
    upper_candidates = np.linspace(empirical[4], node_model.PARAMETER_UPPER[4], upper_count + 1)
    parameter_range = node_model.PARAMETER_UPPER - node_model.PARAMETER_LOWER
    candidates: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for temperature_min in lower_candidates:
        for temperature_max in upper_candidates:
            candidate = empirical.copy()
            candidate[3] = temperature_min
            candidate[4] = temperature_max
            if not equivalent_dynamics_feasible(candidate, training_disturbances):
                continue
            normalized_change = (candidate[3:5] - empirical[3:5]) / parameter_range[3:5]
            total_expansion = (empirical[3] - candidate[3]) + (candidate[4] - empirical[4])
            score = (
                float(normalized_change @ normalized_change),
                float(total_expansion),
                float(empirical[3] - candidate[3]),
            )
            candidates.append((score, candidate))
    if not candidates:
        raise RuntimeError(
            f"No feasible Stage-1 comfort-band seed exists for bus {archive.node_id} "
            "within the registered parameter bounds."
        )
    _, seed = min(candidates, key=lambda item: item[0])
    return seed, {
        "empirical_seed_feasible": False,
        "seed_temperature_min_c": float(seed[3]),
        "seed_temperature_max_c": float(seed[4]),
        "seed_comfort_expansion_c": float((empirical[3] - seed[3]) + (seed[4] - empirical[4])),
    }


def validate_equivalent_model_by_day(
    theta: np.ndarray,
    prices: np.ndarray,
    disturbances: np.ndarray,
    node_id: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Re-solve a fitted equivalent TCL one day at a time.

    The fitting routine uses a 62-day block-diagonal QP.  On bus 20 Gurobi
    returned ``INF_OR_UNBD`` when that large validation QP was rebuilt even
    though the fitted point had already been accepted.  Daily solves avoid the
    block presolve ambiguity, while ``DualReductions=0`` makes any genuine
    infeasibility distinguishable and attributable to an exact day.
    """
    prices = np.asarray(prices, dtype=np.float64)
    disturbances = np.asarray(disturbances, dtype=np.float64)
    predictions = np.empty_like(prices)
    max_stationarity = 0.0
    max_primal_violation = 0.0
    for day_index in range(len(prices)):
        d_matrix, c, q_matrix, q_vector, g_matrix, h_vector = node_model.build_qp_components(
            theta, prices[day_index], disturbances[day_index]
        )
        model = gp.Model(f"stage1_validation_bus_{node_id:02d}_day_{day_index:02d}")
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0
        model.Params.FeasibilityTol = 1.0e-8
        model.Params.OptimalityTol = 1.0e-8
        model.Params.NumericFocus = 2
        state = model.addMVar(HORIZON, lb=-GRB.INFINITY, name="temperature")
        inequalities = model.addMConstr(g_matrix, state, "<", h_vector, name="limits")
        model.setMObjective(q_matrix, q_vector, 0.0, xc=state, sense=GRB.MINIMIZE)
        model.optimize()
        status = int(model.Status)
        if status != GRB.OPTIMAL:
            failure = {
                "node_id": int(node_id),
                "training_day_index": int(day_index),
                "gurobi_status": status,
                "theta": np.asarray(theta, dtype=np.float64).tolist(),
            }
            write_json(OUTPUT_DIR / f"stage1_failure_bus_{node_id:02d}.json", failure)
            raise RuntimeError(
                f"Daily Stage 1 validation failed at bus {node_id}, "
                f"training day {day_index}, Gurobi status {status}."
            )
        state_value = np.asarray(state.X, dtype=np.float64)
        predictions[day_index] = (d_matrix @ state_value - c) / float(theta[1])
        primal_violation = np.maximum(g_matrix @ state_value - h_vector, 0.0)
        max_primal_violation = max(max_primal_violation, float(np.max(primal_violation)))
        raw_pi = np.asarray(inequalities.Pi, dtype=np.float64)
        gradient = 2.0 * q_matrix @ state_value + q_vector
        residual_positive = gradient + g_matrix.T @ np.maximum(raw_pi, 0.0)
        residual_negative = gradient + g_matrix.T @ np.maximum(-raw_pi, 0.0)
        max_stationarity = max(
            max_stationarity,
            min(
                float(np.max(np.abs(residual_positive))),
                float(np.max(np.abs(residual_negative))),
            ),
        )
    return predictions, {
        "max_stationarity_residual": max_stationarity,
        "max_primal_violation": max_primal_violation,
    }


def load_stage1_node_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as z:
        schema = int(np.asarray(z["cache_schema"]).item())
        if schema != STAGE1_CACHE_SCHEMA:
            raise ValueError(f"Unsupported Stage 1 node-cache schema {schema} in {path}.")
        diagnostic = json.loads(str(np.asarray(z["diagnostic_json"]).item()))
        return (
            np.asarray(z["theta"], dtype=np.float64),
            np.asarray(z["disturbance"], dtype=np.float64),
            np.asarray(z["nominal"], dtype=np.float64),
            diagnostic,
        )


def save_stage1_node_cache(
    path: Path,
    theta: np.ndarray,
    disturbance: np.ndarray,
    nominal: np.ndarray,
    diagnostic: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_schema=np.asarray(STAGE1_CACHE_SCHEMA, dtype=np.int32),
        theta=np.asarray(theta, dtype=np.float64),
        disturbance=np.asarray(disturbance, dtype=np.float64),
        nominal=np.asarray(nominal, dtype=np.float64),
        diagnostic_json=np.asarray(json.dumps(dict(diagnostic), ensure_ascii=False)),
    )


def fit_stage1_models(
    archives: Sequence[NodeArchive],
    max_iterations: int,
    force: bool = False,
) -> Stage1Bundle:
    # Stage 1 is independent of the downstream network scenario, so both
    # figures deliberately consume the same registered node-model artifact.
    output_path = OUTPUT_ROOT / "stage1_node_models.npz"
    diagnostic_path = OUTPUT_ROOT / "stage1_fit_diagnostics.csv"
    parameter_path = OUTPUT_ROOT / "stage1_fitted_parameters.csv"
    summary_complete = False
    if output_path.exists() and diagnostic_path.exists() and parameter_path.exists():
        try:
            summary_complete = (
                len(pd.read_csv(diagnostic_path)) == len(archives)
                and len(pd.read_csv(parameter_path)) == len(archives) * len(node_model.PARAMETER_NAMES)
            )
        except (OSError, ValueError, pd.errors.ParserError):
            summary_complete = False
    if output_path.exists() and summary_complete and not force:
        with np.load(output_path, allow_pickle=False) as z:
            return Stage1Bundle(**{key: np.asarray(z[key]) for key in z.files})

    reference_days = archives[0].days
    prices = archives[0].price
    train_mask = reference_days < TRAIN_END
    theta_rows: list[np.ndarray] = []
    disturbance_rows: list[np.ndarray] = []
    nominal_rows: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, object]] = []
    for index, archive in enumerate(archives, start=1):
        cache_path = STAGE1_CACHE_DIR / f"bus_{archive.node_id:02d}_stage1.npz"
        if cache_path.exists() and not force:
            theta, disturbance, nominal, diagnostic = load_stage1_node_cache(cache_path)
            theta_rows.append(theta)
            disturbance_rows.append(disturbance)
            nominal_rows.append(nominal)
            diagnostic_rows.append(diagnostic)
            print(
                f"Loaded Stage 1 node {index:02d}/{len(archives):02d} "
                f"(bus {archive.node_id:02d}) from cache.",
                flush=True,
            )
            continue
        print(f"Fitting Stage 1 node {index:02d}/{len(archives):02d} (bus {archive.node_id:02d})...", flush=True)
        disturbance = archive.disturbance.mean(axis=0)
        nominal = archive.optimal_power.mean(axis=0)
        initial, seed_diagnostic = build_feasible_stage1_seed(
            archive, disturbance[train_mask]
        )
        fit = node_model.fit_equivalent_model(
            prices[train_mask],
            disturbance[train_mask],
            nominal[train_mask],
            initial,
            node_model.ALL_INDICES,
            max_iterations=max_iterations,
        )
        if not fit.success or not np.isfinite(fit.objective) or fit.objective >= 1.0e11:
            raise RuntimeError(f"Stage 1 failed at bus {archive.node_id}: {fit.message}")
        prediction, solve_diag = validate_equivalent_model_by_day(
            fit.theta,
            prices[train_mask],
            disturbance[train_mask],
            archive.node_id,
        )
        rmse = float(np.sqrt(np.mean((prediction - nominal[train_mask]) ** 2)))
        diagnostic = {
            "node_id": archive.node_id,
            "user_count": archive.user_count,
            "training_days": int(np.count_nonzero(train_mask)),
            "training_rmse_kw_per_user": rmse,
            "objective": fit.objective,
            "iterations": fit.iterations,
            "evaluations": fit.evaluations,
            "gradient_norm": fit.gradient_norm,
            "fit_max_stationarity_residual": fit.max_stationarity_residual,
            "prediction_max_stationarity_residual": solve_diag["max_stationarity_residual"],
            "prediction_max_primal_violation": solve_diag["max_primal_violation"],
            "success": fit.success,
            **seed_diagnostic,
        }
        save_stage1_node_cache(cache_path, fit.theta, disturbance, nominal, diagnostic)
        theta_rows.append(fit.theta)
        disturbance_rows.append(disturbance)
        nominal_rows.append(nominal)
        diagnostic_rows.append(diagnostic)
        print(
            f"Cached bus {archive.node_id:02d}: RMSE={rmse:.6f} kW/user, "
            f"max primal violation={solve_diag['max_primal_violation']:.2e}.",
            flush=True,
        )
    bundle = Stage1Bundle(
        node_ids=np.asarray([item.node_id for item in archives], dtype=np.int32),
        user_counts=np.asarray([item.user_count for item in archives], dtype=np.int32),
        theta=np.stack(theta_rows),
        disturbances=np.stack(disturbance_rows, axis=1),
        nominal_power=np.stack(nominal_rows, axis=1),
        prices=prices,
        days=reference_days,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle.__dict__)
    write_csv(diagnostic_path, diagnostic_rows)
    parameter_rows: list[dict[str, object]] = []
    for node_id, theta in zip(bundle.node_ids, bundle.theta):
        for name, value in zip(node_model.PARAMETER_NAMES, theta):
            parameter_rows.append({"node_id": int(node_id), "parameter": name, "value": float(value)})
    write_csv(parameter_path, parameter_rows)
    return bundle


def load_network_profiles(days: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    network = pd.read_csv(INPUT_DIR / "network_33kv.csv")
    if len(network) != 32:
        raise ValueError("The radial network must contain 32 branches.")
    n_hours = len(days) * HORIZON
    inflexible_kw = np.zeros((N_BUSES, n_hours), dtype=np.float64)
    pv_kw = np.zeros((N_BUSES, n_hours), dtype=np.float64)
    for bus in range(1, N_BUSES):
        data = np.load(INPUT_DIR / f"node_{bus:02d}.npy", allow_pickle=False)
        if data.dtype.names is None:
            if data.shape != (n_hours,):
                raise ValueError(f"PV profile shape mismatch at bus {bus}.")
            pv_kw[bus] = np.asarray(data, dtype=np.float64)
        else:
            if "inflexible_power_kw" not in data.dtype.names:
                raise ValueError(f"Inflexible-load field missing at bus {bus}.")
            inflexible_kw[bus] = np.asarray(data["inflexible_power_kw"], dtype=np.float64).sum(axis=0)
    return (
        network,
        (
            (INFLEXIBLE_LOAD_SCALE * inflexible_kw - PV_OUTPUT_SCALE * pv_kw)
            / 1000.0
        ).reshape(N_BUSES, len(days), HORIZON).transpose(1, 0, 2),
        (
            TAN_GAMMA * INFLEXIBLE_LOAD_SCALE * inflexible_kw / 1000.0
        ).reshape(N_BUSES, len(days), HORIZON).transpose(1, 0, 2),
    )


def equivalent_polytope(theta: np.ndarray, disturbance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, pmax, tmin, tmax = map(float, theta[:5])
    lower = np.zeros((HORIZON, HORIZON), dtype=np.float64)
    free = np.empty(HORIZON, dtype=np.float64)
    previous_free = node_model.INITIAL_TEMPERATURE
    for t in range(HORIZON):
        previous_free = a * previous_free + float(disturbance[t])
        free[t] = previous_free
        for j in range(t + 1):
            lower[t, j] = b * (a ** (t - j))
    identity = np.eye(HORIZON, dtype=np.float64)
    a_matrix = np.vstack((identity, -identity, lower, -lower))
    g_vector = np.r_[
        np.full(HORIZON, pmax + BOUND_RELAXATION),
        np.full(HORIZON, BOUND_RELAXATION),
        np.full(HORIZON, tmax) - free + BOUND_RELAXATION,
        free - np.full(HORIZON, tmin) + BOUND_RELAXATION,
    ]
    return a_matrix, g_vector, lower, free


class BoundaryModel:
    def __init__(
        self,
        network: pd.DataFrame,
        node_ids: np.ndarray,
        user_counts: np.ndarray,
        polytopes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        base_p_mw: np.ndarray,
        base_q_mvar: np.ndarray,
    ) -> None:
        self.network = network.reset_index(drop=True)
        self.node_ids = np.asarray(node_ids, dtype=np.int32)
        self.user_counts = np.asarray(user_counts, dtype=np.float64)
        self.polytopes = polytopes
        self.base_rhs = [item[1].copy() for item in polytopes]
        self.node_index = {int(bus): index for index, bus in enumerate(self.node_ids)}
        self.branch_to_index = {
            (int(row.from_bus), int(row.to_bus)): index
            for index, row in enumerate(self.network.itertuples(index=False))
        }
        self.incoming = {
            int(row.to_bus): index for index, row in enumerate(self.network.itertuples(index=False))
        }
        self.children: dict[int, list[int]] = {bus: [] for bus in range(N_BUSES)}
        for index, row in enumerate(self.network.itertuples(index=False)):
            self.children[int(row.from_bus)].append(index)
        self.root_branch = self.branch_to_index[(0, 1)]
        self.model = gp.Model("network_boundary")
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.model.Params.Method = 1
        self.model.Params.Presolve = 1
        self.model.Params.FeasibilityTol = 1.0e-8
        self.model.Params.OptimalityTol = 1.0e-8
        self.model.Params.NumericFocus = 1
        self.p = self.model.addMVar((len(self.node_ids), HORIZON), lb=-GRB.INFINITY, name="p_node_kw_per_user")
        flow_lb = -BRANCH_LIMIT_MW
        flow_ub = BRANCH_LIMIT_MW
        self.branch_p = self.model.addMVar((len(self.network), HORIZON), lb=flow_lb, ub=flow_ub, name="branch_p_mw")
        self.branch_q = self.model.addMVar((len(self.network), HORIZON), lb=flow_lb, ub=flow_ub, name="branch_q_mvar")
        v_lb = np.full((N_BUSES, HORIZON), -GRB.INFINITY, dtype=np.float64)
        v_ub = np.full((N_BUSES, HORIZON), GRB.INFINITY, dtype=np.float64)
        v_lb[0] = NOMINAL_VOLTAGE_KV**2
        v_ub[0] = NOMINAL_VOLTAGE_KV**2
        v_lb[1:] = (VOLTAGE_MIN_PU * NOMINAL_VOLTAGE_KV) ** 2
        v_ub[1:] = (VOLTAGE_MAX_PU * NOMINAL_VOLTAGE_KV) ** 2
        self.voltage_sq = self.model.addMVar((N_BUSES, HORIZON), lb=v_lb, ub=v_ub, name="voltage_sq_kv2")
        self.node_constraints = []
        for index, (a_matrix, g_vector, _, _) in enumerate(polytopes):
            self.node_constraints.append(
                self.model.addMConstr(a_matrix, self.p[index], "<", g_vector, name=f"node_polytope_{self.node_ids[index]:02d}")
            )
        for bus in range(1, N_BUSES):
            incoming_index = self.incoming[bus]
            child_indices = self.children[bus]
            user_index = self.node_index.get(bus)
            scale = 0.0 if user_index is None else self.user_counts[user_index] / 1000.0
            for hour in range(HORIZON):
                tcl_p = 0.0 if user_index is None else scale * self.p[user_index, hour]
                tcl_q = 0.0 if user_index is None else TAN_GAMMA * scale * self.p[user_index, hour]
                child_p = gp.quicksum(self.branch_p[j, hour] for j in child_indices)
                child_q = gp.quicksum(self.branch_q[j, hour] for j in child_indices)
                self.model.addConstr(
                    self.branch_p[incoming_index, hour]
                    == child_p + float(base_p_mw[bus, hour]) + tcl_p,
                    name=f"p_balance_{bus:02d}_{hour:02d}",
                )
                self.model.addConstr(
                    self.branch_q[incoming_index, hour]
                    == child_q + float(base_q_mvar[bus, hour]) + tcl_q,
                    name=f"q_balance_{bus:02d}_{hour:02d}",
                )
        for branch_index, row in enumerate(self.network.itertuples(index=False)):
            parent = int(row.from_bus)
            child = int(row.to_bus)
            r = float(row.resistance_ohm)
            x = float(row.reactance_ohm)
            for hour in range(HORIZON):
                self.model.addConstr(
                    self.voltage_sq[parent, hour] - self.voltage_sq[child, hour]
                    == 2.0 * (r * self.branch_p[branch_index, hour] + x * self.branch_q[branch_index, hour]),
                    name=f"voltage_{parent:02d}_{child:02d}_{hour:02d}",
                )
        self.model.update()

    def set_perturbations(self, epsilon: np.ndarray | None) -> None:
        if epsilon is None:
            epsilon = np.zeros((len(self.node_ids), HORIZON), dtype=np.float64)
        for index, (constraint, polytope, base_rhs) in enumerate(
            zip(self.node_constraints, self.polytopes, self.base_rhs)
        ):
            rhs = base_rhs + polytope[0] @ epsilon[index]
            constraint.RHS = rhs
        self.model.update()

    def solve(self, hour: int, sign: int, require_duals: bool = True) -> dict[str, object]:
        self.model.setObjective(sign * self.branch_p[self.root_branch, hour], GRB.MAXIMIZE)
        self.model.optimize()
        if int(self.model.Status) != int(GRB.OPTIMAL):
            raise RuntimeError(f"Network boundary solve failed with status {int(self.model.Status)}.")
        p_value = np.asarray(self.p.X, dtype=np.float64)
        branch_p = np.asarray(self.branch_p.X, dtype=np.float64)
        branch_q = np.asarray(self.branch_q.X, dtype=np.float64)
        voltage_sq = np.asarray(self.voltage_sq.X, dtype=np.float64)
        support = float(self.model.ObjVal)
        boundary = support if sign > 0 else -support
        dual = []
        kappa = []
        node_active = []
        for constraint, polytope in zip(self.node_constraints, self.polytopes):
            pi = np.asarray(constraint.Pi, dtype=np.float64)
            # Gurobi returns non-negative shadow prices for <= constraints in a maximization LP.
            if np.max(-pi) > 1.0e-6 and np.max(pi) <= 1.0e-9:
                pi = -pi
            pi = np.maximum(pi, 0.0)
            dual.append(pi)
            kappa.append(polytope[0].T @ pi)
            node_active.append(np.asarray(constraint.Slack, dtype=np.float64) <= ACTIVE_SLACK_TOL)
        voltage_pu = np.sqrt(np.maximum(voltage_sq, 0.0)) / NOMINAL_VOLTAGE_KV
        signature = self.active_signature(node_active, branch_p, branch_q, voltage_pu)
        return {
            "support": support,
            "boundary": boundary,
            "root_power_mw": float(branch_p[self.root_branch, hour]),
            "node_power_kw_per_user": p_value,
            "branch_p_mw": branch_p,
            "branch_q_mvar": branch_q,
            "voltage_pu": voltage_pu,
            "dual": np.stack(dual) if require_duals else None,
            "kappa": np.stack(kappa) if require_duals else None,
            "active_signature": signature,
            "minimum_voltage_pu": float(np.min(voltage_pu[:, hour])),
            "maximum_voltage_pu": float(np.max(voltage_pu[:, hour])),
            "maximum_abs_branch_p_mw": float(np.max(np.abs(branch_p[:, hour]))),
            "maximum_abs_branch_q_mvar": float(np.max(np.abs(branch_q[:, hour]))),
        }

    def active_signature(
        self,
        node_active: Sequence[np.ndarray],
        branch_p: np.ndarray,
        branch_q: np.ndarray,
        voltage_pu: np.ndarray,
    ) -> np.ndarray:
        pieces = [np.concatenate(node_active)]
        pieces.extend(
            [
                np.abs(np.abs(branch_p) - BRANCH_LIMIT_MW).ravel() <= ACTIVE_SLACK_TOL,
                np.abs(np.abs(branch_q) - BRANCH_LIMIT_MW).ravel() <= ACTIVE_SLACK_TOL,
                np.minimum(
                    np.abs(voltage_pu[1:] - VOLTAGE_MIN_PU),
                    np.abs(voltage_pu[1:] - VOLTAGE_MAX_PU),
                ).ravel()
                <= ACTIVE_SLACK_TOL,
            ]
        )
        return np.concatenate(pieces).astype(bool)


def state_from_command(polytope: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], command: np.ndarray) -> np.ndarray:
    _, _, lower, free = polytope
    return free + lower @ command


def checkpoint_dir_for_day(day: np.datetime64) -> Path:
    configuration = (
        f"{day}_load{INFLEXIBLE_LOAD_SCALE:.2f}_pv{PV_OUTPUT_SCALE:.2f}_"
        f"branch{BRANCH_LIMIT_MW:.1f}_v{VOLTAGE_MIN_PU:.2f}-{VOLTAGE_MAX_PU:.2f}"
    ).replace(".", "p")
    return CHECKPOINT_ROOT / configuration


def evaluate_node_moments(
    archive: NodeArchive,
    theta: np.ndarray,
    disturbance: np.ndarray,
    command: np.ndarray,
    state: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    temperatures = np.broadcast_to(state[None, :], (archive.user_count, HORIZON))
    hourly_probability, outside_low, outside_high = distribution.interpolate_conditional_pmf(
        archive.delta_probability, archive.preset, temperatures
    )
    cumulative = np.cumsum(hourly_probability[:, ACTIVE_HOURS], axis=-1)
    cumulative[:, :, -1] = 1.0
    rng = np.random.default_rng(seed)
    random_values = rng.random((samples, archive.user_count, len(ACTIVE_HOURS)))
    sampled_index = np.sum(random_values[..., None] > cumulative[None, ...], axis=-1)
    sampled_index = np.minimum(sampled_index, len(archive.delta_support) - 1)
    sampled_delta = archive.delta_support[sampled_index]
    target = np.broadcast_to(state[None, None, :], (samples, archive.user_count, HORIZON)).copy()
    target[:, :, ACTIVE_HOURS] += sampled_delta
    a, b, pmax = map(float, theta[:3])
    previous = np.full((samples, archive.user_count), node_model.INITIAL_TEMPERATURE, dtype=np.float64)
    actual_power = np.empty_like(target)
    saturation_count = 0
    for hour in range(HORIZON):
        raw_power = (target[:, :, hour] - a * previous - float(disturbance[hour])) / b
        power = np.clip(raw_power, 0.0, pmax)
        actual_power[:, :, hour] = power
        previous = a * previous + b * power + float(disturbance[hour])
        saturation_count += int(np.count_nonzero(np.abs(power - raw_power) > 1.0e-10))
    deviation = actual_power - command[None, None, :]
    user_mean = deviation.mean(axis=0)
    centered = deviation - user_mean[None, :, :]
    user_covariance = np.einsum("sut,suv->utv", centered, centered, optimize=True) / samples
    aggregate_mean = user_mean.mean(axis=0)
    aggregate_covariance = user_covariance.sum(axis=0) / (archive.user_count**2)
    aggregate_covariance = 0.5 * (aggregate_covariance + aggregate_covariance.T)
    eigval, eigvec = np.linalg.eigh(aggregate_covariance)
    eigval = np.maximum(eigval, 0.0)
    aggregate_covariance = (eigvec * eigval) @ eigvec.T
    return (
        aggregate_mean,
        aggregate_covariance,
        {
            "conditioning_outside_low_rate": outside_low / (archive.user_count * HORIZON),
            "conditioning_outside_high_rate": outside_high / (archive.user_count * HORIZON),
            "power_saturation_rate": saturation_count / (samples * archive.user_count * HORIZON),
            "minimum_covariance_eigenvalue": float(np.min(eigval)),
        },
    )


def gaussian_draws(mean: np.ndarray, covariance: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigval, eigvec = np.linalg.eigh(covariance)
    eigval = np.maximum(eigval, 0.0)
    transform = eigvec @ np.diag(np.sqrt(eigval))
    return mean[None, :] + rng.standard_normal((samples, HORIZON)) @ transform.T


def run_directions(
    archives: Sequence[NodeArchive],
    network: pd.DataFrame,
    bundle: Stage1Bundle,
    base_p: np.ndarray,
    base_q: np.ndarray,
    day_index: int,
    mc_samples: int,
    moment_samples: int,
    seed: int,
    force: bool = False,
) -> None:
    checkpoint_dir = checkpoint_dir_for_day(bundle.days[day_index])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    polytopes = [
        equivalent_polytope(bundle.theta[j], bundle.disturbances[day_index, j])
        for j in range(len(bundle.node_ids))
    ]
    model = BoundaryModel(
        network,
        bundle.node_ids,
        bundle.user_counts,
        polytopes,
        base_p[day_index],
        base_q[day_index],
    )
    for hour in range(HORIZON):
        for sign in (1, -1):
            name = direction_name(hour, sign)
            checkpoint = checkpoint_dir / f"{name}.npz"
            if checkpoint.exists() and not force:
                print(f"Skipping cached {name}.", flush=True)
                continue
            print(f"Solving {name}: nominal + moments + {mc_samples} MC re-optimizations...", flush=True)
            model.set_perturbations(None)
            nominal = model.solve(hour, sign, require_duals=True)
            command = np.asarray(nominal["node_power_kw_per_user"], dtype=np.float64)
            kappa = np.asarray(nominal["kappa"], dtype=np.float64)
            means = np.empty((len(archives), HORIZON), dtype=np.float64)
            covariances = np.empty((len(archives), HORIZON, HORIZON), dtype=np.float64)
            support_diag = np.empty((len(archives), 4), dtype=np.float64)
            for node_index, archive in enumerate(archives):
                state = state_from_command(polytopes[node_index], command[node_index])
                mean, covariance, diagnostic = evaluate_node_moments(
                    archive,
                    bundle.theta[node_index],
                    bundle.disturbances[day_index, node_index],
                    command[node_index],
                    state,
                    moment_samples,
                    seed + hour * 100003 + (1 if sign > 0 else 2) * 1009 + archive.node_id,
                )
                means[node_index] = mean
                covariances[node_index] = covariance
                support_diag[node_index] = [
                    diagnostic["conditioning_outside_low_rate"],
                    diagnostic["conditioning_outside_high_rate"],
                    diagnostic["power_saturation_rate"],
                    diagnostic["minimum_covariance_eigenvalue"],
                ]
            node_mean_contribution = np.einsum("nt,nt->n", kappa, means)
            node_variance_contribution = np.einsum("nt,ntv,nv->n", kappa, covariances, kappa, optimize=True)
            proposed_mean_shift = float(np.sum(node_mean_contribution))
            proposed_variance = max(float(np.sum(node_variance_contribution)), 0.0)
            proposed_std = math.sqrt(proposed_variance)
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, day_index, hour, 71 if sign > 0 else 73])
            )
            epsilon = np.stack(
                [gaussian_draws(means[n], covariances[n], mc_samples, rng) for n in range(len(archives))],
                axis=1,
            )
            mc_support = np.empty(mc_samples, dtype=np.float64)
            active_switch = np.zeros(mc_samples, dtype=bool)
            nominal_signature = np.asarray(nominal["active_signature"], dtype=bool)
            for sample_index in range(mc_samples):
                model.set_perturbations(epsilon[sample_index])
                result = model.solve(hour, sign, require_duals=False)
                mc_support[sample_index] = float(result["support"])
                active_switch[sample_index] = not np.array_equal(
                    nominal_signature, np.asarray(result["active_signature"], dtype=bool)
                )
                if (sample_index + 1) % 100 == 0:
                    print(f"  {name}: {sample_index + 1}/{mc_samples}", flush=True)
            model.set_perturbations(None)
            np.savez_compressed(
                checkpoint,
                hour=np.asarray(hour, dtype=np.int16),
                sign=np.asarray(sign, dtype=np.int8),
                date=np.asarray(str(bundle.days[day_index])),
                nominal_support=np.asarray(float(nominal["support"])),
                nominal_boundary=np.asarray(float(nominal["boundary"])),
                nominal_root_power_mw=np.asarray(float(nominal["root_power_mw"])),
                node_power_kw_per_user=command.astype(np.float32),
                kappa=kappa.astype(np.float64),
                node_mean=means.astype(np.float64),
                node_covariance=covariances.astype(np.float64),
                node_mean_contribution=node_mean_contribution.astype(np.float64),
                node_variance_contribution=node_variance_contribution.astype(np.float64),
                support_diagnostics=support_diag.astype(np.float64),
                proposed_mean_shift=np.asarray(proposed_mean_shift),
                proposed_std=np.asarray(proposed_std),
                mc_support=mc_support.astype(np.float64),
                active_switch=active_switch,
                nominal_minimum_voltage_pu=np.asarray(float(nominal["minimum_voltage_pu"])),
                nominal_maximum_voltage_pu=np.asarray(float(nominal["maximum_voltage_pu"])),
                nominal_maximum_abs_branch_p_mw=np.asarray(float(nominal["maximum_abs_branch_p_mw"])),
            )


def compile_results(selected_day: np.datetime64) -> pd.DataFrame:
    checkpoint_dir = checkpoint_dir_for_day(selected_day)
    rows: list[dict[str, object]] = []
    for hour in range(HORIZON):
        for sign in (1, -1):
            name = direction_name(hour, sign)
            checkpoint = checkpoint_dir / f"{name}.npz"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing direction checkpoint: {checkpoint}")
            with np.load(checkpoint, allow_pickle=False) as z:
                nominal_support = float(z["nominal_support"])
                nominal_boundary = float(z["nominal_boundary"])
                mean_shift = float(z["proposed_mean_shift"])
                proposed_std = float(z["proposed_std"])
                mc_support = np.asarray(z["mc_support"], dtype=np.float64)
                mc_boundary = mc_support if sign > 0 else -mc_support
                proposed_boundary_mean = (
                    nominal_support + mean_shift if sign > 0 else -nominal_support - mean_shift
                )
                proposed_low = proposed_boundary_mean - 1.96 * proposed_std
                proposed_high = proposed_boundary_mean + 1.96 * proposed_std
                quantile_low, quantile_high = np.quantile(mc_boundary, [0.025, 0.975])
                mc_mean = float(np.mean(mc_boundary))
                mc_std = float(np.std(mc_boundary, ddof=0))
                gaussian_quantiles = proposed_boundary_mean + proposed_std * norm.ppf(
                    (np.arange(len(mc_boundary)) + 0.5) / len(mc_boundary)
                )
                w1 = float(wasserstein_distance(mc_boundary, gaussian_quantiles))
                coverage_tolerance = max(1.0e-9, 1.0e-8 * max(1.0, abs(proposed_boundary_mean)))
                coverage = float(
                    np.mean(
                        (mc_boundary >= proposed_low - coverage_tolerance)
                        & (mc_boundary <= proposed_high + coverage_tolerance)
                    )
                )
                rows.append(
                    {
                        "hour": hour,
                        "direction": "upper" if sign > 0 else "lower",
                        "nominal_boundary_mw": nominal_boundary,
                        "proposed_mean_mw": proposed_boundary_mean,
                        "proposed_p025_mw": proposed_low,
                        "proposed_p975_mw": proposed_high,
                        "mc_mean_mw": mc_mean,
                        "mc_p025_mw": float(quantile_low),
                        "mc_p975_mw": float(quantile_high),
                        "proposed_std_mw": proposed_std,
                        "mc_std_mw": mc_std,
                        "mean_absolute_error_mw": abs(proposed_boundary_mean - mc_mean),
                        "p025_absolute_error_mw": abs(proposed_low - quantile_low),
                        "p975_absolute_error_mw": abs(proposed_high - quantile_high),
                        "wasserstein_1_mw": w1,
                        "proposed_95_coverage": coverage,
                        "active_set_switch_rate": float(np.mean(z["active_switch"])),
                        "nominal_minimum_voltage_pu": float(z["nominal_minimum_voltage_pu"]),
                        "nominal_maximum_voltage_pu": float(z["nominal_maximum_voltage_pu"]),
                        "nominal_maximum_abs_branch_p_mw": float(z["nominal_maximum_abs_branch_p_mw"]),
                        "conditioning_outside_low_rate": float(np.mean(z["support_diagnostics"][:, 0])),
                        "conditioning_outside_high_rate": float(np.mean(z["support_diagnostics"][:, 1])),
                        "power_saturation_rate": float(np.mean(z["support_diagnostics"][:, 2])),
                        "mc_samples": len(mc_boundary),
                    }
                )
    write_csv(OUTPUT_DIR / "network_boundary_results.csv", rows)
    frame = pd.DataFrame(rows)
    upper = frame[frame["direction"] == "upper"].sort_values("hour")
    lower = frame[frame["direction"] == "lower"].sort_values("hour")
    np.savez_compressed(
        OUTPUT_DIR / "figure_source_data.npz",
        hour=np.arange(HORIZON, dtype=np.int16),
        upper_nominal=upper["nominal_boundary_mw"].to_numpy(),
        upper_proposed_mean=upper["proposed_mean_mw"].to_numpy(),
        upper_proposed_low=upper["proposed_p025_mw"].to_numpy(),
        upper_proposed_high=upper["proposed_p975_mw"].to_numpy(),
        upper_mc_mean=upper["mc_mean_mw"].to_numpy(),
        upper_mc_low=upper["mc_p025_mw"].to_numpy(),
        upper_mc_high=upper["mc_p975_mw"].to_numpy(),
        upper_switch_rate=upper["active_set_switch_rate"].to_numpy(),
        lower_nominal=lower["nominal_boundary_mw"].to_numpy(),
        lower_proposed_mean=lower["proposed_mean_mw"].to_numpy(),
        lower_proposed_low=lower["proposed_p025_mw"].to_numpy(),
        lower_proposed_high=lower["proposed_p975_mw"].to_numpy(),
        lower_mc_mean=lower["mc_mean_mw"].to_numpy(),
        lower_mc_low=lower["mc_p025_mw"].to_numpy(),
        lower_mc_high=lower["mc_p975_mw"].to_numpy(),
        lower_switch_rate=lower["active_set_switch_rate"].to_numpy(),
    )
    return frame


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    arial = [item for item in font_manager.findSystemFonts() if Path(item).name.lower() in {"arial.ttf", "arialbd.ttf"}]
    if arial:
        for item in arial:
            font_manager.fontManager.addfont(item)
    mpl.rcParams.update(
        {
            "font.size": 12.5,
            "axes.labelsize": 14.0,
            "axes.labelpad": 3.5,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.8,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def make_figure() -> None:
    source_path = OUTPUT_DIR / "figure_source_data.npz"
    if not source_path.exists():
        raise FileNotFoundError("Compile results before plotting.")
    with np.load(source_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key], dtype=np.float64) for key in z.files}
    configure_matplotlib()
    neutral = "#3F3F3F"
    proposed = "#0F4D92"
    benchmark = "#D97724"
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
    hour = data["hour"]
    for prefix in ("upper", "lower"):
        show_label = prefix == "upper"
        ax.plot(
            hour,
            data[f"{prefix}_nominal"],
            color=neutral,
            linestyle="--",
            linewidth=1.25,
            label="No-BR" if show_label else None,
            zorder=4,
        )
        ax.plot(
            hour,
            data[f"{prefix}_proposed_mean"],
            color=proposed,
            label="Proposed mean" if show_label else None,
            zorder=5,
        )
        ax.plot(
            hour,
            data[f"{prefix}_mc_mean"],
            color=benchmark,
            linestyle=(0, (3.0, 1.6)),
            label="Monte Carlo mean" if show_label else None,
            zorder=5,
        )
    for prefix, label in (("upper", "upper limit"), ("lower", "lower limit")):
        label_y = max(
            data[f"{prefix}_nominal"][1],
            data[f"{prefix}_proposed_mean"][1],
            data[f"{prefix}_mc_mean"][1],
        )
        ax.annotate(
            label,
            # Start the direct label safely inside the plotting area so its
            # left edge never crosses the y axis after PDF scaling.
            xy=(0.7, label_y),
            xytext=(2, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=13.5,
            fontweight="semibold",
            color=neutral,
            zorder=6,
        )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Root-node active power (MW)")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.margins(x=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.01),
        ncol=3,
        columnspacing=1.0,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.19, top=0.82)
    FIGURE_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def build_report(frame: pd.DataFrame, bundle: Stage1Bundle, day_index: int, elapsed_seconds: float) -> None:
    mean_mae = float(frame["mean_absolute_error_mw"].mean())
    max_mae = float(frame["mean_absolute_error_mw"].max())
    mean_w1 = float(frame["wasserstein_1_mw"].mean())
    mean_coverage = float(frame["proposed_95_coverage"].mean())
    switch_rate = float(frame["active_set_switch_rate"].mean())
    nondegenerate = (frame["proposed_p975_mw"] - frame["proposed_p025_mw"]) > 1.0e-6
    stable = frame["active_set_switch_rate"] < 0.01
    transitioning = frame["active_set_switch_rate"] >= 0.5
    stable_nondegenerate = stable & nondegenerate
    stable_coverage = float(frame.loc[stable_nondegenerate, "proposed_95_coverage"].mean())
    stable_mean_mae = float(frame.loc[stable_nondegenerate, "mean_absolute_error_mw"].mean())
    transitioning_coverage = float(frame.loc[transitioning, "proposed_95_coverage"].mean())
    transitioning_mean_mae = float(frame.loc[transitioning, "mean_absolute_error_mw"].mean())
    network_active = int(
        np.count_nonzero(
            (frame["nominal_minimum_voltage_pu"] <= VOLTAGE_MIN_PU + 5.0e-5)
            | (frame["nominal_maximum_voltage_pu"] >= VOLTAGE_MAX_PU - 5.0e-5)
            | (frame["nominal_maximum_abs_branch_p_mw"] >= BRANCH_LIMIT_MW - 5.0e-5)
        )
    )
    close_match = frame["mean_absolute_error_mw"] <= 0.01
    upper = frame[frame["direction"] == "upper"].sort_values("hour")
    lower = frame[frame["direction"] == "lower"].sort_values("hour")
    nominal_width = float(
        np.sum(upper["nominal_boundary_mw"].to_numpy() - lower["nominal_boundary_mw"].to_numpy())
    )
    proposed_width = float(
        np.sum(upper["proposed_mean_mw"].to_numpy() - lower["proposed_mean_mw"].to_numpy())
    )
    mc_width = float(np.sum(upper["mc_mean_mw"].to_numpy() - lower["mc_mean_mw"].to_numpy()))
    report = f"""# Network-level flexibility propagation validation

## Material Passport

- Origin Skill: experiment-agent + nature-figure + pdf
- Origin Mode: run + validate
- Origin Date: 2026-09-23
- Verification Status: ANALYZED
- Version Label: network_flexibility_two_scenario_v1_{SCENARIO_NAME}

## Configuration

- Scenario: {SCENARIO_LABEL}.
- Selected date: {bundle.days[day_index]}
- Date rule: fixed to the original August 16 reference day; no outcome-based date selection.
- User nodes: {len(bundle.node_ids)}; users: {int(bundle.user_counts.sum())}.
- Voltage limits: [{VOLTAGE_MIN_PU:.2f}, {VOLTAGE_MAX_PU:.2f}] p.u.
- Uniform active/reactive branch limits: +/-{BRANCH_LIMIT_MW:.1f} MW/MVAr.
- Non-flexible-load scaling: {INFLEXIBLE_LOAD_SCALE:.2f}.
- PV-output scaling: {PV_OUTPUT_SCALE:.2f}.
- Node moment samples: {DEFAULT_MOMENT_SAMPLES} per user-equivalent response.
- Network Monte Carlo samples: {DEFAULT_MC_SAMPLES} per direction.
- Directions: 24 upper and 24 lower pointwise support problems.
- Cross-node deviation assumption: conditionally independent.
- Runtime: {elapsed_seconds:.1f} s.

## Accuracy summary

- Mean absolute error of propagated means: {mean_mae:.6f} MW.
- Maximum absolute error of propagated means: {max_mae:.6f} MW.
- Mean 1-Wasserstein distance: {mean_w1:.6f} MW.
- Mean coverage of the proposed Gaussian 95% prediction interval: {mean_coverage:.3%}.
- Mean active-set switch rate across Monte Carlo scenarios: {switch_rate:.3%}.
- Nominal directions with an active voltage or branch limit: {network_active}/48.
- Stable, non-degenerate directions: {int(stable_nondegenerate.sum())}/48; mean coverage {stable_coverage:.3%}; mean error {stable_mean_mae:.6f} MW.
- Active-set-transition directions: {int(transitioning.sum())}/48; mean coverage {transitioning_coverage:.3%}; mean error {transitioning_mean_mae:.6f} MW.
- Proposed and Monte Carlo means agree within 0.01 MW in {int(close_match.sum())}/48 directions: {int(close_match[frame['direction'] == 'upper'].sum())}/24 upper and {int(close_match[frame['direction'] == 'lower'].sum())}/24 lower.
- Daily flexibility-width indices: No BR {nominal_width:.6f}, Proposed {proposed_width:.6f}, and Monte Carlo {mc_width:.6f} MW h.
- Width reductions relative to No BR: Proposed {1.0 - proposed_width / nominal_width:.3%}; Monte Carlo {1.0 - mc_width / nominal_width:.3%}.

## Interpretation boundary

The Monte Carlo benchmark re-solves the perturbed network problem and is the
ground truth for network propagation conditional on the supplied node-level
Gaussian moments.  It is not an independent ground truth for the behavioral
model itself.  That upstream distribution was evaluated in the node-level case.

The proposed result is a local first-order approximation.  Active-set switches
are therefore reported explicitly rather than removed or smoothed.
"""
    (OUTPUT_DIR / "validation_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "schema_version": "1.4",
        "scenario": SCENARIO_NAME,
        "scenario_label": SCENARIO_LABEL,
        "selected_date": str(bundle.days[day_index]),
        "voltage_limits_pu": [VOLTAGE_MIN_PU, VOLTAGE_MAX_PU],
        "branch_limit_mw": BRANCH_LIMIT_MW,
        "inflexible_load_scale": INFLEXIBLE_LOAD_SCALE,
        "pv_output_scale": PV_OUTPUT_SCALE,
        "mc_samples": DEFAULT_MC_SAMPLES,
        "moment_samples": DEFAULT_MOMENT_SAMPLES,
        "seed": DEFAULT_SEED,
        "selection": {
            "rule": "fixed original reference date",
            "reference_date": str(REFERENCE_DATE),
            "uses_bounded_rationality_outcome": False,
        },
        "accuracy": {
            "directions_with_mean_error_at_most_0p01_mw": int(close_match.sum()),
            "upper_directions_with_mean_error_at_most_0p01_mw": int(
                close_match[frame["direction"] == "upper"].sum()
            ),
            "lower_directions_with_mean_error_at_most_0p01_mw": int(
                close_match[frame["direction"] == "lower"].sum()
            ),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "gurobi": ".".join(map(str, gp.gurobi.version())),
        "figure_contract": {
            "core_conclusion": f"On the original reference day under the {SCENARIO_LABEL} setting, bounded rationality changes both root-power boundaries and the analytical and Monte Carlo means agree within 0.01 MW in {int(close_match.sum())}/48 directions; remaining discrepancies identify network-limit regime transitions.",
            "archetype": "quantitative grid",
            "panels": {"single": "upper and lower root-power boundaries on one shared axis"},
            "backend": "Python/matplotlib only",
            "export": ["tightly cropped PDF"],
        },
    }
    write_json(OUTPUT_DIR / "metadata.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("market", "relaxed"), default="market")
    parser.add_argument("--phase", choices=("fit", "preflight", "run", "plot", "all"), default="all")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--moment-samples", type=int, default=DEFAULT_MOMENT_SAMPLES)
    parser.add_argument("--stage1-iterations", type=int, default=DEFAULT_STAGE1_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_scenario(args.scenario)
    if args.phase == "plot":
        make_figure()
        print("Completed network-level plot from registered source data.", flush=True)
        return
    if args.mc_samples != DEFAULT_MC_SAMPLES:
        raise ValueError(f"This registered run requires exactly {DEFAULT_MC_SAMPLES} Monte Carlo samples.")
    if args.moment_samples != DEFAULT_MOMENT_SAMPLES:
        raise ValueError(f"This registered run requires exactly {DEFAULT_MOMENT_SAMPLES} moment samples.")
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archives = load_node_archives()
    bundle = fit_stage1_models(archives, args.stage1_iterations, force=args.force and args.phase == "fit")
    if args.phase == "fit":
        return
    network, base_p, base_q = load_network_profiles(bundle.days)
    matches = np.flatnonzero(bundle.days == REFERENCE_DATE)
    if len(matches) != 1:
        raise ValueError(f"Reference date {REFERENCE_DATE} is not uniquely available.")
    day_index = int(matches[0])
    print(f"Using fixed original reference day: {bundle.days[day_index]}", flush=True)
    if args.phase == "preflight":
        return
    if args.phase in ("run", "all"):
        run_directions(
            archives,
            network,
            bundle,
            base_p,
            base_q,
            day_index,
            args.mc_samples,
            args.moment_samples,
            args.seed,
            force=args.force,
        )
    frame = compile_results(bundle.days[day_index])
    if args.phase in ("plot", "all"):
        make_figure()
    reported_runtime = float(time.perf_counter() - start)
    build_report(frame, bundle, day_index, reported_runtime)
    print(f"Completed network-level Figure 1 workflow in {time.perf_counter() - start:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
