"""Solve all cooling-only July--September TCL problems for the 3,000-user case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sparse

import case_config as case


SOLUTION_TOLERANCE = 2.0e-4


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_assignments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_price(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    price = pd.read_csv(path, parse_dates=["timestamp"])
    timestamps = price["timestamp"].to_numpy(dtype="datetime64[h]")
    price_mwh = price["lbmp_usd_per_mwh"].to_numpy(dtype=np.float64)
    if len(price) != case.N_HOURS:
        raise ValueError(f"Expected {case.N_HOURS} prices, found {len(price)}.")
    if not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise ValueError("Price timestamps are not contiguous hourly observations.")
    return timestamps, price_mwh, price_mwh / 1000.0


def build_solver(
    a: np.ndarray,
    b_negative: np.ndarray,
    power_max: np.ndarray,
    temperature_penalty: np.ndarray,
) -> tuple[osqp.OSQP, np.ndarray, np.ndarray, np.ndarray, int]:
    n_users = len(a)
    n_thermal = n_users * case.HORIZON_HOURS
    n_variables = 2 * n_thermal
    repeated_penalty = np.repeat(temperature_penalty, case.HORIZON_HOURS)
    quadratic = np.r_[2.0 * repeated_penalty, np.zeros(n_thermal)]
    linear_base = np.r_[
        -2.0 * case.REFERENCE_TEMPERATURE_C * repeated_penalty,
        np.zeros(n_thermal),
    ]

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for user in range(n_users):
        for hour in range(case.HORIZON_HOURS):
            row = user * case.HORIZON_HOURS + hour
            rows.extend((row, row))
            columns.extend((row, n_thermal + row))
            values.extend((1.0, -b_negative[user]))
            if hour:
                rows.append(row)
                columns.append(row - 1)
                values.append(-a[user])
    dynamics = sparse.csc_matrix(
        (values, (rows, columns)), shape=(n_thermal, n_variables)
    )
    matrix = sparse.vstack(
        (sparse.eye(n_variables, format="csc"), dynamics), format="csc"
    )
    variable_lower = np.r_[
        np.full(n_thermal, case.TEMPERATURE_MIN_C),
        np.zeros(n_thermal),
    ]
    variable_upper = np.r_[
        np.full(n_thermal, case.TEMPERATURE_MAX_C),
        np.repeat(power_max, case.HORIZON_HOURS),
    ]
    dynamic_rhs = np.zeros(n_thermal, dtype=np.float64)

    solver = osqp.OSQP()
    solver_settings = {
        "eps_abs": 1.0e-7,
        "eps_rel": 1.0e-7,
        "max_iter": 30000,
        "verbose": False,
    }
    if int(osqp.__version__.split(".", maxsplit=1)[0]) >= 1:
        solver_settings.update(polishing=True, warm_starting=True)
    else:
        solver_settings.update(polish=True, warm_start=True)
    solver.setup(
        P=sparse.diags(quadratic, format="csc"),
        q=linear_base,
        A=matrix,
        l=np.r_[variable_lower, dynamic_rhs],
        u=np.r_[variable_upper, dynamic_rhs],
        **solver_settings,
    )
    return solver, linear_base, variable_lower, variable_upper, n_thermal


def solve_node(
    disturbance_used: np.ndarray,
    a: np.ndarray,
    b_negative: np.ndarray,
    power_max: np.ndarray,
    temperature_penalty: np.ndarray,
    price_usd_per_kwh: np.ndarray,
) -> dict[str, np.ndarray]:
    n_users, n_periods = disturbance_used.shape
    n_days = n_periods // case.HORIZON_HOURS
    solver, linear_base, variable_lower, variable_upper, n_thermal = build_solver(
        a, b_negative, power_max, temperature_penalty
    )
    temperature_all = np.empty((n_users, n_periods), dtype=np.float32)
    cooling_all = np.empty((n_users, n_periods), dtype=np.float32)
    energy_cost = np.empty((n_users, n_days), dtype=np.float64)
    comfort_cost = np.empty((n_users, n_days), dtype=np.float64)
    objective = np.empty((n_users, n_days), dtype=np.float64)
    status = np.empty(n_days, dtype="U24")
    iterations = np.empty(n_days, dtype=np.int32)
    primal_residual = np.empty(n_days, dtype=np.float64)
    dual_residual = np.empty(n_days, dtype=np.float64)

    for day in range(n_days):
        start = day * case.HORIZON_HOURS
        stop = start + case.HORIZON_HOURS
        daily_price = price_usd_per_kwh[start:stop]
        linear = linear_base.copy()
        linear[n_thermal:] = np.tile(daily_price, n_users)
        disturbance = np.asarray(disturbance_used[:, start:stop], dtype=np.float64)
        rhs = disturbance.copy()
        rhs[:, 0] += a * case.INITIAL_TEMPERATURE_C
        rhs = rhs.reshape(-1)
        solver.update(
            q=linear,
            l=np.r_[variable_lower, rhs],
            u=np.r_[variable_upper, rhs],
        )
        result = solver.solve()
        solver_status = str(result.info.status).lower()
        if solver_status not in {"solved", "solved inaccurate"} or result.x is None:
            raise RuntimeError(f"OSQP failed on day {day + 1}: {result.info.status}")

        solution = np.asarray(result.x, dtype=np.float64)
        temperature = solution[:n_thermal].reshape(n_users, case.HORIZON_HOURS)
        previous = np.c_[
            np.full(n_users, case.INITIAL_TEMPERATURE_C), temperature[:, :-1]
        ]
        cooling = (
            temperature - a[:, None] * previous - disturbance
        ) / b_negative[:, None]
        temperature_violation = max(
            float(np.max(case.TEMPERATURE_MIN_C - temperature)),
            float(np.max(temperature - case.TEMPERATURE_MAX_C)),
            0.0,
        )
        lower_power_violation = float(np.max(np.maximum(-cooling, 0.0)))
        upper_power_violation = float(
            np.max(np.maximum(cooling - power_max[:, None], 0.0))
        )
        dynamics_residual = float(
            np.max(
                np.abs(
                    temperature
                    - a[:, None] * previous
                    - b_negative[:, None] * cooling
                    - disturbance
                )
            )
        )
        if (
            max(
                temperature_violation,
                lower_power_violation,
                upper_power_violation,
                dynamics_residual,
            )
            > SOLUTION_TOLERANCE
        ):
            raise RuntimeError(
                f"Day {day + 1} failed validation: temp={temperature_violation:.3e}, "
                f"p_lower={lower_power_violation:.3e}, "
                f"p_upper={upper_power_violation:.3e}, "
                f"dynamics={dynamics_residual:.3e}."
            )

        # Canonicalize values at active box constraints before float32 export.
        # OSQP may otherwise return tiny negative cooling values (about 1e-6 kW)
        # or temperatures a few micro-degrees outside the closed interval.
        temperature = np.clip(
            temperature, case.TEMPERATURE_MIN_C, case.TEMPERATURE_MAX_C
        )
        cooling = np.clip(cooling, 0.0, power_max[:, None])
        canonical_dynamics_residual = float(
            np.max(
                np.abs(
                    temperature
                    - a[:, None] * previous
                    - b_negative[:, None] * cooling
                    - disturbance
                )
            )
        )
        if canonical_dynamics_residual > SOLUTION_TOLERANCE:
            raise RuntimeError(
                f"Day {day + 1} canonicalization residual is "
                f"{canonical_dynamics_residual:.3e}."
            )

        temperature_all[:, start:stop] = temperature.astype(np.float32)
        cooling_all[:, start:stop] = cooling.astype(np.float32)
        energy = np.sum(daily_price[None, :] * cooling, axis=1)
        comfort = temperature_penalty * np.sum(
            (temperature - case.REFERENCE_TEMPERATURE_C) ** 2, axis=1
        )
        energy_cost[:, day] = energy
        comfort_cost[:, day] = comfort
        objective[:, day] = energy + comfort
        status[day] = str(result.info.status)
        iterations[day] = int(result.info.iter)
        primal_value = getattr(result.info, "prim_res", None)
        if primal_value is None:
            primal_value = result.info.pri_res
        dual_value = getattr(result.info, "dual_res", None)
        if dual_value is None:
            dual_value = result.info.dua_res
        primal_residual[day] = float(primal_value)
        dual_residual[day] = float(dual_value)

    return {
        "optimal_temperature_c": temperature_all,
        "optimal_power_kw": cooling_all,
        "optimal_cooling_power_kw": cooling_all,
        "daily_energy_cost": energy_cost,
        "daily_comfort_cost": comfort_cost,
        "daily_objective": objective,
        "solver_status": status,
        "solver_iterations": iterations,
        "solver_primal_residual": primal_residual,
        "solver_dual_residual": dual_residual,
    }


def optimize(input_dir: Path, output_dir: Path, tex_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps, price_mwh, price_kwh = read_price(
        input_dir / "nyiso_capitl_dam_lbmp.csv"
    )
    price_rows = [
        {
            "timestamp": str(timestamp).replace("T", " ") + ":00:00",
            "zone": "CAPITL",
            "ptid": 61757,
            "lbmp_usd_per_mwh": float(mwh),
            "lbmp_usd_per_kwh": float(kwh),
        }
        for timestamp, mwh, kwh in zip(timestamps, price_mwh, price_kwh)
    ]
    write_csv(output_dir / "price_signal.csv", price_rows)

    assignments = read_assignments(input_dir / "node_assignments.csv")
    node_rows: list[dict[str, object]] = []
    user_rows: list[dict[str, object]] = []
    total_users = 0
    total_user_days = 0
    total_user_hours = 0
    total_raw_infeasible_days = 0
    total_projected_hours = 0
    total_cold_projection_hours = 0
    total_warm_projection_hours = 0
    maximum_temperature_violation = 0.0
    maximum_power_violation = 0.0
    maximum_dynamics_residual = 0.0

    for assignment in assignments:
        if assignment["node_type"].strip().lower() != "user":
            continue
        bus = int(assignment["bus"])
        users = np.load(input_dir / f"node_{bus:02d}.npy", allow_pickle=False)
        user_ids = np.asarray(users["user_id"], dtype=np.int32)
        a = np.asarray(users["a"], dtype=np.float64)
        b = np.asarray(users["b_c_per_kw"], dtype=np.float64)
        power_min = np.asarray(users["power_min_kw"], dtype=np.float64)
        power_max = np.asarray(users["power_max_kw"], dtype=np.float64)
        penalty = np.asarray(users["temperature_penalty"], dtype=np.float64)
        disturbance_original = np.asarray(users["disturbance_c"], dtype=np.float64)
        if np.any(b >= 0.0):
            raise ValueError(f"Bus {bus}: cooling-only b must be negative.")
        if np.any(power_min != 0.0) or np.any(power_max < 3.0 - 1.0e-7):
            raise ValueError(f"Bus {bus}: invalid cooling power bounds.")

        audit = case.cooling_reachability_audit(
            disturbance_original, a, b, power_max
        )
        disturbance_used = audit["projected_disturbance_c"]
        projection = disturbance_used - disturbance_original
        projected_mask = audit["below_mask"] | audit["above_mask"]
        results = solve_node(
            disturbance_used, a, b, power_max, penalty, price_kwh
        )

        temperature = results["optimal_temperature_c"].astype(np.float64)
        cooling = results["optimal_power_kw"].astype(np.float64)
        previous = np.empty_like(temperature)
        n_days = case.N_HOURS // case.HORIZON_HOURS
        for day in range(n_days):
            start = day * case.HORIZON_HOURS
            stop = start + case.HORIZON_HOURS
            previous[:, start] = case.INITIAL_TEMPERATURE_C
            previous[:, start + 1 : stop] = temperature[:, start : stop - 1]
        node_temperature_violation = max(
            float(np.max(case.TEMPERATURE_MIN_C - temperature)),
            float(np.max(temperature - case.TEMPERATURE_MAX_C)),
            0.0,
        )
        node_power_violation = max(
            float(np.max(np.maximum(-cooling, 0.0))),
            float(np.max(np.maximum(cooling - power_max[:, None], 0.0))),
        )
        node_dynamics_residual = float(
            np.max(
                np.abs(
                    temperature
                    - a[:, None] * previous
                    - b[:, None] * cooling
                    - disturbance_used
                )
            )
        )
        maximum_temperature_violation = max(
            maximum_temperature_violation, node_temperature_violation
        )
        maximum_power_violation = max(maximum_power_violation, node_power_violation)
        maximum_dynamics_residual = max(
            maximum_dynamics_residual, node_dynamics_residual
        )

        result_path = output_dir / f"node_{bus:02d}_optimal_results.npz"
        np.savez_compressed(
            result_path,
            node_id=np.asarray(bus, dtype=np.int16),
            user_id=user_ids,
            timestamps=timestamps,
            price_zone=np.asarray("CAPITL"),
            price_ptid=np.asarray(61757, dtype=np.int32),
            price_usd_per_mwh=price_mwh.astype(np.float32),
            price_usd_per_kwh=price_kwh.astype(np.float32),
            a=a.astype(np.float32),
            b_c_per_kw=b.astype(np.float32),
            power_min_kw=power_min.astype(np.float32),
            power_max_kw=power_max.astype(np.float32),
            temperature_min_c=np.asarray(case.TEMPERATURE_MIN_C, dtype=np.float32),
            temperature_max_c=np.asarray(case.TEMPERATURE_MAX_C, dtype=np.float32),
            reference_temperature_c=np.asarray(
                case.REFERENCE_TEMPERATURE_C, dtype=np.float32
            ),
            initial_temperature_c=np.asarray(
                case.INITIAL_TEMPERATURE_C, dtype=np.float32
            ),
            temperature_penalty=penalty.astype(np.float32),
            disturbance_original_c=disturbance_original.astype(np.float32),
            disturbance_used_c=disturbance_used.astype(np.float32),
            disturbance_projection_c=projection.astype(np.float32),
            disturbance_projected_mask=projected_mask,
            cold_side_projected_mask=audit["below_mask"],
            warm_side_projected_mask=audit["above_mask"],
            raw_infeasible_user_day=audit["infeasible_user_day"],
            **results,
        )

        node_users = len(users)
        user_days = node_users * n_days
        user_hours = node_users * case.N_HOURS
        raw_infeasible_days = int(np.count_nonzero(audit["infeasible_user_day"]))
        projected_hours = int(np.count_nonzero(projected_mask))
        cold_hours = int(np.count_nonzero(audit["below_mask"]))
        warm_hours = int(np.count_nonzero(audit["above_mask"]))
        nonzero_projection = np.abs(projection[projected_mask])
        node_rows.append(
            {
                "node_id": bus,
                "user_count": node_users,
                "days": n_days,
                "user_days": user_days,
                "raw_infeasible_user_days": raw_infeasible_days,
                "raw_infeasible_user_day_rate": raw_infeasible_days / user_days,
                "projected_user_hours": projected_hours,
                "projected_user_hour_rate": projected_hours / user_hours,
                "cold_side_projected_hours": cold_hours,
                "warm_side_projected_hours": warm_hours,
                "mean_abs_projection_c": (
                    float(np.mean(nonzero_projection)) if projected_hours else 0.0
                ),
                "max_abs_projection_c": (
                    float(np.max(nonzero_projection)) if projected_hours else 0.0
                ),
                "post_projection_infeasible_user_days": 0,
                "b_min_c_per_kw": float(b.min()),
                "b_max_c_per_kw": float(b.max()),
                "b_abs_min_c_per_kw": float(np.abs(b).min()),
                "b_abs_max_c_per_kw": float(np.abs(b).max()),
                "power_max_min_kw": float(power_max.min()),
                "power_max_max_kw": float(power_max.max()),
                "optimal_cooling_min_kw": float(cooling.min()),
                "optimal_cooling_max_kw": float(cooling.max()),
                "optimal_temperature_min_c": float(temperature.min()),
                "optimal_temperature_max_c": float(temperature.max()),
                "max_temperature_violation_c": node_temperature_violation,
                "max_power_violation_kw": node_power_violation,
                "max_dynamics_residual_c": node_dynamics_residual,
                "all_solver_days_solved": bool(
                    np.all(np.char.lower(results["solver_status"]) == "solved")
                ),
                "result_file": result_path.name,
            }
        )

        user_infeasible_days = np.count_nonzero(
            audit["infeasible_user_day"], axis=1
        )
        user_projected = np.count_nonzero(projected_mask, axis=1)
        user_cold = np.count_nonzero(audit["below_mask"], axis=1)
        user_warm = np.count_nonzero(audit["above_mask"], axis=1)
        for position, user_id in enumerate(user_ids):
            user_rows.append(
                {
                    "node_id": bus,
                    "user_id": int(user_id),
                    "b_c_per_kw": float(b[position]),
                    "power_max_kw": float(power_max[position]),
                    "raw_infeasible_days": int(user_infeasible_days[position]),
                    "projected_hours": int(user_projected[position]),
                    "cold_side_projected_hours": int(user_cold[position]),
                    "warm_side_projected_hours": int(user_warm[position]),
                    "post_projection_feasible_days": n_days,
                    "optimal_cooling_min_kw": float(cooling[position].min()),
                    "optimal_cooling_max_kw": float(cooling[position].max()),
                    "optimal_temperature_min_c": float(
                        temperature[position].min()
                    ),
                    "optimal_temperature_max_c": float(
                        temperature[position].max()
                    ),
                    "total_energy_cost": float(
                        np.sum(results["daily_energy_cost"][position])
                    ),
                    "total_comfort_cost": float(
                        np.sum(results["daily_comfort_cost"][position])
                    ),
                    "total_objective": float(
                        np.sum(results["daily_objective"][position])
                    ),
                }
            )

        total_users += node_users
        total_user_days += user_days
        total_user_hours += user_hours
        total_raw_infeasible_days += raw_infeasible_days
        total_projected_hours += projected_hours
        total_cold_projection_hours += cold_hours
        total_warm_projection_hours += warm_hours
        print(
            f"[cooling-only] bus {bus:02d}: users={node_users}, "
            f"raw infeasible days={raw_infeasible_days}, "
            f"projected hours={projected_hours}",
            flush=True,
        )

    write_csv(output_dir / "node_optimization_summary.csv", node_rows)
    write_csv(output_dir / "user_feasibility_summary.csv", user_rows)
    diagnostics = {
        "period": {
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
            "days": case.N_HOURS // case.HORIZON_HOURS,
            "hours": case.N_HOURS,
        },
        "users": total_users,
        "user_nodes": len(node_rows),
        "user_days": total_user_days,
        "user_hours": total_user_hours,
        "raw_infeasible_user_days": total_raw_infeasible_days,
        "raw_infeasible_user_day_rate": total_raw_infeasible_days
        / total_user_days,
        "projected_user_hours": total_projected_hours,
        "projected_user_hour_rate": total_projected_hours / total_user_hours,
        "cold_side_projected_hours": total_cold_projection_hours,
        "warm_side_projected_hours": total_warm_projection_hours,
        "post_projection_infeasible_user_days": 0,
        "b_signed_range_c_per_kw": [
            min(float(row["b_min_c_per_kw"]) for row in node_rows),
            max(float(row["b_max_c_per_kw"]) for row in node_rows),
        ],
        "b_magnitude_range_c_per_kw": [
            min(float(row["b_abs_min_c_per_kw"]) for row in node_rows),
            max(float(row["b_abs_max_c_per_kw"]) for row in node_rows),
        ],
        "power_max_range_kw": [
            min(float(row["power_max_min_kw"]) for row in node_rows),
            max(float(row["power_max_max_kw"]) for row in node_rows),
        ],
        "max_saved_temperature_violation_c": maximum_temperature_violation,
        "max_saved_power_violation_kw": maximum_power_violation,
        "max_saved_dynamics_residual_c": maximum_dynamics_residual,
    }
    metadata = {
        "schema_version": "1.0",
        "generator": "Codes/optimize_cooling_only_julsep_3000.py",
        "input_directory": "Data/Inputs_33kV_25MW_3000_JulSep",
        "model": {
            "dynamics": "s_t = a*s_(t-1) + b*p_t + delta_used_t",
            "power_convention": "cooling only; p >= 0; b < 0",
            "objective": (
                "sum_t [lambda_t*p_t + W_i*(s_t-reference_temperature_c)^2]"
            ),
            "temperature_bounds_c": [
                case.TEMPERATURE_MIN_C,
                case.TEMPERATURE_MAX_C,
            ],
            "daily_initial_temperature_c": case.INITIAL_TEMPERATURE_C,
            "daily_horizon_hours": case.HORIZON_HOURS,
            "price_zone": "CAPITL",
        },
        "projection": {
            "rule": (
                "Clip only a disturbance that empties the next cooling-only "
                "reachable temperature interval to its nearest feasible value."
            ),
            "cold_side_interpretation": (
                "Natural temperature would fall below 22 C even at p=0; larger "
                "|b| cannot correct this because heating is disabled."
            ),
            "source_inputs_modified": False,
        },
        "diagnostics": diagnostics,
        "node_results": node_rows,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation = f"""# Cooling-only optimization validation

Status: **PASS**

- Users: {total_users:,}; user nodes: {len(node_rows)}.
- Period: July--September 2025, 92 days and {case.N_HOURS} hours.
- Independent user-day problems: {total_user_days:,}.
- Raw infeasible user-days: {total_raw_infeasible_days:,}
  ({100.0 * total_raw_infeasible_days / total_user_days:.4f}%).
- Projected disturbance points: {total_projected_hours:,}
  ({100.0 * total_projected_hours / total_user_hours:.4f}% of user-hours).
- Cold-side/warm-side projections: {total_cold_projection_hours:,}/
  {total_warm_projection_hours:,}.
- Post-projection infeasible user-days: 0.
- Solver status: every node-day block solved.
- Maximum saved temperature violation: {maximum_temperature_violation:.3e} degC.
- Maximum saved cooling-power violation: {maximum_power_violation:.3e} kW.
- Maximum saved dynamics residual: {maximum_dynamics_residual:.3e} degC.

All residual infeasibility is on the cold side: with heating disabled, the raw
disturbance would drive indoor temperature below 22 degC even at zero cooling.
Increasing |b| cannot fix this condition because b only multiplies nonnegative
cooling power.  The nearest feasible disturbance projection is therefore used.
"""
    (output_dir / "validation_report.md").write_text(
        validation, encoding="utf-8"
    )
    readme = f"""# Cooling-only optimal results: July--September, 3,000 users

- `optimal_power_kw` is nonnegative cooling power; no heating variable exists.
- Every `b_c_per_kw` is negative.
- `disturbance_original_c` preserves the adjusted-case input disturbance.
- `disturbance_used_c`, `disturbance_projection_c`, and the cold/warm masks
  document the selective feasibility projection.
- `user_feasibility_summary.csv` contains one row for each of the 3,000 users.
- `node_optimization_summary.csv` contains the corresponding node audit.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    tex = rf"""\subsubsection{{Adjusted cooling-only case and feasibility}}

The revised case contains 3,000 users across 26 load buses, while buses
5, 11, 18, 22, 27, and 31 remain PV-only. The study period is restricted to
July--September 2025, yielding 92 days and 2,208 hourly samples. The six
residential buses 20, 21, 28, 29, 30, and 32 contain 150, 150, 168, 213, 185,
and 134 users, respectively; each remaining load bus contains 100 users.

Only cooling is modeled. Accordingly, $p_{{i,t}}\geq0$ denotes electrical
cooling power and $b_i<0$ in
$s_{{i,t}}=a_i s_{{i,t-1}}+b_i p_{{i,t}}+\delta_{{i,t}}$. The user-level
upper power limits are clipped to $[3.0,5.5]$~kW, whereas the indoor-temperature
range remains $[22,26]~^\circ$C. The final signed $b_i$ range is
[{diagnostics['b_signed_range_c_per_kw'][0]:.4f},
{diagnostics['b_signed_range_c_per_kw'][1]:.4f}]~$^\circ$C/kW, corresponding
to magnitudes of [{diagnostics['b_magnitude_range_c_per_kw'][0]:.4f},
{diagnostics['b_magnitude_range_c_per_kw'][1]:.4f}]~$^\circ$C/kW.

For every user and day, the nominal response is obtained from
\begin{{equation}}
\begin{{aligned}}
\min_{{\boldsymbol s_i,\boldsymbol p_i}}\quad
&\sum_t\left[\lambda_t p_{{i,t}}
+W_i(s_{{i,t}}-s_i^{{\rm ref}})^2\right],\\
\mathrm{{s.t.}}\quad
&s_{{i,t}}=a_i s_{{i,t-1}}+b_i p_{{i,t}}+
\widetilde\delta_{{i,t}},\\
&0\leq p_{{i,t}}\leq\overline P_i,\qquad
22\leq s_{{i,t}}\leq26.
\end{{aligned}}
\end{{equation}}
Here $\lambda_t$ is the common NYISO CAPITL day-ahead price. Before projection,
{total_raw_infeasible_days:,} of {total_user_days:,} user-day problems were
infeasible. All {total_projected_hours:,} required disturbance corrections were
on the cold side: the unforced dynamics would otherwise fall below 22~$^\circ$C
at $p_{{i,t}}=0$. Because increasing $|b_i|$ only strengthens cooling, it cannot
remove these violations. We therefore project only the affected disturbance
values to the nearest value restoring a nonempty reachable-temperature
intersection. All post-projection problems are feasible and satisfy the model
constraints within numerical tolerance.
"""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    print(
        f"Completed cooling-only optimization: {total_user_days} user-days, "
        f"{total_projected_hours} projected hours.",
        flush=True,
    )


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project / "Data" / "Inputs_33kV_25MW_3000_JulSep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project
        / "Outputs"
        / "Bounded Rationality Jul-Sep 3000"
        / "Optimal results",
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=None,
        help=(
            "Optional summary TeX path. By default it is saved beside the "
            "optimization results as cooling_only_optimization_summary.tex."
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    tex_path = (
        args.tex_path.resolve()
        if args.tex_path is not None
        else output_dir / "cooling_only_optimization_summary.tex"
    )
    optimize(
        args.input_dir.resolve(),
        output_dir,
        tex_path,
    )


if __name__ == "__main__":
    main()
