"""Run the registered case-study workflow from downloaded data to all figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
FIGURE_DIR = PROJECT_ROOT / "Figures"
TEMPLATE_SHA256 = "79b3ca3cd5dcfec72ad5781f8f6acefd28ff9610a7381acb93f9f0b9a510616a"
INITIALIZATION_SHA256 = "29d007f994b3fb1ae9c3ef613714d9dfa7ebd226a17180006310aa2f7a9fb0d0"
USER_BUSES = tuple(bus for bus in range(1, 33) if bus not in {5, 11, 18, 22, 27, 31})


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    command: tuple[str, ...] | None
    expected_outputs: tuple[Path, ...]
    action: Callable[[], None] | None = None


def validate_data() -> None:
    """Fail early when the downloaded Google Drive folder is incomplete."""

    case_dir = DATA_DIR / "Inputs_33kV_25MW_3000_JulSep"
    required = [
        case_dir / "metadata.json",
        case_dir / "network_33kv.csv",
        case_dir / "node_assignments.csv",
        case_dir / "nyiso_capitl_dam_lbmp.csv",
        DATA_DIR / "ecobee" / "processed" / "template_conditional_pmf.npz",
        DATA_DIR
        / "ecobee"
        / "processed"
        / "registered_node_modeling_initialization.csv",
    ]
    required.extend(case_dir / f"node_{bus:02d}.npy" for bus in range(1, 33))
    missing = [path for path in required if not path.is_file()]

    real_time_dir = DATA_DIR / "NYISO Price" / "RTLBMP_202509"
    real_time_files = sorted(real_time_dir.glob("*rtlbmp_zone.csv"))
    if len(real_time_files) != 30:
        missing.append(real_time_dir / "<30 daily real-time price CSV files>")

    if missing:
        rendered = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in missing)
        raise FileNotFoundError(
            "The downloaded Data folder is incomplete. Missing required inputs:\n"
            + rendered
        )

    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    period = metadata.get("period", {})
    if metadata.get("total_users") != 3_000 or period.get("hours") != 2_208:
        raise ValueError(
            "The downloaded case is not the registered 3,000-user, 2,208-hour case."
        )
    template_path = DATA_DIR / "ecobee" / "processed" / "template_conditional_pmf.npz"
    digest = hashlib.sha256(template_path.read_bytes()).hexdigest()
    if digest != TEMPLATE_SHA256:
        raise ValueError(
            "The processed ecobee PMF does not match the registered SHA-256 digest."
        )
    initialization_path = (
        DATA_DIR
        / "ecobee"
        / "processed"
        / "registered_node_modeling_initialization.csv"
    )
    initialization_digest = hashlib.sha256(initialization_path.read_bytes()).hexdigest()
    if initialization_digest != INITIALIZATION_SHA256:
        raise ValueError(
            "The node-model initialization does not match the registered SHA-256 digest."
        )
    print("Data validation passed.", flush=True)


def python_command(script: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, str(PROJECT_ROOT / "Codes" / script), *arguments)


STEPS = (
    Step("validate-data", "Validate the downloaded data layout.", None, (), validate_data),
    Step(
        "plot-network",
        "Draw the IEEE 33-bus system.",
        python_command("plot_ieee33_network_julsep_3000.py"),
        (FIGURE_DIR / "IEEE-bus.pdf",),
    ),
    Step(
        "plot-inputs",
        "Draw representative PV, load, and disturbance profiles.",
        python_command("plot_data_preparation_profiles_julsep_3000.py"),
        (
            FIGURE_DIR / "typical_pv_and_nonflexible_profiles.pdf",
            FIGURE_DIR / "typical_equivalent_thermal_disturbance.pdf",
        ),
    ),
    Step(
        "optimize-users",
        "Solve all individual cooling-only optimal-response problems.",
        python_command("optimize_cooling_only_julsep_3000.py"),
        (
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "Optimal results"
            / "metadata.json",
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "Optimal results"
            / "node_optimization_summary.csv",
        )
        + tuple(
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "Optimal results"
            / f"node_{bus:02d}_optimal_results.npz"
            for bus in USER_BUSES
        ),
    ),
    Step(
        "prepare-behavior",
        "Expand the empirical bounded-rationality PMF to all users.",
        python_command("prepare_bounded_rationality_julsep_3000.py"),
        (
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "bounded_rationality_metadata.json",
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "user_distribution_summary.csv",
            FIGURE_DIR / "bounded_rationality_temperature_distribution.pdf",
        )
        + tuple(
            OUTPUT_DIR
            / "Bounded Rationality Jul-Sep 3000"
            / "user_probability_distributions"
            / f"node_{bus:02d}_bounded_rationality.npz"
            for bus in USER_BUSES
        ),
    ),
    Step(
        "node-density",
        "Run the 300-user aggregate-distribution density experiment.",
        python_command(
            "analyze_node_level_distribution_accuracy.py",
            "--analysis-users",
            "300",
            "--density-only",
            "--output-dir",
            str(OUTPUT_DIR / "Node-level Aggregate Distribution Direct Dynamics Density"),
        ),
        (
            FIGURE_DIR / "node_level_aggregate_distribution_validation_density.pdf",
            OUTPUT_DIR
            / "Node-level Aggregate Distribution Direct Dynamics Density"
            / "aggregate_mc_samples.npz",
            OUTPUT_DIR
            / "Node-level Aggregate Distribution Direct Dynamics Density"
            / "metadata.json",
        ),
    ),
    Step(
        "node-wasserstein",
        "Run the 300-to-3,000-user Wasserstein sensitivity experiment.",
        python_command(
            "analyze_node_level_distribution_accuracy.py",
            "--analysis-users",
            "3000",
            "--wasserstein-only",
            "--output-dir",
            str(OUTPUT_DIR / "Node-level Aggregate Distribution Direct Dynamics Sensitivity"),
        ),
        (
            FIGURE_DIR / "node_level_aggregate_distribution_validation_wasserstein.pdf",
            OUTPUT_DIR
            / "Node-level Aggregate Distribution Direct Dynamics Sensitivity"
            / "wasserstein_by_user_count.csv",
            OUTPUT_DIR
            / "Node-level Aggregate Distribution Direct Dynamics Sensitivity"
            / "metadata.json",
        ),
    ),
    Step(
        "prepare-node-modeling",
        "Create the common daily data for node-level model comparisons.",
        python_command(
            "analyze_node_level_modeling_impact.py", "--phase", "prepare"
        ),
        (OUTPUT_DIR / "Node-level Modeling Impact" / "aggregate_daily_data.npz",),
    ),
    Step(
        "fit-node-models",
        "Fit the proposed, No-BR, TCN, and Bi-SRU models and draw both figures.",
        python_command("run_node_level_soft_physics_benchmarks.py"),
        (
            FIGURE_DIR / "node_level_modeling_rmse_boxplot.pdf",
            FIGURE_DIR / "node_level_modeling_sensitivity_heatmap.pdf",
            OUTPUT_DIR / "Node-level Modeling Impact" / "daily_rmse.csv",
            OUTPUT_DIR / "Node-level Modeling Impact" / "rmse_summary.csv",
            OUTPUT_DIR / "Node-level Modeling Impact" / "sensitivity_rmse.csv",
            OUTPUT_DIR / "Node-level Modeling Impact" / "metadata.json",
        ),
    ),
    Step(
        "network-7.5mw",
        "Compute network flexibility boundaries with the 7.5-MW branch limit.",
        python_command(
            "analyze_network_level_flexibility.py", "--scenario", "market"
        ),
        (
            FIGURE_DIR / "network_level_flexibility_bounds_market_clearing.pdf",
            OUTPUT_DIR
            / "Network-level Flexibility Figure 1"
            / "market_clearing"
            / "metadata.json",
            OUTPUT_DIR
            / "Network-level Flexibility Figure 1"
            / "market_clearing"
            / "network_boundary_results.csv",
        ),
    ),
    Step(
        "network-15mw",
        "Compute network flexibility boundaries with the 15-MW branch limit.",
        python_command(
            "analyze_network_level_flexibility.py", "--scenario", "relaxed"
        ),
        (
            FIGURE_DIR / "network_level_flexibility_bounds_15MW.pdf",
            OUTPUT_DIR
            / "Network-level Flexibility Figure 1"
            / "relaxed_15mw"
            / "metadata.json",
            OUTPUT_DIR
            / "Network-level Flexibility Figure 1"
            / "relaxed_15mw"
            / "network_boundary_results.csv",
        ),
    ),
    Step(
        "market-clearing",
        "Run all 30 September day-ahead market-clearing cases.",
        python_command("analyze_market_clearing_result1.py"),
        (
            OUTPUT_DIR / "Market Clearing Result 1" / "daily_costs.csv",
            OUTPUT_DIR / "Market Clearing Result 1" / "market_clearing_result1.csv",
            OUTPUT_DIR / "Market Clearing Result 1" / "stage2_node_models.npz",
            OUTPUT_DIR / "Market Clearing Result 1" / "metadata.json",
        ),
    ),
    Step(
        "finalize-market-figure",
        "Apply the registered recourse valuation and draw the cost figure.",
        python_command("finalize_market_clearing_result1.py"),
        (
            OUTPUT_DIR / "Market Clearing Result 1" / "daily_costs.csv",
            OUTPUT_DIR / "Market Clearing Result 1" / "market_clearing_result1_table.csv",
            FIGURE_DIR / "market_clearing_daily_total_cost.pdf",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    names = [step.name for step in STEPS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-step", choices=names, default=names[0])
    parser.add_argument("--to-step", choices=names, default=names[-1])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a step only when all of its registered outputs already exist.",
    )
    parser.add_argument("--list-steps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_steps:
        for index, step in enumerate(STEPS, start=1):
            print(f"{index:02d}. {step.name}: {step.description}")
        return

    names = [step.name for step in STEPS]
    start = names.index(args.from_step)
    stop = names.index(args.to_step)
    if start > stop:
        raise ValueError("--from-step must not occur after --to-step.")

    for index, step in enumerate(STEPS[start : stop + 1], start=start + 1):
        print(f"\n[{index:02d}/{len(STEPS):02d}] {step.name}: {step.description}", flush=True)
        if args.resume and step.expected_outputs and all(
            path.exists() for path in step.expected_outputs
        ):
            print("Registered outputs already exist; skipping.", flush=True)
            continue
        if step.action is not None:
            step.action()
        elif step.command is not None:
            subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
        else:
            raise RuntimeError(f"Step {step.name} has no action.")


if __name__ == "__main__":
    main()
