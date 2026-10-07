"""Prepare bounded-rationality distributions for the active 3,000-user case.

The ecobee-derived conditional PMF is treated as an immutable empirical
template. This script expands that template to every user in the active
July--September case by reproducible Dirichlet sampling and exports the
single-column manuscript figure. It does not solve the TCL optimization;
that task is handled by ``optimize_cooling_only_julsep_3000.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


EXPECTED_USERS = 3_000
EXPECTED_DAYS = 92
EXPECTED_HOURS = EXPECTED_DAYS * 24
RANDOM_SEED = 20260909
DIRICHLET_CONCENTRATION = 80.0
FIGURE_WIDTH_MM = 86.0
FIGURE_HEIGHT_MM = 56.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate_case(input_dir: Path) -> tuple[list[dict[str, str]], dict]:
    assignments = read_csv(input_dir / "node_assignments.csv")
    with (input_dir / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    period = metadata.get("period", {})
    if metadata.get("total_users") != EXPECTED_USERS:
        raise ValueError("The input case is not the active 3,000-user case.")
    if period.get("days") != EXPECTED_DAYS or period.get("hours") != EXPECTED_HOURS:
        raise ValueError("The input case is not the active 92-day July--September case.")
    if metadata.get("power_convention") != "cooling only; p >= 0 and b < 0":
        raise ValueError("The input case does not use the cooling-only sign convention.")

    assigned_users = sum(
        int(row["users"])
        for row in assignments
        if row["node_type"].strip().lower() == "user"
    )
    if assigned_users != EXPECTED_USERS:
        raise ValueError(f"Assignments contain {assigned_users} users, not 3,000.")
    return assignments, metadata


def load_template(template_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(template_path, allow_pickle=False) as archive:
        required = {
            "preset_setpoint_c",
            "actual_setpoint_support_c",
            "observed_count",
            "conditional_probability",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Template is missing arrays: {sorted(missing)}")
        preset = np.asarray(archive["preset_setpoint_c"], dtype=np.float64)
        support = np.asarray(archive["actual_setpoint_support_c"], dtype=np.float64)
        count = np.asarray(archive["observed_count"], dtype=np.int32)
        probability = np.asarray(archive["conditional_probability"], dtype=np.float64)

    expected_shape = (len(preset), len(support))
    if count.shape != expected_shape or probability.shape != expected_shape:
        raise ValueError("The empirical PMF arrays have inconsistent shapes.")
    if np.any(probability < 0.0) or np.any(probability.sum(axis=1) <= 0.0):
        raise ValueError("The empirical PMF contains an invalid probability row.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-6):
        raise ValueError("Each empirical conditional PMF must sum to one.")
    return preset, support, count, probability


def expand_to_users(
    input_dir: Path,
    output_dir: Path,
    assignments: list[dict[str, str]],
    preset: np.ndarray,
    support: np.ndarray,
    template_probability: np.ndarray,
    seed: int,
    concentration: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    user_dir = output_dir / "user_probability_distributions"
    user_dir.mkdir(parents=True, exist_ok=True)
    expected_files: set[str] = set()
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    total_users = 0
    user_nodes = 0

    below = support[None, :] < preset[:, None] - 0.25
    above = support[None, :] > preset[:, None] + 0.25
    same = ~(below | above)

    for assignment in assignments:
        if assignment["node_type"].strip().lower() != "user":
            continue
        bus = int(assignment["bus"])
        source_path = input_dir / f"node_{bus:02d}.npy"
        users = np.load(source_path, allow_pickle=False, mmap_mode="r")
        if users.dtype.names is None or "user_id" not in users.dtype.names:
            raise ValueError(f"Expected a structured user array: {source_path}")
        if "disturbance_c" not in users.dtype.names:
            raise ValueError(f"Missing disturbance field: {source_path}")
        if users["disturbance_c"].shape[1] != EXPECTED_HOURS:
            raise ValueError(f"Unexpected horizon in {source_path}")

        user_ids = np.asarray(users["user_id"], dtype=np.int32)
        expected_count = int(assignment["users"])
        if len(user_ids) != expected_count:
            raise ValueError(
                f"Bus {bus}: assignment has {expected_count} users, file has {len(user_ids)}."
            )

        probability = np.empty(
            (len(user_ids), len(preset), len(support)), dtype=np.float32
        )
        for user_index in range(len(user_ids)):
            for preset_index in range(len(preset)):
                template_row = template_probability[preset_index]
                positive_support = template_row > 0.0
                sampled_row = np.zeros(len(support), dtype=np.float32)
                alpha = concentration * template_row[positive_support]
                sampled_row[positive_support] = rng.dirichlet(alpha).astype(np.float32)
                probability[user_index, preset_index] = sampled_row

        output_path = user_dir / f"node_{bus:02d}_bounded_rationality.npz"
        expected_files.add(output_path.name)
        np.savez_compressed(
            output_path,
            node_id=np.asarray(bus, dtype=np.int16),
            user_id=user_ids,
            preset_setpoint_c=preset.astype(np.float32),
            actual_setpoint_support_c=support.astype(np.float32),
            conditional_probability=probability,
            template_probability=template_probability.astype(np.float32),
            dirichlet_concentration=np.asarray(concentration, dtype=np.float32),
            random_seed=np.asarray(seed, dtype=np.int64),
        )

        for user_position, user_id in enumerate(user_ids):
            for preset_index, preset_c in enumerate(preset):
                pmf = probability[user_position, preset_index].astype(np.float64)
                mean_actual = float(np.dot(pmf, support))
                variance = float(np.dot(pmf, (support - mean_actual) ** 2))
                summary_rows.append(
                    {
                        "node_id": bus,
                        "user_id": int(user_id),
                        "preset_setpoint_c": float(preset_c),
                        "mean_actual_setpoint_c": mean_actual,
                        "std_actual_setpoint_c": math.sqrt(max(variance, 0.0)),
                        "probability_below_preset": float(
                            pmf[below[preset_index]].sum()
                        ),
                        "probability_same_bin": float(pmf[same[preset_index]].sum()),
                        "probability_above_preset": float(
                            pmf[above[preset_index]].sum()
                        ),
                    }
                )

        index_rows.append(
            {
                "node_id": bus,
                "user_count": len(user_ids),
                "source_case_file": f"Data/Inputs_33kV_25MW_3000_JulSep/{source_path.name}",
                "probability_file": (
                    f"user_probability_distributions/{output_path.name}"
                ),
                "probability_shape": str(tuple(probability.shape)),
            }
        )
        total_users += len(user_ids)
        user_nodes += 1

    for stale_path in user_dir.glob("node_*_bounded_rationality.npz"):
        if stale_path.name not in expected_files:
            stale_path.unlink()
    return summary_rows, index_rows, total_users, user_nodes


def blue_scale(value: float) -> Color:
    fraction = float(np.clip(value, 0.0, 1.0))
    dark = HexColor("#0F4C8A")
    return Color(
        1.0 - fraction * (1.0 - dark.red),
        1.0 - fraction * (1.0 - dark.green),
        1.0 - fraction * (1.0 - dark.blue),
    )


def plot_template(
    preset: np.ndarray,
    support: np.ndarray,
    probability: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The original fixed 86 x 56 mm page left a visible white frame after the
    # figure was scaled in LaTeX.  Define the occupied drawing bounds once and
    # translate them onto a content-sized page, which is the ReportLab analogue
    # of Matplotlib's ``bbox_inches="tight"`` export.
    content_left = 3.5 * mm
    content_bottom = 2.5 * mm
    content_right = 84.2 * mm
    content_top = 53.5 * mm
    tight_pad = 0.8 * mm
    page_width = content_right - content_left + 2.0 * tight_pad
    page_height = content_top - content_bottom + 2.0 * tight_pad
    pdf = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    pdf.setTitle("Empirical conditional distribution of cooling setpoints")
    pdf.translate(tight_pad - content_left, tight_pad - content_bottom)

    left = 17.5 * mm
    bottom = 13.0 * mm
    plot_width = 47.0 * mm
    plot_height = 40.0 * mm
    cell_width = plot_width / len(preset)
    cell_height = plot_height / len(support)
    maximum = float(probability.max())

    for x_index in range(len(preset)):
        for y_index in range(len(support)):
            pdf.setFillColor(blue_scale(probability[x_index, y_index] / maximum))
            pdf.setStrokeColor(blue_scale(probability[x_index, y_index] / maximum))
            pdf.rect(
                left + x_index * cell_width,
                bottom + y_index * cell_height,
                cell_width,
                cell_height,
                stroke=0,
                fill=1,
            )

    axis_color = HexColor("#2B2B2B")
    pdf.setStrokeColor(axis_color)
    pdf.setLineWidth(0.8)
    pdf.rect(left, bottom, plot_width, plot_height, stroke=1, fill=0)
    pdf.setFillColor(axis_color)
    pdf.setFont("Helvetica", 8.5)
    for index, value in enumerate(preset):
        # Place the first and last labels directly at the two x-axis ends.
        x = left + index / (len(preset) - 1) * plot_width
        pdf.line(x, bottom, x, bottom - 1.3 * mm)
        pdf.drawCentredString(x, bottom - 4.0 * mm, f"{value:.1f}")
    y_tick_values = np.arange(20.0, 28.1, 2.0)
    for index, value in enumerate(y_tick_values):
        # Likewise, label the y axis from its lower boundary to its upper one.
        y = bottom + index / (len(y_tick_values) - 1) * plot_height
        pdf.line(left - 1.3 * mm, y, left, y)
        pdf.drawRightString(left - 2.0 * mm, y - 1.0 * mm, f"{value:.0f}")

    pdf.setFont("Helvetica", 9.0)
    pdf.drawCentredString(
        left + plot_width / 2.0, 4.0 * mm, "Preset cooling setpoint (°C)"
    )
    pdf.saveState()
    # Keep the vertical title close to the 20--28 tick labels while retaining
    # a narrow printable gap.
    pdf.translate(7.0 * mm, bottom + plot_height / 2.0)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Actual cooling setpoint (°C)")
    pdf.restoreState()

    x_low = left
    x_high = left + plot_width
    y_axis_min = float(y_tick_values[0])
    y_axis_max = float(y_tick_values[-1])
    y_low = bottom + (preset[0] - y_axis_min) / (y_axis_max - y_axis_min) * plot_height
    y_high = bottom + (preset[-1] - y_axis_min) / (y_axis_max - y_axis_min) * plot_height
    pdf.setStrokeColor(axis_color)
    pdf.setDash(3, 2)
    pdf.line(x_low, y_low, x_high, y_high)
    pdf.setDash()

    colorbar_x = left + plot_width + 5.0 * mm
    colorbar_y = bottom
    colorbar_width = 3.2 * mm
    strips = 60
    for index in range(strips):
        fraction = index / (strips - 1)
        pdf.setFillColor(blue_scale(fraction))
        pdf.rect(
            colorbar_x,
            colorbar_y + index * plot_height / strips,
            colorbar_width,
            plot_height / strips + 0.1,
            stroke=0,
            fill=1,
        )
    pdf.setStrokeColor(axis_color)
    pdf.rect(colorbar_x, colorbar_y, colorbar_width, plot_height, stroke=1, fill=0)
    pdf.setFillColor(axis_color)
    pdf.setFont("Helvetica", 7.5)
    for fraction in (0.0, 0.5, 1.0):
        y = colorbar_y + fraction * plot_height
        pdf.line(colorbar_x + colorbar_width, y, colorbar_x + colorbar_width + 1.0 * mm, y)
        pdf.drawString(
            colorbar_x + colorbar_width + 1.5 * mm,
            y - 0.9 * mm,
            f"{fraction * maximum:.2f}",
        )
    pdf.saveState()
    # Place the colorbar title beside its tick labels instead of against the
    # outer page edge.
    pdf.translate(colorbar_x + colorbar_width + 8.3 * mm, colorbar_y + plot_height / 2.0)
    pdf.rotate(90)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawCentredString(0, 0, "Conditional probability")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()


def prepare(
    input_dir: Path,
    template_path: Path,
    output_dir: Path,
    figure_path: Path,
    seed: int,
    concentration: float,
) -> None:
    assignments, input_metadata = load_and_validate_case(input_dir)
    preset, support, observed_count, template_probability = load_template(
        template_path
    )
    summary_rows, index_rows, total_users, user_nodes = expand_to_users(
        input_dir=input_dir,
        output_dir=output_dir,
        assignments=assignments,
        preset=preset,
        support=support,
        template_probability=template_probability,
        seed=seed,
        concentration=concentration,
    )
    if total_users != EXPECTED_USERS:
        raise RuntimeError(f"Generated {total_users} users, not 3,000.")

    write_csv(output_dir / "user_distribution_summary.csv", summary_rows)
    write_csv(output_dir / "node_distribution_index.csv", index_rows)
    plot_template(preset, support, template_probability, figure_path)

    metadata = {
        "schema_version": "2.0",
        "generator": "Codes/prepare_bounded_rationality_julsep_3000.py",
        "active_case": {
            "input_directory": "Data/Inputs_33kV_25MW_3000_JulSep",
            "period": input_metadata["period"],
            "total_users": total_users,
            "user_nodes": user_nodes,
            "pv_only_nodes": len(assignments) - user_nodes,
            "power_convention": input_metadata["power_convention"],
        },
        "empirical_template": {
            "source_dataset": "Ecobee Donate Your Data 1,000 Homes in 2017",
            "doi": "10.25584/ecobee/1854924",
            "template_file": "Data/ecobee/processed/template_conditional_pmf.npz",
            "preset_setpoint_c": preset.tolist(),
            "actual_setpoint_support_c": support.tolist(),
            "observed_event_count": int(observed_count.sum()),
        },
        "user_expansion": {
            "method": "Dirichlet(kappa * empirical template PMF)",
            "dirichlet_concentration": concentration,
            "random_seed": seed,
        },
        "separation_of_tasks": {
            "probability_preparation": (
                "Codes/prepare_bounded_rationality_julsep_3000.py"
            ),
            "cooling_optimization": "Codes/optimize_cooling_only_julsep_3000.py",
            "optimal_results": "Optimal results",
        },
        "figure": "Figures/bounded_rationality_temperature_distribution.pdf",
    }
    with (output_dir / "bounded_rationality_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Prepared {total_users} user PMFs across {user_nodes} user nodes.")
    print(f"Saved bounded-rationality data: {output_dir}")
    print(f"Saved figure: {figure_path}")


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
        default=project / "Outputs" / "Bounded Rationality Jul-Sep 3000",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=project / "Data" / "ecobee" / "processed" / "template_conditional_pmf.npz",
        help="Processed ecobee conditional PMF distributed with the Data folder.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=project / "Figures" / "bounded_rationality_temperature_distribution.pdf",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--concentration", type=float, default=DIRICHLET_CONCENTRATION
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the tightly cropped PDF from the registered template only.",
    )
    args = parser.parse_args()
    if args.concentration <= 0.0:
        parser.error("--concentration must be positive")
    figure_path = args.figure_path.resolve()
    if args.plot_only:
        template_path = args.template_path.resolve()
        preset, support, _, template_probability = load_template(template_path)
        plot_template(preset, support, template_probability, figure_path)
        print(f"Saved tightly cropped figure: {figure_path}")
        return
    prepare(
        input_dir=args.input_dir.resolve(),
        template_path=args.template_path.resolve(),
        output_dir=args.output_dir.resolve(),
        figure_path=figure_path,
        seed=args.seed,
        concentration=args.concentration,
    )


if __name__ == "__main__":
    main()
