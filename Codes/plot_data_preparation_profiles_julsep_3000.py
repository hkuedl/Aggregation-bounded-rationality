"""Plot per-unit input profiles for the active July--September case.

The script creates two tightly cropped vector PDFs sized to one journal column. The
first uses left and right y axes for PV and non-flexible demand; the second
shows the equivalent TCL thermal disturbance. A representative day is chosen
by proximity to the median daily shapes of all three series.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PV_BUSES = (5, 11, 18, 22, 27, 31)
N_DAYS = 92
HOURS_PER_DAY = 24
N_HOURS = N_DAYS * HOURS_PER_DAY
START_DATE = datetime(2025, 7, 1)
FIGURE_WIDTH_MM = 86.0
FIGURE_HEIGHT_MM = 53.0

COLOR_PV = HexColor("#D28B00")
COLOR_LOAD = HexColor("#0F4D92")
COLOR_DISTURBANCE = HexColor("#B64342")
COLOR_AXIS = HexColor("#272727")


def load_pv_profile(input_dir: Path) -> np.ndarray:
    profiles = []
    for bus in PV_BUSES:
        values = np.load(
            input_dir / f"node_{bus:02d}.npy", allow_pickle=False
        ).astype(np.float64)
        if values.shape != (N_HOURS,) or values.dtype.names is not None:
            raise ValueError(f"Node {bus} is not a valid pure-PV profile.")
        peak = float(values.max())
        if peak <= 0.0:
            raise ValueError(f"Node {bus} has no positive PV output.")
        profiles.append(values / peak)
    return np.median(np.vstack(profiles), axis=0)


def load_demand_profiles(
    input_dir: Path, demand_bus: int
) -> tuple[np.ndarray, np.ndarray]:
    if demand_bus in PV_BUSES:
        raise ValueError("The representative demand bus cannot be PV-only.")
    users = np.load(
        input_dir / f"node_{demand_bus:02d}.npy",
        allow_pickle=False,
        mmap_mode="r",
    )
    required = {"inflexible_power_kw", "disturbance_c"}
    if users.dtype.names is None or not required.issubset(users.dtype.names):
        raise ValueError(f"Node {demand_bus} is not a valid user array.")
    if users["inflexible_power_kw"].shape[1] != N_HOURS:
        raise ValueError("Unexpected non-flexible profile length.")
    nonflexible = users["inflexible_power_kw"].sum(axis=0).astype(np.float64)
    disturbance = np.median(users["disturbance_c"], axis=0).astype(np.float64)
    return nonflexible, disturbance


def daily_per_unit(values: np.ndarray) -> np.ndarray:
    daily = np.asarray(values, dtype=np.float64).reshape(N_DAYS, HOURS_PER_DAY)
    peaks = np.max(np.abs(daily), axis=1, keepdims=True)
    if np.any(peaks <= 0.0):
        raise ValueError("A daily profile has zero amplitude.")
    return daily / peaks


def select_representative_day(
    pv: np.ndarray, nonflexible: np.ndarray, disturbance: np.ndarray
) -> int:
    combined_score = np.zeros(N_DAYS, dtype=np.float64)
    for values in (pv, nonflexible, disturbance):
        daily = daily_per_unit(values)
        median_shape = np.median(daily, axis=0)
        combined_score += np.sqrt(np.mean((daily - median_shape) ** 2, axis=1))
    return int(np.argmin(combined_score))


def selected_day_per_unit(values: np.ndarray, day_index: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64).reshape(
        N_DAYS, HOURS_PER_DAY
    )[day_index]
    scale = float(np.max(np.abs(selected)))
    if scale <= 0.0:
        raise ValueError("The selected profile has zero amplitude.")
    return selected / scale


def map_point(
    hour: float,
    value: float,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    x = left + (hour - 1.0) / 23.0 * width
    y = bottom + value / 1.05 * height
    return x, y


def draw_axes(
    pdf: canvas.Canvas,
    left_label: str,
    left_color,
    right_label: str | None = None,
    right_color=None,
) -> tuple[float, float, float, float]:
    page_width = FIGURE_WIDTH_MM * mm
    page_height = FIGURE_HEIGHT_MM * mm
    left = 15.5 * mm
    right_margin = 13.5 * mm if right_label else 3.0 * mm
    bottom = 12.0 * mm
    top = 5.0 * mm
    width = page_width - left - right_margin
    height = page_height - bottom - top

    pdf.setStrokeColor(COLOR_AXIS)
    pdf.setFillColor(COLOR_AXIS)
    pdf.setLineWidth(0.8)
    pdf.line(left, bottom, left + width, bottom)
    pdf.line(left, bottom, left, bottom + height)
    if right_label:
        pdf.setStrokeColor(right_color)
        pdf.line(left + width, bottom, left + width, bottom + height)

    pdf.setFont("Helvetica", 9.0)
    for hour in (1, 6, 12, 18, 24):
        x, _ = map_point(hour, 0.0, left, bottom, width, height)
        pdf.setStrokeColor(COLOR_AXIS)
        pdf.line(x, bottom, x, bottom - 1.2 * mm)
        pdf.setFillColor(COLOR_AXIS)
        pdf.drawCentredString(x, bottom - 4.0 * mm, str(hour))
    for value in (0.0, 0.25, 0.50, 0.75, 1.00):
        _, y = map_point(1.0, value, left, bottom, width, height)
        pdf.setStrokeColor(left_color)
        pdf.line(left - 1.2 * mm, y, left, y)
        pdf.setFillColor(left_color)
        pdf.drawRightString(left - 1.8 * mm, y - 0.9 * mm, f"{value:.2g}")
        if right_label:
            pdf.setStrokeColor(right_color)
            pdf.line(left + width, y, left + width + 1.2 * mm, y)
            pdf.setFillColor(right_color)
            pdf.drawString(left + width + 1.8 * mm, y - 0.9 * mm, f"{value:.2g}")

    pdf.setFillColor(COLOR_AXIS)
    pdf.setFont("Helvetica", 10.0)
    pdf.drawCentredString(left + width / 2.0, 3.0 * mm, "Hour of day")
    pdf.saveState()
    pdf.setFillColor(left_color)
    pdf.translate(7.0 * mm, bottom + height / 2.0)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, left_label)
    pdf.restoreState()
    if right_label:
        pdf.saveState()
        pdf.setFillColor(right_color)
        pdf.translate(page_width - 3.2 * mm, bottom + height / 2.0)
        pdf.rotate(90)
        pdf.drawCentredString(0, 0, right_label)
        pdf.restoreState()
    return left, bottom, width, height


def draw_series(
    pdf: canvas.Canvas,
    values: np.ndarray,
    color,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    path = pdf.beginPath()
    for index, value in enumerate(values):
        x, y = map_point(index + 1, float(value), left, bottom, width, height)
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    pdf.setStrokeColor(color)
    pdf.setLineWidth(1.4)
    pdf.drawPath(path, stroke=1, fill=0)


def plot_pv_and_nonflexible(
    pv: np.ndarray, nonflexible: np.ndarray, output_path: Path
) -> None:
    page_size = (FIGURE_WIDTH_MM * mm, FIGURE_HEIGHT_MM * mm)
    pdf = canvas.Canvas(str(output_path), pagesize=page_size)
    pdf.setTitle("Typical PV and non-flexible load profiles")
    left, bottom, width, height = draw_axes(
        pdf,
        "PV output (p.u.)",
        COLOR_PV,
        "Non-flexible load (p.u.)",
        COLOR_LOAD,
    )
    draw_series(pdf, pv, COLOR_PV, left, bottom, width, height)
    draw_series(pdf, nonflexible, COLOR_LOAD, left, bottom, width, height)
    # Keep the legend in the reserved top margin so it cannot cover either
    # profile, including days whose normalized peak is close to 1 p.u.
    legend_y = bottom + height + 2.3 * mm
    pdf.setFont("Helvetica", 8.5)
    pdf.setStrokeColor(COLOR_PV)
    pdf.line(left + 2 * mm, legend_y, left + 8 * mm, legend_y)
    pdf.setFillColor(COLOR_PV)
    pdf.drawString(left + 9 * mm, legend_y - 0.9 * mm, "PV")
    pdf.setStrokeColor(COLOR_LOAD)
    pdf.line(left + 22 * mm, legend_y, left + 28 * mm, legend_y)
    pdf.setFillColor(COLOR_LOAD)
    pdf.drawString(left + 29 * mm, legend_y - 0.9 * mm, "Non-flexible load")
    pdf.showPage()
    pdf.save()


def plot_disturbance(disturbance: np.ndarray, output_path: Path) -> None:
    page_size = (FIGURE_WIDTH_MM * mm, FIGURE_HEIGHT_MM * mm)
    pdf = canvas.Canvas(str(output_path), pagesize=page_size)
    pdf.setTitle("Typical equivalent thermal disturbance")
    left, bottom, width, height = draw_axes(
        pdf, "Equivalent disturbance (p.u.)", COLOR_AXIS
    )
    draw_series(pdf, disturbance, COLOR_DISTURBANCE, left, bottom, width, height)
    pdf.showPage()
    pdf.save()


def write_source_data(
    path: Path,
    selected_date: str,
    pv: np.ndarray,
    nonflexible: np.ndarray,
    disturbance: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "date",
                "hour",
                "pv_pu",
                "nonflexible_load_pu",
                "equivalent_disturbance_pu",
            ],
        )
        writer.writeheader()
        for index in range(HOURS_PER_DAY):
            writer.writerow(
                {
                    "date": selected_date,
                    "hour": index + 1,
                    "pv_pu": float(pv[index]),
                    "nonflexible_load_pu": float(nonflexible[index]),
                    "equivalent_disturbance_pu": float(disturbance[index]),
                }
            )


def create_figures(input_dir: Path, output_dir: Path, demand_bus: int) -> tuple[Path, Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pv = load_pv_profile(input_dir)
    nonflexible, disturbance = load_demand_profiles(input_dir, demand_bus)
    day_index = select_representative_day(pv, nonflexible, disturbance)
    selected_date = (START_DATE + timedelta(days=day_index)).strftime("%Y-%m-%d")
    pv_pu = selected_day_per_unit(pv, day_index)
    nonflexible_pu = selected_day_per_unit(nonflexible, day_index)
    disturbance_pu = selected_day_per_unit(disturbance, day_index)

    first = output_dir / "typical_pv_and_nonflexible_profiles.pdf"
    second = output_dir / "typical_equivalent_thermal_disturbance.pdf"
    plot_pv_and_nonflexible(pv_pu, nonflexible_pu, first)
    plot_disturbance(disturbance_pu, second)
    write_source_data(
        output_dir / "typical_profiles_source_data.csv",
        selected_date,
        pv_pu,
        nonflexible_pu,
        disturbance_pu,
    )
    return first, second, selected_date


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project / "Data" / "Inputs_33kV_25MW_3000_JulSep",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project / "Figures"
    )
    parser.add_argument("--demand-bus", type=int, default=29)
    args = parser.parse_args()
    first, second, selected_date = create_figures(
        args.input_dir.resolve(), args.output_dir.resolve(), args.demand_bus
    )
    print(f"Representative day: {selected_date}")
    print(f"Saved: {first}")
    print(f"Saved: {second}")


if __name__ == "__main__":
    main()
