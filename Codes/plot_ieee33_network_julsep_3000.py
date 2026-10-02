"""Draw the active 33-kV, 3,000-user IEEE 33-bus case as a vector PDF."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


PAGE_WIDTH_MM = 88.0
PAGE_HEIGHT_MM = 50.0
PV_BUSES = {5, 11, 18, 22, 27, 31}
USER_COLOR = HexColor("#8FB8E5")
LINE_COLOR = HexColor("#111111")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bus_positions() -> dict[int, tuple[float, float]]:
    x0 = 5.5 * mm
    x1 = 11.0 * mm
    step = (84.0 * mm - x1) / 16.0
    main_y = 25.0 * mm
    positions = {0: (x0, main_y)}
    for bus in range(1, 18):
        positions[bus] = (x1 + (bus - 1) * step, main_y)
    top_y = 36.0 * mm
    for offset, bus in enumerate(range(18, 22)):
        positions[bus] = (positions[1][0] + offset * step, top_y)
    bottom_y = 13.0 * mm
    for offset, bus in enumerate(range(22, 25)):
        positions[bus] = (positions[2][0] + offset * step, bottom_y)
    for offset, bus in enumerate(range(25, 33)):
        positions[bus] = (positions[5][0] + offset * step, bottom_y)
    return positions


def draw_triangle(pdf: canvas.Canvas, x: float, y: float, radius: float) -> None:
    path = pdf.beginPath()
    path.moveTo(x, y + radius)
    path.lineTo(x - 0.90 * radius, y - 0.75 * radius)
    path.lineTo(x + 0.90 * radius, y - 0.75 * radius)
    path.close()
    pdf.setFillColor(USER_COLOR)
    pdf.setStrokeColor(LINE_COLOR)
    pdf.drawPath(path, stroke=1, fill=1)


def draw_network(input_dir: Path, output_path: Path) -> None:
    assignments = read_rows(input_dir / "node_assignments.csv")
    network = read_rows(input_dir / "network_33kv.csv")
    counts = {
        int(row["bus"]): int(row["users"])
        for row in assignments
        if row["node_type"].strip().lower() == "user"
    }
    if sum(counts.values()) != 3_000 or set(PV_BUSES).intersection(counts):
        raise ValueError("The input is not the active 3,000-user, PV-only-node case.")
    positions = bus_positions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(
        str(output_path), pagesize=(PAGE_WIDTH_MM * mm, PAGE_HEIGHT_MM * mm)
    )
    pdf.setTitle("IEEE 33-bus distribution system with TCL users")

    pdf.setStrokeColor(LINE_COLOR)
    pdf.setLineWidth(0.9)
    for branch in network:
        start = positions[int(branch["from_bus"])]
        stop = positions[int(branch["to_bus"])]
        pdf.line(start[0], start[1], stop[0], stop[1])

    radius = 1.25 * mm
    for bus in range(33):
        x, y = positions[bus]
        if bus in PV_BUSES:
            draw_triangle(pdf, x, y, radius)
        else:
            pdf.setFillColor(white if bus == 0 else USER_COLOR)
            pdf.setStrokeColor(LINE_COLOR)
            pdf.circle(x, y, radius, stroke=1, fill=1)

        pdf.setFillColor(LINE_COLOR)
        pdf.setFont("Helvetica-Bold", 6.5)
        if 18 <= bus <= 21:
            pdf.drawCentredString(x, y + 2.2 * mm, str(bus))
        elif 22 <= bus <= 32:
            pdf.drawCentredString(x, y - 3.4 * mm, str(bus))
        else:
            pdf.drawCentredString(x, y + 2.2 * mm, str(bus))

    legend_x = 50.0 * mm
    legend_y = 40.0 * mm
    legend_w = 35.0 * mm
    legend_h = 8.0 * mm
    pdf.setFillColor(white)
    pdf.setStrokeColor(LINE_COLOR)
    pdf.roundRect(legend_x, legend_y, legend_w, legend_h, 1.5 * mm, stroke=1, fill=1)
    pdf.setFillColor(LINE_COLOR)
    pdf.setFont("Helvetica", 6.8)
    pdf.drawString(legend_x + 2.0 * mm, legend_y + 5.2 * mm, "PV bus")
    draw_triangle(pdf, legend_x + 14.0 * mm, legend_y + 5.5 * mm, 1.15 * mm)
    pdf.setFillColor(LINE_COLOR)
    pdf.drawString(legend_x + 18.0 * mm, legend_y + 5.2 * mm, "User bus")
    pdf.setFillColor(USER_COLOR)
    pdf.circle(legend_x + 30.5 * mm, legend_y + 5.5 * mm, 1.15 * mm, stroke=1, fill=1)
    pdf.setFillColor(LINE_COLOR)
    pdf.drawString(
        legend_x + 2.0 * mm,
        legend_y + 1.8 * mm,
        f"Users per user bus: {min(counts.values())}-{max(counts.values())}",
    )

    pdf.showPage()
    pdf.save()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    draw_network(
        project / "Data" / "Inputs_33kV_25MW_3000_JulSep",
        project / "Figures" / "IEEE-bus.pdf",
    )
    print(project / "Figures" / "IEEE-bus.pdf")


if __name__ == "__main__":
    main()
