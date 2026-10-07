"""Finalize the registered Market Clearing Result 1 outputs and figure.

The schedules, behavioral samples, security violations, and minimum nodal TCL
shedding quantities are fixed by ``analyze_market_clearing_result1.py``.  This
script applies the manuscript's registered security-recourse valuation of
2,540 USD/MWh, refreshes the reported summaries, and draws the 30-day total-
cost comparison used in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Market Clearing Result 1"
FIGURE_PATH = PROJECT_ROOT / "Figures" / "market_clearing_daily_total_cost.pdf"
NO_BR_DEVIATION_PATH = OUTPUT_DIR / "no_br_daily_power_deviation.csv"
NO_BR_DEVIATION_THRESHOLD_PATH = (
    OUTPUT_DIR / "no_br_daily_security_deviation_threshold.csv"
)
SECURITY_RECOURSE_PRICE_USD_PER_MWH = 2540.0
HOURS_PER_DAY = 24.0
METHOD_ORDER = ["Ground truth", "Proposed", "No BR"]
DISPLAY_LABELS = {
    "Ground truth": "Ground-truth",
    "Proposed": "Proposed",
    "No BR": "No-BR",
}


def _validate_daily(frame: pd.DataFrame) -> None:
    required = {
        "date",
        "method",
        "day_ahead_energy_cost_usd",
        "balancing_cost_usd",
        "expected_tcl_shedding_mwh",
        "security_violation_probability",
        "post_recourse_violation_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_costs.csv is missing columns: {sorted(missing)}")
    counts = frame.groupby("method")["date"].nunique().to_dict()
    if counts != {method: 30 for method in METHOD_ORDER}:
        raise ValueError(f"Expected 30 days per method, found {counts}")
    if not np.isfinite(
        frame[
            [
                "day_ahead_energy_cost_usd",
                "balancing_cost_usd",
                "expected_tcl_shedding_mwh",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Cost inputs contain non-finite values")


def revalue_outputs() -> pd.DataFrame:
    daily_path = OUTPUT_DIR / "daily_costs.csv"
    daily = pd.read_csv(daily_path)
    _validate_daily(daily)

    daily["security_recourse_cost_usd"] = (
        SECURITY_RECOURSE_PRICE_USD_PER_MWH
        * daily["expected_tcl_shedding_mwh"]
    )
    daily["additional_economic_cost_usd"] = (
        daily["balancing_cost_usd"] + daily["security_recourse_cost_usd"]
    )
    daily["total_cost_usd"] = (
        daily["day_ahead_energy_cost_usd"]
        + daily["additional_economic_cost_usd"]
    )
    daily["method"] = pd.Categorical(
        daily["method"], categories=METHOD_ORDER, ordered=True
    )
    daily = daily.sort_values(["date", "method"]).reset_index(drop=True)
    daily["method"] = daily["method"].astype(str)
    daily.to_csv(daily_path, index=False, float_format="%.12g")

    summary_path = OUTPUT_DIR / "market_clearing_result1.csv"
    summary = pd.read_csv(summary_path).set_index("method")
    grouped = daily.groupby("method", sort=False)
    for method in METHOD_ORDER:
        rows = grouped.get_group(method)
        summary.loc[method, "day_ahead_energy_cost_usd_per_day"] = rows[
            "day_ahead_energy_cost_usd"
        ].mean()
        summary.loc[method, "balancing_cost_usd_per_day"] = rows[
            "balancing_cost_usd"
        ].mean()
        summary.loc[method, "security_recourse_cost_usd_per_day"] = rows[
            "security_recourse_cost_usd"
        ].mean()
        summary.loc[method, "additional_economic_cost_usd_per_day"] = rows[
            "additional_economic_cost_usd"
        ].mean()
        summary.loc[
            method, "additional_economic_cost_daily_standard_error_usd"
        ] = rows["additional_economic_cost_usd"].std(ddof=1) / np.sqrt(len(rows))
        summary.loc[method, "total_cost_usd_per_day"] = rows[
            "total_cost_usd"
        ].mean()
    ground_cost = float(summary.loc["Ground truth", "total_cost_usd_per_day"])
    summary["total_cost_gap_vs_ground_truth_percent"] = (
        100.0 * (summary["total_cost_usd_per_day"] - ground_cost) / ground_cost
    )
    summary = summary.loc[METHOD_ORDER].reset_index()
    summary.to_csv(summary_path, index=False, float_format="%.12g")

    table = pd.DataFrame(
        {
            "metric": [
                "Security violation probability (%)",
                "Day-ahead cost (USD/day)",
                "Additional economic cost (USD/day)",
                "Total cost (USD/day)",
            ],
            **{
                method: [
                    float(
                        summary.loc[
                            summary["method"] == method,
                            "security_violation_probability_percent",
                        ].iloc[0]
                    ),
                    float(
                        summary.loc[
                            summary["method"] == method,
                            "day_ahead_energy_cost_usd_per_day",
                        ].iloc[0]
                    ),
                    float(
                        summary.loc[
                            summary["method"] == method,
                            "additional_economic_cost_usd_per_day",
                        ].iloc[0]
                    ),
                    float(
                        summary.loc[
                            summary["method"] == method,
                            "total_cost_usd_per_day",
                        ].iloc[0]
                    ),
                ]
                for method in METHOD_ORDER
            },
        }
    )
    table.to_csv(
        OUTPUT_DIR / "market_clearing_result1_table.csv",
        index=False,
        float_format="%.12g",
    )

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shedding_penalty_usd_per_mwh"] = (
        SECURITY_RECOURSE_PRICE_USD_PER_MWH
    )
    metadata["security_recourse_price_source"] = (
        "NYISO 2020 State of the Market Report: deep 10-minute reserve-shortage "
        "pricing including locational adders"
    )
    metadata["finalizer"] = "Codes/finalize_market_clearing_result1.py"
    metadata["daily_cost_figure_lower_panel"] = {
        "method": "No BR",
        "source": "no_br_daily_power_deviation.csv",
        "behavioral_samples": 200,
        "statistic": (
            "mean across behavioral samples of the daily sum of absolute "
            "actual-minus-command power deviations over all users and hours, "
            "divided by 24 hours for the plotted daily mean"
        ),
        "unit": "kW",
        "plot_transformation": "total_absolute_user_power_deviation_kw / 24",
        "display": (
            "downward bars on a truncated 300-kW baseline with magnitude tick "
            "labels"
        ),
        "display_axis_minimum_kw": 300,
        "security_threshold_source": (
            "no_br_daily_security_deviation_threshold.csv"
        ),
        "security_threshold_definition": (
            "minimum observed violating sample on realized-violation days; "
            "otherwise minimum counterfactual first-violation deviation after "
            "scaling each sampled nodal-hourly deviation direction"
        ),
        "security_threshold_marker": (
            "filled diamond; 300-700 kW shown on the primary lower-panel "
            "segment and 700-1500 kW on a compressed segment"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    values = summary.set_index("method")
    proposed = values.loc["Proposed"]
    no_br = values.loc["No BR"]
    total_saving = 100.0 * (
        no_br["total_cost_usd_per_day"] - proposed["total_cost_usd_per_day"]
    ) / no_br["total_cost_usd_per_day"]
    oracle_gap = 100.0 * (
        proposed["total_cost_usd_per_day"]
        - values.loc["Ground truth", "total_cost_usd_per_day"]
    ) / values.loc["Ground truth", "total_cost_usd_per_day"]
    calibration_rounds = int(metadata["disaggregation_calibration_rounds"])
    disaggregation_method = str(metadata["disaggregation_method"])
    report = f"""# Market Clearing Result 1 validation report

## Registered evaluation

- Test period: all 30 days of September 2025; 30 x 24 expected-response hours.
- Network limits: [0.95, 1.05] p.u. and +/-7.5 MW/MVAr.
- Exogenous stress case: inflexible load x1.15; PV x0.80; TCL inputs unchanged.
- Behavioral estimator: average 200 independent draws first, then evaluate one expected network profile per method and hour.
- Proposed disaggregation: {calibration_rounds} ex-ante calibration rounds; {disaggregation_method}.
- Disaggregation rule selection: no September test outcome was used.
- Corrective action: minimum nodal TCL shedding, valued at {SECURITY_RECOURSE_PRICE_USD_PER_MWH:.0f} USD/MWh; post-recourse violation probability is zero.

## Result-1 ordering audit

- Security violation probability: Ground truth {values.loc['Ground truth', 'security_violation_probability_percent']:.4f}%, Proposed {proposed['security_violation_probability_percent']:.4f}%, No BR {no_br['security_violation_probability_percent']:.4f}%.
- Additional economic cost: Ground truth {values.loc['Ground truth', 'additional_economic_cost_usd_per_day']:.2f}, Proposed {proposed['additional_economic_cost_usd_per_day']:.2f}, No BR {no_br['additional_economic_cost_usd_per_day']:.2f} USD/day.
- Total cost: Ground truth {values.loc['Ground truth', 'total_cost_usd_per_day']:.2f}, Proposed {proposed['total_cost_usd_per_day']:.2f}, No BR {no_br['total_cost_usd_per_day']:.2f} USD/day.
- Proposed reduces mean total cost by {total_saving:.2f}% relative to No BR and remains {oracle_gap:.2f}% above the perfect-information lower bound.

Ground truth is a wait-and-see economic lower bound: the true individual BR
laws are reconditioned inside optimization until the command and anticipated
post-BR response are self-consistent.  Proposed and No BR use the same ex-post
200-draw estimator, so the reported violation denominator is 720 hours rather
than 200 x 720 sample-hours.  Result 2 is not included.
"""
    (OUTPUT_DIR / "validation_report.md").write_text(report, encoding="utf-8")
    return daily


def plot_daily_cost(daily: pd.DataFrame, preview_png: Path | None) -> None:
    pivot = daily.pivot(index="date", columns="method", values="total_cost_usd")
    pivot = pivot[METHOD_ORDER]
    if pivot.shape != (30, 3):
        raise ValueError(f"Expected a 30 x 3 plotting table, found {pivot.shape}")

    deviation = pd.read_csv(NO_BR_DEVIATION_PATH)
    required_deviation = {
        "date",
        "method",
        "total_absolute_user_power_deviation_kw",
        "behavioral_samples",
    }
    missing = required_deviation.difference(deviation.columns)
    if missing:
        raise ValueError(
            f"{NO_BR_DEVIATION_PATH.name} is missing columns: {sorted(missing)}"
        )
    deviation = deviation.loc[deviation["method"] == "No BR"].copy()
    deviation = deviation.sort_values("date")
    if len(deviation) != 30 or deviation["date"].nunique() != 30:
        raise ValueError("Expected exactly 30 No-BR daily deviation values")
    if not np.array_equal(
        pivot.index.astype(str).to_numpy(), deviation["date"].astype(str).to_numpy()
    ):
        raise ValueError("No-BR deviation dates do not align with daily costs")
    if not np.all(deviation["behavioral_samples"].to_numpy(dtype=int) == 200):
        raise ValueError("The No-BR deviation panel requires 200 behavioral draws")
    summed_deviation_values = deviation[
        "total_absolute_user_power_deviation_kw"
    ].to_numpy(dtype=float)
    if not np.isfinite(summed_deviation_values).all() or np.any(
        summed_deviation_values < 0.0
    ):
        raise ValueError("No-BR power-deviation values are invalid")
    # Convert the 24-hour user-level deviation sum to the day's mean hourly
    # aggregate absolute deviation, retaining kW as the displayed unit.
    deviation_values = summed_deviation_values / HOURS_PER_DAY

    threshold = pd.read_csv(NO_BR_DEVIATION_THRESHOLD_PATH)
    required_threshold = {
        "date",
        "method",
        "daily_mean_power_deviation_kw",
        "security_violation_threshold_kw",
        "violating_samples_at_realized_deviation",
        "behavioral_samples",
        "threshold_method",
    }
    missing = required_threshold.difference(threshold.columns)
    if missing:
        raise ValueError(
            f"{NO_BR_DEVIATION_THRESHOLD_PATH.name} is missing columns: "
            f"{sorted(missing)}"
        )
    threshold = threshold.loc[threshold["method"] == "No BR"].sort_values(
        "date"
    )
    if len(threshold) != 30 or threshold["date"].nunique() != 30:
        raise ValueError("Expected exactly 30 No-BR security thresholds")
    if not np.array_equal(
        pivot.index.astype(str).to_numpy(), threshold["date"].astype(str).to_numpy()
    ):
        raise ValueError("No-BR threshold dates do not align with daily costs")
    if not np.all(threshold["behavioral_samples"].to_numpy(dtype=int) == 200):
        raise ValueError("Security thresholds require 200 behavioral draws")
    threshold_values = threshold[
        "security_violation_threshold_kw"
    ].to_numpy(dtype=float)
    threshold_daily_mean = threshold[
        "daily_mean_power_deviation_kw"
    ].to_numpy(dtype=float)
    if not np.allclose(threshold_daily_mean, deviation_values, atol=1.0e-6):
        raise ValueError("Threshold and deviation source data do not agree")
    if not np.isfinite(threshold_values).all() or np.any(
        threshold_values <= 0.0
    ):
        raise ValueError("No-BR security thresholds are invalid")
    realized_violation_samples = threshold[
        "violating_samples_at_realized_deviation"
    ].to_numpy(dtype=int)
    if int(np.count_nonzero(realized_violation_samples)) != 11:
        raise ValueError("Expected 11 realized-violation days")
    realized_mask = realized_violation_samples > 0
    if not np.all(threshold_values[realized_mask] <= deviation_values[realized_mask]):
        raise ValueError("A realized-violation threshold lies outside its bar")
    if not np.all(threshold_values[~realized_mask] > deviation_values[~realized_mask]):
        raise ValueError("A counterfactual threshold does not exceed its safe bar")

    styles = {
        "Ground truth": {"color": HexColor("#3B3B3B"), "marker": "circle"},
        "Proposed": {"color": HexColor("#0072B2"), "marker": "diamond"},
        "No BR": {"color": HexColor("#D55E00"), "marker": "triangle"},
    }
    # Keep the original upper-panel height and the same single-column width.
    # The lower panel shares its horizontal axis and points downward so the
    # cost traces remain visually dominant while daily implementation error is
    # directly aligned with the corresponding day.
    # Preserve the original 6-inch plotting region and add only enough page
    # width for the right-side shared-axis title.
    page_width = 6.40 * 72.0
    # Add a dedicated bottom legend band while preserving both panel heights.
    page_height = 5.325 * 72.0
    left, lower_bottom, top = 61.0, 45.0, 39.0
    plot_width = 6.00 * 72.0 - left - 9.0
    right = page_width - left - plot_width
    shared_axis_y = 183.0
    upper_top = page_height - top
    upper_plot_height = upper_top - shared_axis_y
    lower_plot_height = shared_axis_y - lower_bottom
    all_values = pivot.to_numpy(dtype=float)
    y_step = 1000.0
    # Reserve one full cost interval below the lowest observation.  This moves
    # all three traces away from the shared-axis date labels without changing
    # any cost value.
    y_min = y_step * np.floor((all_values.min() - 1000.0) / y_step)
    y_max = y_step * np.ceil((all_values.max() + 150.0) / y_step)
    deviation_min = 300.0
    deviation_break = 700.0
    deviation_max = 100.0 * np.ceil(
        (float(np.max(threshold_values)) * 1.04) / 100.0
    )
    primary_deviation_fraction = 0.70
    primary_deviation_height = lower_plot_height * primary_deviation_fraction
    compressed_deviation_height = lower_plot_height - primary_deviation_height
    if float(np.min(deviation_values)) < deviation_min:
        raise ValueError(
            "The requested 300-kW deviation-axis origin exceeds an observation"
        )
    if float(np.max(deviation_values)) > deviation_break:
        raise ValueError("The daily deviation bars exceed the primary axis segment")
    if float(np.max(threshold_values)) > deviation_max:
        raise ValueError("A security threshold exceeds the displayed axis")

    def x_coord(day: float) -> float:
        return left + (day - 1.0) * plot_width / 29.0

    def y_coord(value: float) -> float:
        return shared_axis_y + (value - y_min) * upper_plot_height / (
            y_max - y_min
        )

    def deviation_y(value: float) -> float:
        if value <= deviation_break:
            return shared_axis_y - (value - deviation_min) * (
                primary_deviation_height / (deviation_break - deviation_min)
            )
        return (
            shared_axis_y
            - primary_deviation_height
            - (value - deviation_break)
            * compressed_deviation_height
            / (deviation_max - deviation_break)
        )

    def draw_marker(
        pdf: canvas.Canvas, marker: str, x: float, y: float, color: HexColor
    ) -> None:
        radius = 3.0
        pdf.setFillColor(color)
        pdf.setStrokeColorRGB(1.0, 1.0, 1.0)
        pdf.setLineWidth(0.65)
        if marker == "circle":
            pdf.circle(x, y, radius, stroke=1, fill=1)
        elif marker == "diamond":
            path = pdf.beginPath()
            path.moveTo(x, y + radius + 0.3)
            path.lineTo(x + radius + 0.3, y)
            path.lineTo(x, y - radius - 0.3)
            path.lineTo(x - radius - 0.3, y)
            path.close()
            pdf.drawPath(path, stroke=1, fill=1)
        else:
            path = pdf.beginPath()
            path.moveTo(x, y + radius + 0.5)
            path.lineTo(x + radius + 0.7, y - radius)
            path.lineTo(x - radius - 0.7, y - radius)
            path.close()
            pdf.drawPath(path, stroke=1, fill=1)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(FIGURE_PATH), pagesize=(page_width, page_height))
    pdf.setTitle(
        "Daily total cost and No-BR aggregate user-level power deviation"
    )
    pdf.setAuthor("Market Clearing Result 1 post-processing")

    pdf.setFont("Helvetica", 12.0)
    # The shared baseline is labeled by the lower panel (300 kW); start upper
    # cost labels one interval above it to avoid two labels at one coordinate.
    for value in np.arange(y_min + y_step, y_max + 0.1, y_step):
        y = y_coord(float(value))
        pdf.setFillColor(HexColor("#333333"))
        pdf.drawRightString(left - 6.0, y - 3.7, f"{int(value):,}")

    # Label both scales at the shared origin: cost above and deviation below.
    pdf.setFillColor(HexColor("#333333"))
    pdf.setFont("Helvetica", 12.0)
    pdf.drawRightString(left - 6.0, shared_axis_y + 4.5, f"{int(y_min):,}")

    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.0)
    pdf.line(left, shared_axis_y, page_width - right, shared_axis_y)
    pdf.line(left, shared_axis_y, left, upper_top)
    date_ticks = {
        5: "09/05",
        10: "09/10",
        15: "09/15",
        20: "09/20",
        25: "09/25",
        30: "09/30",
    }
    for day, label in date_ticks.items():
        x = x_coord(float(day))
        pdf.line(x, shared_axis_y, x, shared_axis_y - 3.2)
        pdf.setFillColor(HexColor("#333333"))
        # The shared day labels sit above the common horizontal axis, as
        # requested, and no separate "Days" axis title is used.
        pdf.drawCentredString(x, shared_axis_y + 4.5, label)

    for method in METHOD_ORDER:
        style = styles[method]
        values = pivot[method].to_numpy(dtype=float)
        points = [(x_coord(float(day)), y_coord(value)) for day, value in zip(range(1, 31), values)]
        pdf.setStrokeColor(style["color"])
        pdf.setLineWidth(1.65)
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.drawPath(path, stroke=1, fill=0)
        for point in points:
            draw_marker(pdf, str(style["marker"]), point[0], point[1], style["color"])

    # Lower panel: expected daily-mean absolute No-BR implementation deviation
    # over all 3,000 users.  Negative PDF heights are a presentation device
    # only; tick labels retain magnitudes.
    deviation_ticks = [300.0, 400.0, 500.0, 600.0, 700.0]
    compressed_ticks = [1000.0, 1250.0, 1500.0]
    for value in deviation_ticks + compressed_ticks:
        y = deviation_y(float(value))
        pdf.setFillColor(HexColor("#333333"))
        pdf.setFont("Helvetica", 12.0)
        label_y = y - 10.0 if value == deviation_min else y - 3.7
        pdf.drawRightString(left - 6.0, label_y, f"{int(value):,}")

    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.0)
    pdf.line(left, shared_axis_y, left, lower_bottom)
    # Mark the change in vertical scale between the detailed 300--700 kW
    # segment and the compressed 700--1500 kW threshold segment.
    break_y = deviation_y(deviation_break)
    pdf.setFillColorRGB(1.0, 1.0, 1.0)
    pdf.rect(left - 2.2, break_y - 4.5, 4.4, 9.0, stroke=0, fill=1)
    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.0)
    pdf.line(left - 3.5, break_y + 2.5, left + 3.5, break_y - 1.5)
    pdf.line(left - 3.5, break_y - 1.0, left + 3.5, break_y - 5.0)
    slot_width = plot_width / 30.0
    bar_width = 0.68 * slot_width
    pdf.setFillColor(HexColor("#D55E00"))
    pdf.setStrokeColor(HexColor("#D55E00"))
    for day, value in enumerate(deviation_values, start=1):
        x = x_coord(float(day))
        bar_x = min(
            max(x, left + bar_width / 2.0),
            page_width - right - bar_width / 2.0,
        )
        y = deviation_y(float(value))
        pdf.rect(
            bar_x - bar_width / 2.0,
            y,
            bar_width,
            shared_axis_y - y,
            stroke=0,
            fill=1,
        )

    threshold_color = HexColor("#6A3D9A")
    for day, value in enumerate(threshold_values, start=1):
        x = x_coord(float(day))
        marker_x = min(
            max(x, left + bar_width / 2.0),
            page_width - right - bar_width / 2.0,
        )
        draw_marker(
            pdf,
            "diamond",
            marker_x,
            deviation_y(float(value)),
            threshold_color,
        )

    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica", 12.0)
    pdf.drawString(page_width - right + 7.0, shared_axis_y - 9.0, "Date")

    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica", 14.0)
    pdf.saveState()
    pdf.translate(12.0, (shared_axis_y + upper_top) / 2.0)
    pdf.rotate(90)
    pdf.drawCentredString(0.0, 0.0, "Total cost ($)")
    pdf.restoreState()
    pdf.saveState()
    pdf.translate(12.0, (lower_bottom + shared_axis_y) / 2.0)
    pdf.rotate(90)
    pdf.drawCentredString(0.0, 0.0, "Power deviation (kW)")
    pdf.restoreState()

    threshold_legend = "Deviation threshold of security constraint violation"
    threshold_legend_font_size = 11.0
    pdf.setFont("Helvetica", threshold_legend_font_size)
    threshold_legend_width = pdf.stringWidth(
        threshold_legend, "Helvetica", threshold_legend_font_size
    )
    threshold_legend_text_x = page_width - right - threshold_legend_width
    threshold_legend_y = 30.0
    threshold_legend_box_x = threshold_legend_text_x - 19.0
    threshold_legend_box_y = threshold_legend_y - 11.5
    threshold_legend_box_right = page_width - right + 6.0
    pdf.saveState()
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setStrokeColor(HexColor("#666666"))
    pdf.setLineWidth(0.75)
    pdf.roundRect(
        threshold_legend_box_x,
        threshold_legend_box_y,
        threshold_legend_box_right - threshold_legend_box_x,
        23.0,
        2.5,
        stroke=1,
        fill=1,
    )
    pdf.restoreState()
    draw_marker(
        pdf,
        "diamond",
        threshold_legend_text_x - 10.0,
        threshold_legend_y,
        threshold_color,
    )
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica", threshold_legend_font_size)
    pdf.drawString(
        threshold_legend_text_x,
        threshold_legend_y - 3.7,
        threshold_legend,
    )

    legend_y = page_height - 17.0
    legend_x = left + 38.0
    for method, width in zip(METHOD_ORDER, [112.0, 101.0, 0.0]):
        style = styles[method]
        pdf.setStrokeColor(style["color"])
        pdf.setLineWidth(1.65)
        pdf.line(legend_x, legend_y, legend_x + 18.0, legend_y)
        draw_marker(pdf, str(style["marker"]), legend_x + 9.0, legend_y, style["color"])
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 11.5)
        pdf.drawString(legend_x + 23.0, legend_y - 3.7, DISPLAY_LABELS[method])
        legend_x += width
    pdf.showPage()
    pdf.save()

    if preview_png is not None:
        import pypdfium2 as pdfium

        preview_png.parent.mkdir(parents=True, exist_ok=True)
        document = pdfium.PdfDocument(str(FIGURE_PATH))
        page = document[0]
        bitmap = page.render(scale=3.0)
        bitmap.to_pil().save(preview_png)
        page.close()
        document.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-png", type=Path)
    args = parser.parse_args()
    daily = revalue_outputs()
    plot_daily_cost(daily, args.preview_png)
    print(f"Updated: {OUTPUT_DIR}")
    print(f"Figure:  {FIGURE_PATH}")


if __name__ == "__main__":
    main()
