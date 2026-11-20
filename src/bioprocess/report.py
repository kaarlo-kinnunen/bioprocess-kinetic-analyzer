"""Executive PDF report generation for a synthetic Knallgas gas-fermentation run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .kinetics import TIME_H, summarize_run
from .phases import PhaseSummary, detect_phases, summarize_phases
from .plots import make_all_plots

# Standard output limit values.
DEFAULT_LIMITS: Dict[str, tuple[float, float]] = {
    "pH": (6.5, 7.5),
    "temperature_C": (28.0, 33.0),
    "pressure_bar": (1.0, 1.5),
}

PALETTE: Dict[str, str] = {
    "biomass": "#1f77b4",
    "od600": "#ff7f0e",
    "nh3": "#2ca02c",
    "phosphate": "#9467bd",
    "h2": "#d62728",
    "co2": "#17becf",
    "o2": "#bcbd22",
    "offgas_h2": "#d62728",
    "offgas_co2": "#17becf",
    "offgas_o2": "#bcbd22",
    "do": "#8c564b",
    "mu": "#e377c2",
    "pH": "#6a3d9a",
    "temperature": "#b15928",
    "agitation": "#fb9a99",
    "pressure": "#a6cee3",
    "good": "#2ca02c",
    "bad": "#d62728",
    "warning": "#f6c65b",
    "critical": "#f4cccc",
    "zebra": "#f7f7f7",
    "text": "#1f1f1f",
    "muted": "#666666",
}


@dataclass(frozen=True)
class Excursion:
    """One continuous excursion beyond a process bound."""

    parameter: str
    start_h: float
    end_h: float
    duration_h: float
    min_value: float
    max_value: float
    limit_low: Optional[float]
    limit_high: Optional[float]
    severity: str


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert values to float while handling missing inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _make_styles() -> Dict[str, ParagraphStyle]:
    """Return the report’s paragraph styles."""
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=HexColor(PALETTE["text"]),
            spaceAfter=10,
            alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=HexColor(PALETTE["muted"]),
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=HexColor(PALETTE["text"]),
            spaceBefore=12,
            spaceAfter=6,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=HexColor(PALETTE["muted"]),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=HexColor(PALETTE["text"]),
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            textColor=HexColor(PALETTE["text"]),
            alignment=1,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=8,
            textColor=HexColor(PALETTE["muted"]),
            alignment=1,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=HexColor(PALETTE["text"]),
        ),
        "badge_good": ParagraphStyle(
            "BadgeGood",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=HexColor(PALETTE["good"]),
            borderPadding=4,
            borderWidth=1,
            borderColor=HexColor(PALETTE["good"]),
            backColor=HexColor("#eaf7ea"),
            alignment=1,
        ),
        "badge_bad": ParagraphStyle(
            "BadgeBad",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=HexColor(PALETTE["bad"]),
            borderPadding=4,
            borderWidth=1,
            borderColor=HexColor(PALETTE["bad"]),
            backColor=HexColor(PALETTE["critical"]),
            alignment=1,
        ),
    }


def _format_metric(value: float, digits: int = 3, nan_value: str = "-") -> str:
    """Return a readable value string for a metric cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return nan_value
    if abs(value) >= 1000:
        digits = 1
    return f"{value:.{digits}f}"


def _draw_footer(canvas, doc) -> None:
    """Draw the standard footer on each report page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor(PALETTE["muted"]))
    page_number = canvas.getPageNumber()
    canvas.drawString(2 * cm, 0.8 * cm, f"{doc.run_id}")
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.8 * cm, str(page_number))
    canvas.drawRightString(doc.pagesize[0] - 2 * cm, 0.8 * cm, "Generated by bioprocess-kinetic-analyzer")
    canvas.restoreState()


def _phase_table_rows(phase_summary: Optional[Dict[str, Optional[PhaseSummary]]]) -> List[List[str]]:
    """Return the rows for the phase summary table."""
    rows: List[List[str]] = [["Phase", "Start (h)", "End (h)", "Duration (h)", "Mean mu (1/h)", "Biomass start (g/L)", "Biomass end (g/L)"]]
    if not phase_summary:
        rows.append(["-", "-", "-", "-", "-", "-", "-"])
        return rows

    for phase_name in ["lag", "exponential", "transition", "stationary"]:
        phase = phase_summary.get(phase_name)
        if phase is None:
            rows.append([phase_name, "-", "-", "-", "-", "-", "-"])
            continue
        rows.append(
            [
                phase.phase,
                _format_metric(phase.start_h, digits=1),
                _format_metric(phase.end_h, digits=1),
                _format_metric(phase.duration_h, digits=2),
                _format_metric(phase.mu_mean_1_per_h, digits=3),
                _format_metric(phase.biomass_start_g_L, digits=2),
                _format_metric(phase.biomass_end_g_L, digits=2),
            ]
        )
    return rows


def _phase_note_text(phase_summary: Optional[Dict[str, Optional[PhaseSummary]]], nutrient_time: Optional[float]) -> str:
    """Construct a plain-language note about the phase boundary and nutrient switch."""
    if not phase_summary:
        return "Phase analysis was not available for this run."

    stationary = phase_summary.get("stationary")
    if stationary is None:
        return "No stationary phase was detected in the current dataset."

    if nutrient_time is None:
        return "Nitrogen depletion was not detected within the sampling window; stationary behavior was inferred from growth-rate decay alone."

    if stationary.nutrient_matches_boundary:
        return f"Nitrogen depletion aligned with the stationary boundary at approximately {nutrient_time:.1f} h."
    return f"Nitrogen depletion occurred at approximately {nutrient_time:.1f} h, but it did not coincide exactly with the stationary-phase boundary."


def _key_metrics_table(summary) -> List[List[object]]:
    """Return the 2x3 executive summary grid."""
    rows = [
        [
            ("Max specific growth rate\n(1/h)", _format_metric(summary.mu_max_1_per_h, digits=3)),
            ("Doubling time\n(h)", _format_metric(summary.doubling_time_h, digits=2)),
            ("Final biomass\n(g/L)", _format_metric(summary.final_biomass_g_L, digits=2)),
        ],
        [
            ("Biomass yield on H2\n(g/g)", _format_metric(summary.yield_h2_g_per_g, digits=3)),
            ("Volumetric productivity\n(g/L/h)", _format_metric(summary.productivity_g_L_h, digits=3)),
            ("Peak H2 uptake\n(mmol/L/h)", _format_metric(summary.peak_h2_uptake_mmol_L_h, digits=1)),
        ],
    ]
    metric_rows: List[List[object]] = []
    for row in rows:
        cell_row: List[object] = []
        for label, value in row:
            cell_row.append(
                Table(
                    [[Paragraph(label, style=_make_styles()["metric_label"]),], [Paragraph(value, style=_make_styles()["metric"]) ]],
                    colWidths=[90],
                )
            )
        metric_rows.append(cell_row)
    return metric_rows


def _make_executive_summary(summary, nutrient_time: Optional[float]) -> str:
    """Write a short narrative summary from the headline kinetics."""
    mu_max = _safe_float(summary.mu_max_1_per_h, 0.0)
    mu_time = _safe_float(summary.mu_max_time_h, np.nan)
    final_biomass = _safe_float(summary.final_biomass_g_L, 0.0)
    nutrient_text = "Nitrogen did not fall below the depletion threshold in the sampled run." if nutrient_time is None else f"Growth stopped when nitrogen ran out at approximately {nutrient_time:.1f} h."
    if np.isfinite(mu_time):
        growth_text = f"The culture reached a maximum specific growth rate of {mu_max:.3f} 1/h at {mu_time:.1f} h, rising to {final_biomass:.2f} g/L by the end of the run."
    else:
        growth_text = f"The culture reached a maximum specific growth rate of {mu_max:.3f} 1/h and finished the run at {final_biomass:.2f} g/L."
    return " ".join([growth_text, nutrient_text])


def find_cpp_excursions(df: pd.DataFrame, limits: Optional[Mapping[str, tuple[float, float]]] = None, min_consecutive_points: int = 2) -> List[Excursion]:
    """Return groupings of consecutive out-of-range CPP points as excursions."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if limits is None:
        limits = DEFAULT_LIMITS

    excursions: List[Excursion] = []
    time_h = pd.to_numeric(df[TIME_H], errors="coerce") if TIME_H in df.columns else pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if time_h.empty:
        return excursions

    for parameter, (lower, upper) in limits.items():
        if parameter not in df.columns:
            continue
        values = pd.to_numeric(df[parameter], errors="coerce").to_numpy(dtype=float)
        out_of_range = (values < lower) | (values > upper)
        if not np.any(out_of_range):
            continue

        mask = np.flatnonzero(out_of_range)
        start_index = 0
        while start_index < len(mask):
            end_index = start_index
            while end_index + 1 < len(mask) and mask[end_index + 1] - mask[end_index] <= 1:
                end_index += 1
            segment = mask[start_index : end_index + 1]
            if len(segment) >= min_consecutive_points:
                segment_values = values[segment]
                start_h = float(time_h.iloc[segment[0]]) if isinstance(time_h, pd.Series) else float(time_h[segment[0]])
                end_h = float(time_h.iloc[segment[-1]]) if isinstance(time_h, pd.Series) else float(time_h[segment[-1]])
                duration_h = float(end_h - start_h)
                severity = "warning" if duration_h < 1.0 else "critical"
                excursions.append(
                    Excursion(
                        parameter=parameter,
                        start_h=start_h,
                        end_h=end_h,
                        duration_h=duration_h,
                        min_value=float(np.min(segment_values)),
                        max_value=float(np.max(segment_values)),
                        limit_low=float(lower),
                        limit_high=float(upper),
                        severity=severity,
                    )
                )
            start_index = end_index + 1

    return sorted(excursions, key=lambda item: (item.start_h, item.parameter))


def _cpp_table(excursions: List[Excursion]) -> List[List[str | object]]:
    """Return the process alert table rows."""
    table_rows: List[List[str | object]] = [["Parameter", "Start (h)", "End (h)", "Duration (h)", "Min", "Max", "Limit range", "Severity"]]
    if not excursions:
        table_rows.append(["All critical process parameters stayed within limits", "-", "-", "-", "-", "-", "-", "-"])
        return table_rows

    for excursion in excursions:
        table_rows.append(
            [
                excursion.parameter,
                _format_metric(excursion.start_h, digits=1),
                _format_metric(excursion.end_h, digits=1),
                _format_metric(excursion.duration_h, digits=2),
                _format_metric(excursion.min_value, digits=2),
                _format_metric(excursion.max_value, digits=2),
                f"{excursion.limit_low:.2f} to {excursion.limit_high:.2f}",
                excursion.severity,
            ]
        )
    return table_rows


def build_report(df: pd.DataFrame, out_path: str | Path, run_id: str = "RUN-001", limits: Optional[Mapping[str, tuple[float, float]]] = None) -> Path:
    """Build the executive PDF report for a batch Knallgas gas-fermentation run."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    working = df.copy()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if limits is None:
        limits = DEFAULT_LIMITS

    summary = summarize_run(working)
    labels = detect_phases(working)
    phase_summary = summarize_phases(working, labels) if labels is not None else None
    nutrient_time = None
    if phase_summary is not None and "stationary" in phase_summary and phase_summary["stationary"] is not None:
        nutrient_time = phase_summary["stationary"].nutrient_limitation_time_h

    excursions = find_cpp_excursions(working, limits)
    status_text = "NO CPP ALERTS" if not excursions else f"{len(excursions)} CPP ALERTS"
    status_color = HexColor(PALETTE["good"]) if not excursions else HexColor(PALETTE["bad"])

    styles = _make_styles()
    story = []

    start_time = _safe_float(working.iloc[0]["time_h"], 0.0) if "time_h" in working.columns else 0.0
    end_time = _safe_float(working.iloc[-1]["time_h"], 0.0) if "time_h" in working.columns else 0.0
    run_duration = max(0.0, end_time - start_time)
    generated_on = datetime.now().strftime("%Y-%m-%d")

    header_table = Table(
        [[Paragraph("Bioprocess Kinetic Report", style=styles["title"]), Paragraph(status_text, style=styles["badge_good" if not excursions else "badge_bad"]) ]],
        colWidths=[A4[0] * 0.72, A4[0] * 0.28],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(header_table)
    story.append(
        Paragraph(
            f"Run ID: {run_id} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {generated_on} &nbsp;&nbsp;|&nbsp;&nbsp; Duration: {run_duration:.1f} h",
            style=styles["subtitle"],
        )
    )

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Executive summary", style=styles["section"]))

    metric_table = Table(
        [[
            Table([[Paragraph("Max specific growth rate\n(1/h)", style=styles["metric_label"]), Paragraph(_format_metric(summary.mu_max_1_per_h, digits=3), style=styles["metric"]) ]], colWidths=[125]),
            Table([[Paragraph("Doubling time\n(h)", style=styles["metric_label"]), Paragraph(_format_metric(summary.doubling_time_h, digits=2), style=styles["metric"]) ]], colWidths=[125]),
            Table([[Paragraph("Final biomass\n(g/L)", style=styles["metric_label"]), Paragraph(_format_metric(summary.final_biomass_g_L, digits=2), style=styles["metric"]) ]], colWidths=[125]),
        ], [
            Table([[Paragraph("Biomass yield on H2\n(g/g)", style=styles["metric_label"]), Paragraph(_format_metric(summary.yield_h2_g_per_g, digits=3), style=styles["metric"]) ]], colWidths=[125]),
            Table([[Paragraph("Volumetric productivity\n(g/L/h)", style=styles["metric_label"]), Paragraph(_format_metric(summary.productivity_g_L_h, digits=3), style=styles["metric"]) ]], colWidths=[125]),
            Table([[Paragraph("Peak H2 uptake\n(mmol/L/h)", style=styles["metric_label"]), Paragraph(_format_metric(summary.peak_h2_uptake_mmol_L_h, digits=1), style=styles["metric"]) ]], colWidths=[125]),
        ]],
        colWidths=[170, 170, 170],
        rowHeights=[80, 80],
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d9d9d9")),
                ("BACKGROUND", (0, 0), (-1, -1), HexColor(PALETTE["zebra"])),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(_make_executive_summary(summary, nutrient_time), style=styles["body"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Phase summary", style=styles["section"]))
    if labels is None:
        story.append(Paragraph("Phase detection was not available for this dataset; the phase summary block was skipped.", style=styles["body"]))
    else:
        phase_rows = _phase_table_rows(phase_summary)
        phase_table = Table(phase_rows, colWidths=[55, 50, 50, 55, 60, 70, 70])
        phase_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#eef3f8")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 7),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cfcfcf")),
                    ("BACKGROUND", (0, 1), (-1, -1), HexColor("#ffffff")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8f8f8")]),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(phase_table)
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(_phase_note_text(phase_summary, nutrient_time), style=styles["body"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("CPP alerts", style=styles["section"]))
    if not excursions:
        story.append(Paragraph("All critical process parameters stayed within limits", style=styles["badge_good"]))
    else:
        exc_rows = _cpp_table(excursions)
        exc_table = Table(exc_rows, colWidths=[55, 50, 50, 55, 50, 50, 75, 50])
        exc_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#eef3f8")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 7),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d0d0d0")),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        for row_index in range(1, len(exc_rows)):
            severity = exc_rows[row_index][-1]
            color = HexColor(PALETTE["critical"]) if severity == "critical" else HexColor(PALETTE["warning"])
            exc_table.setStyle(TableStyle([("BACKGROUND", (0, row_index), (-1, row_index), color)]))
        story.append(exc_table)

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Charts", style=styles["section"]))

    with tempfile.TemporaryDirectory(prefix="bioprocess-report-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        plot_map = make_all_plots(working, labels, limits, temp_dir_path)
        ordered_plot_names = ["growth", "growth_rate", "nutrients", "gas_uptake", "offgas_do", "cpp"]
        captions = {
            "growth": "Growth and OD600 over time",
            "growth_rate": "Specific growth rate profile with mu_max annotation",
            "nutrients": "Ammonia-nitrogen and phosphate depletion",
            "gas_uptake": "H2, CO2, and O2 uptake rates",
            "offgas_do": "Off-gas composition and dissolved oxygen",
            "cpp": "Process control parameters and operating limits",
        }

        for index, plot_name in enumerate(ordered_plot_names, start=1):
            plot_path = plot_map.get(plot_name)
            if plot_path is None or not plot_path.exists():
                story.append(Paragraph(f"Figure {index}: {captions.get(plot_name, plot_name)} not available for this run.", style=styles["body"]))
                story.append(Spacer(1, 0.2 * cm))
                continue

            try:
                fig = Image(str(plot_path))
            except Exception:
                story.append(Paragraph(f"Figure {index}: {captions.get(plot_name, plot_name)} could not be loaded.", style=styles["body"]))
                story.append(Spacer(1, 0.2 * cm))
                continue

            max_width = 15 * cm
            max_height = 10.5 * cm
            scale = min(max_width / fig.imageWidth, max_height / fig.imageHeight, 1.0)
            fig.drawWidth = fig.imageWidth * scale
            fig.drawHeight = fig.imageHeight * scale
            fig.hAlign = "CENTER"
            story.append(KeepTogether([Paragraph(f"Figure {index}: {captions[plot_name]}", style=styles["small"]), fig, Spacer(1, 0.2 * cm)]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Methods and assumptions", style=styles["section"]))
    story.append(
        Paragraph(
            "Specific growth rates were calculated from rolling log-linear fits to biomass data over a 4 h window. Biomass yield on H2 was obtained from total H2 uptake across the run, while volumetric productivity was estimated from the net change in biomass over the full run duration. Phase boundaries were assigned from growth-rate transitions and nutrient depletion thresholds, and the report is based on synthetic simulated data.",
            style=styles["body"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.run_id = run_id
    doc.pagesize = A4
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output_path


__all__ = [
    "DEFAULT_LIMITS",
    "Excursion",
    "find_cpp_excursions",
    "build_report",
]
