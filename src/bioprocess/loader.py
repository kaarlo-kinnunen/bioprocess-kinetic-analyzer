"""Load and validate batch hydrogen-fermentation CSV logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Exact column names from data/sample_run.csv.
TIME_H = "time_h"
BIOMASS_G_L = "biomass_gL"
OD600 = "OD600"
NH3_N_G_L = "NH3_N_gL"
PO4_G_L = "PO4_gL"
H2_UPTAKE_MMOL_L_H = "H2_uptake_mmol_L_h"
CO2_UPTAKE_MMOL_L_H = "CO2_uptake_mmol_L_h"
O2_UPTAKE_MMOL_L_H = "O2_uptake_mmol_L_h"
OFFGAS_H2_PCT = "offgas_H2_pct"
OFFGAS_CO2_PCT = "offgas_CO2_pct"
OFFGAS_O2_PCT = "offgas_O2_pct"
DO_PERCENT = "DO_percent"
PH = "pH"
TEMPERATURE_C = "temperature_C"
AGITATION_RPM = "agitation_rpm"
PRESSURE_BAR = "pressure_bar"

REQUIRED_COLUMNS = [TIME_H, BIOMASS_G_L, NH3_N_G_L]
OPTIONAL_COLUMNS = [
	OD600,
	PO4_G_L,
	H2_UPTAKE_MMOL_L_H,
	CO2_UPTAKE_MMOL_L_H,
	O2_UPTAKE_MMOL_L_H,
	OFFGAS_H2_PCT,
	OFFGAS_CO2_PCT,
	OFFGAS_O2_PCT,
	DO_PERCENT,
	PH,
	TEMPERATURE_C,
	AGITATION_RPM,
	PRESSURE_BAR,
]

# Inclusive lower and upper physical bounds; None denotes no bound.
PHYSICAL_RANGES: dict[str, tuple[float | None, float | None]] = {
	TIME_H: (0.0, None),
	BIOMASS_G_L: (0.0, None),
	OD600: (0.0, None),
	NH3_N_G_L: (0.0, None),
	PO4_G_L: (0.0, None),
	H2_UPTAKE_MMOL_L_H: (0.0, None),
	CO2_UPTAKE_MMOL_L_H: (0.0, None),
	O2_UPTAKE_MMOL_L_H: (0.0, None),
	OFFGAS_H2_PCT: (0.0, 100.0),
	OFFGAS_CO2_PCT: (0.0, 100.0),
	OFFGAS_O2_PCT: (0.0, 100.0),
	DO_PERCENT: (0.0, 100.0),
	PH: (0.0, 14.0),
	TEMPERATURE_C: (0.0, 100.0),
	AGITATION_RPM: (0.0, None),
	PRESSURE_BAR: (0.0, None),
}

DEFAULT_MAX_GAP = 3


class DataValidationError(ValueError):
	"""Raised when a bioreactor CSV cannot be validated for analysis."""


@dataclass(frozen=True)
class ValidationReport:
	"""Summary of changes and warnings produced while loading a run."""

	n_rows: int
	n_missing_filled: int
	n_out_of_range: int
	n_duplicates_dropped: int
	missing_optional_columns: list[str]
	warnings: list[str]


def _interpolate_short_gaps(
	frame: pd.DataFrame,
	time_values: np.ndarray,
	max_gap: int,
) -> int:
	"""Fill bounded NaN runs no longer than max_gap by time-linear interpolation."""
	filled_count = 0
	for column in frame.columns:
		if column == TIME_H:
			continue

		values = frame[column].to_numpy(dtype=float, copy=True)
		missing_indices = np.flatnonzero(np.isnan(values))
		position = 0
		while position < len(missing_indices):
			run_start = position
			while (
				position + 1 < len(missing_indices)
				and missing_indices[position + 1] == missing_indices[position] + 1
			):
				position += 1

			start_index = int(missing_indices[run_start])
			end_index = int(missing_indices[position])
			run_length = end_index - start_index + 1
			left_index = start_index - 1
			right_index = end_index + 1

			if (
				run_length <= max_gap
				and left_index >= 0
				and right_index < len(values)
				and np.isfinite(values[left_index])
				and np.isfinite(values[right_index])
				and np.isfinite(time_values[left_index])
				and np.isfinite(time_values[right_index])
				and time_values[right_index] > time_values[left_index]
			):
				values[start_index : end_index + 1] = np.interp(
					time_values[start_index : end_index + 1],
					[time_values[left_index], time_values[right_index]],
					[values[left_index], values[right_index]],
				)
				filled_count += run_length

			position += 1

		frame[column] = values

	return filled_count


def load_run_with_report(
	path: str | Path,
	max_gap: int = DEFAULT_MAX_GAP,
) -> tuple[pd.DataFrame, ValidationReport]:
	"""Load a CSV log, clean valid short gaps, and return a validation report."""
	if max_gap < 0:
		raise ValueError("max_gap must be non-negative")

	input_path = Path(path)
	if not input_path.is_file():
		raise DataValidationError(f"CSV file does not exist: {input_path}")
	if input_path.stat().st_size == 0:
		raise DataValidationError(f"CSV file is empty: {input_path}")

	try:
		frame = pd.read_csv(input_path)
	except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
		raise DataValidationError(f"Could not read CSV file '{input_path}': {exc}") from exc

	if frame.empty:
		raise DataValidationError(f"CSV file contains no data rows: {input_path}")

	frame.columns = [str(column).strip() for column in frame.columns]
	if frame.columns.duplicated().any():
		duplicates = frame.columns[frame.columns.duplicated()].tolist()
		raise DataValidationError(f"Duplicate column names after whitespace normalization: {duplicates}")

	missing_required = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
	if missing_required:
		raise DataValidationError(f"Missing required columns: {', '.join(missing_required)}")

	missing_optional = [column for column in OPTIONAL_COLUMNS if column not in frame.columns]
	warnings: list[str] = []
	if missing_optional:
		warnings.append(f"Optional columns not present: {', '.join(missing_optional)}")

	parse_failures: dict[str, int] = {}
	for column in frame.columns:
		original = frame[column]
		numeric = pd.to_numeric(original, errors="coerce")
		populated = original.notna() & original.astype("string").str.strip().ne("")
		failed_count = int((populated & numeric.isna()).sum())
		if failed_count:
			parse_failures[column] = failed_count
		frame[column] = numeric

	if parse_failures:
		details = ", ".join(f"{column}: {count}" for column, count in parse_failures.items())
		warnings.append(f"Values that failed numeric parsing were set to missing ({details})")

	if frame[TIME_H].isna().any() or not np.isfinite(frame[TIME_H].to_numpy(dtype=float)).all():
		raise DataValidationError(f"Required time column '{TIME_H}' contains missing or non-finite values")

	frame = frame.sort_values(TIME_H, kind="stable")
	original_rows = len(frame)
	frame = frame.drop_duplicates(subset=[TIME_H], keep="first")
	duplicates_dropped = original_rows - len(frame)
	frame = frame.reset_index(drop=True)

	time_values = frame[TIME_H].to_numpy(dtype=float)
	if len(time_values) > 1 and not np.all(np.diff(time_values) > 0):
		raise DataValidationError(f"'{TIME_H}' must be strictly increasing after duplicate timestamps are removed")

	if len(time_values) > 1:
		intervals = np.diff(time_values)
		median_interval = float(np.median(intervals))
		if median_interval > 0 and float(np.max(np.abs(intervals - median_interval))) / median_interval > 0.2:
			warnings.append("Sampling interval is irregular (more than 20% variation from the median interval)")
	if len(time_values) and float(time_values[-1] - time_values[0]) < 10.0:
		warnings.append("Run duration is shorter than 10 hours")

	missing_filled = _interpolate_short_gaps(frame, time_values, max_gap)

	out_of_range_count = 0
	for column, (lower, upper) in PHYSICAL_RANGES.items():
		if column not in frame.columns:
			continue
		values = frame[column]
		outside = pd.Series(False, index=frame.index)
		if lower is not None:
			outside |= values < lower
		if upper is not None:
			outside |= values > upper
		count = int(outside.sum())
		if count:
			out_of_range_count += count
			frame.loc[outside, column] = np.nan

	if out_of_range_count:
		warnings.append(f"{out_of_range_count} values outside physical ranges were set to missing")

	report = ValidationReport(
		n_rows=len(frame),
		n_missing_filled=missing_filled,
		n_out_of_range=out_of_range_count,
		n_duplicates_dropped=duplicates_dropped,
		missing_optional_columns=missing_optional,
		warnings=warnings,
	)
	return frame.reset_index(drop=True), report


def load_run(path: str | Path) -> pd.DataFrame:
	"""Load and validate a CSV log, returning its cleaned DataFrame."""
	frame, _ = load_run_with_report(path)
	return frame


__all__ = [
	"REQUIRED_COLUMNS",
	"OPTIONAL_COLUMNS",
	"PHYSICAL_RANGES",
	"DataValidationError",
	"ValidationReport",
	"load_run",
	"load_run_with_report",
]
