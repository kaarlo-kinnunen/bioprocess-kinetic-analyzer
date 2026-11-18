from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.insert(0, str(SRC_PATH))

from bioprocess.loader import DataValidationError, load_run, load_run_with_report


def _write_csv(tmp_path: Path, name: str, data: dict[str, list[object]]) -> Path:
	"""Write a small CSV log for a loader test."""
	path = tmp_path / name
	pd.DataFrame(data).to_csv(path, index=False)
	return path


def test_loads_valid_csv(tmp_path):
	"""A valid CSV should load with required columns and numeric values intact."""
	path = _write_csv(
		tmp_path,
		"valid.csv",
		{
			"time_h": [0, 1, 2],
			"biomass_gL": [0.1, 0.2, 0.3],
			"NH3_N_gL": [2.0, 1.9, 1.8],
		},
	)

	result = load_run(path)

	assert list(result.columns) == ["time_h", "biomass_gL", "NH3_N_gL"]
	assert result["time_h"].tolist() == [0, 1, 2]
	assert result["biomass_gL"].tolist() == [0.1, 0.2, 0.3]


def test_missing_required_column_raises(tmp_path):
	"""A missing required nitrogen column should be named in the error."""
	path = _write_csv(
		tmp_path,
		"missing_column.csv",
		{"time_h": [0, 1], "biomass_gL": [0.1, 0.2]},
	)

	with pytest.raises(DataValidationError, match="NH3_N_gL"):
		load_run(path)


def test_duplicate_timestamps_keep_first_row(tmp_path):
	"""Duplicate timestamps should be removed after stable sorting."""
	path = _write_csv(
		tmp_path,
		"duplicates.csv",
		{
			"time_h": [0, 1, 1, 2],
			"biomass_gL": [0.1, 0.2, 9.9, 0.4],
			"NH3_N_gL": [2.0, 1.9, 0.1, 1.7],
		},
	)

	result, report = load_run_with_report(path)

	assert result["time_h"].tolist() == [0, 1, 2]
	assert result.loc[result["time_h"] == 1, "biomass_gL"].item() == 0.2
	assert report.n_duplicates_dropped == 1


def test_short_nan_gap_is_interpolated(tmp_path):
	"""An interior gap of at most three points should be interpolated linearly."""
	path = _write_csv(
		tmp_path,
		"short_gap.csv",
		{
			"time_h": [0, 1, 2, 3],
			"biomass_gL": [0.0, np.nan, np.nan, 6.0],
			"NH3_N_gL": [2.0, 1.9, 1.8, 1.7],
		},
	)

	result, report = load_run_with_report(path)

	assert result["biomass_gL"].tolist() == [0.0, 2.0, 4.0, 6.0]
	assert report.n_missing_filled == 2


def test_out_of_range_ph_is_set_to_nan(tmp_path):
	"""pH outside the physical range should be flagged as missing, not dropped."""
	path = _write_csv(
		tmp_path,
		"bad_ph.csv",
		{
			"time_h": [0, 1],
			"biomass_gL": [0.1, 0.2],
			"NH3_N_gL": [2.0, 1.9],
			"pH": [7.0, 15.0],
		},
	)

	result, report = load_run_with_report(path)

	assert len(result) == 2
	assert result.loc[0, "pH"] == 7.0
	assert pd.isna(result.loc[1, "pH"])
	assert report.n_out_of_range == 1


def test_nonexistent_path_raises_clear_error(tmp_path):
	"""A missing file should raise a clear custom validation error."""
	path = tmp_path / "does_not_exist.csv"

	with pytest.raises(DataValidationError, match="CSV file does not exist"):
		load_run(path)
