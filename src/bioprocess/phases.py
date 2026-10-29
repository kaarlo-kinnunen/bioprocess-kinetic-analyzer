"""Phase detection for batch hydrogen-fermentation runs using dynamic growth-rate changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .kinetics import BIOMASS_G_L, NH3_N_G_L, TIME_H, specific_growth_rate_series

# Exact CSV column names from data/sample_run.csv.
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

REQUIRED_COLUMNS = [
    TIME_H,
    BIOMASS_G_L,
    OD600,
    NH3_N_G_L,
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

EPSILON = 1e-9


@dataclass(frozen=True)
class PhaseSummary:
    """Summary for a detected fermentation phase."""

    phase: str
    start_h: float
    end_h: float
    duration_h: float
    mu_mean_1_per_h: float
    biomass_start_g_L: float
    biomass_end_g_L: float
    nutrient_limitation_time_h: Optional[float]
    nutrient_matches_boundary: bool


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a non-mutating, time-sorted copy with numeric data and required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()
    cleaned[TIME_H] = pd.to_numeric(cleaned[TIME_H], errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TIME_H]).sort_values(TIME_H).reset_index(drop=True)
    cleaned[BIOMASS_G_L] = cleaned[BIOMASS_G_L].clip(lower=EPSILON)
    return cleaned


def _smooth_mu(mu_series: pd.Series, window_length: int = 5, polyorder: int = 2) -> pd.Series:
    """Smooth the specific growth-rate series to reduce noise and phase flicker."""
    if mu_series.empty:
        return mu_series.copy()

    values = mu_series.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return pd.Series(np.zeros_like(values, dtype=float), index=mu_series.index, name=mu_series.name)

    if finite.sum() < 3:
        filled = np.where(finite, values, 0.0)
        return pd.Series(filled, index=mu_series.index, name=mu_series.name)

    filled = values.copy()
    if not finite.all():
        valid_indices = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), valid_indices, values[finite])

    if window_length <= 1:
        return pd.Series(filled, index=mu_series.index, name=mu_series.name)

    window = min(window_length, len(filled))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return pd.Series(filled, index=mu_series.index, name=mu_series.name)
    if window > len(filled):
        window = len(filled) if len(filled) % 2 == 1 else len(filled) - 1
    if window < 3:
        return pd.Series(filled, index=mu_series.index, name=mu_series.name)

    poly = min(polyorder, window - 1)
    if poly < 1:
        poly = 1

    smoothed = savgol_filter(filled, window_length=window, polyorder=poly, mode="interp")
    return pd.Series(smoothed, index=mu_series.index, name=mu_series.name)


def _segment_duration_h(time_h: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Return phase duration in hours from the start and end indices."""
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(time_h):
        return 0.0
    return float(time_h[end_idx] - time_h[start_idx])


def _apply_min_duration(labels: np.ndarray, time_h: np.ndarray, min_phase_duration_h: float) -> np.ndarray:
    """Relabel short segments to suppress flicker between phase states."""
    if len(labels) == 0 or min_phase_duration_h <= 0:
        return labels.copy()

    result = labels.copy()
    if len(result) < 2:
        return result

    start = 0
    while start < len(result):
        end = start + 1
        while end < len(result) and result[end] == result[start]:
            end += 1
        segment_duration_h = _segment_duration_h(time_h, start, end - 1)
        if segment_duration_h < min_phase_duration_h and result[start] in {"lag", "exponential", "stationary"}:
            neighbor_labels = []
            if start > 0:
                neighbor_labels.append(result[start - 1])
            if end < len(result):
                neighbor_labels.append(result[end])
            if neighbor_labels:
                replacement = max(set(neighbor_labels), key=lambda item: list(neighbor_labels).count(item))
                result[start:end] = replacement
        start = end
    return result


def detect_phases(
    df: pd.DataFrame,
    *,
    window_h: float = 4.0,
    smoothing_window: int = 9,
    min_phase_duration_h: float = 2.0,
    lag_fraction: float = 0.20,
    exponential_threshold_fraction: float = 0.50,
    stationary_threshold_fraction: float = 0.10,
    ammonia_threshold_g_l: float = 0.10,
    nutrient_tolerance_h: float = 3.0,
) -> pd.Series:
    """Classify each time point as lag, exponential, stationary, or transition using mu(t)."""
    working = _prepare_df(df)
    if working.empty or len(working) < 3:
        return pd.Series(np.full(len(working), "lag", dtype=object), index=working.index, name="phase")

    mu_series = specific_growth_rate_series(working, window_h=window_h)
    mu_smoothed = _smooth_mu(mu_series, window_length=smoothing_window, polyorder=2)
    mu_smoothed = mu_smoothed.ffill().bfill().fillna(0.0)

    mu_max = float(mu_smoothed.max()) if not mu_smoothed.empty else 0.0
    if not np.isfinite(mu_max) or mu_max <= 0:
        labels = pd.Series(np.full(len(working), "stationary", dtype=object), index=working.index, name="phase")
        return labels

    lag_threshold = lag_fraction * mu_max
    exp_threshold = exponential_threshold_fraction * mu_max
    stationary_threshold = stationary_threshold_fraction * mu_max

    peak_index = int(mu_smoothed.idxmax())
    time_h = working[TIME_H].to_numpy(dtype=float)
    labels = np.full(len(working), "transition", dtype=object)

    # Lag phase: before growth first reaches the lag threshold.
    lag_end_index = None
    for idx in range(0, peak_index + 1):
        if mu_smoothed.iloc[idx] >= lag_threshold:
            lag_end_index = idx
            break
    if lag_end_index is None:
        lag_end_index = peak_index
    labels[:lag_end_index] = "lag"

    # Exponential phase: starts once lag threshold is crossed and continues until the post-peak
    # decline drops below 50% of mu_max.
    exp_end_index = len(working) - 1
    for idx in range(peak_index + 1, len(working)):
        if mu_smoothed.iloc[idx] < exp_threshold:
            exp_end_index = idx
            break
    labels[lag_end_index:exp_end_index] = "exponential"

    # Stationary phase: occurs after the exponential phase when mu stays low for a minimum duration.
    stationary_start_index = None
    for idx in range(exp_end_index + 1, len(working)):
        if mu_smoothed.iloc[idx] < stationary_threshold:
            stationary_start_index = idx
            break
    if stationary_start_index is not None:
        labels[stationary_start_index:] = "stationary"

    # Keep the early part of the run in lag if no growth is observed.
    if lag_end_index == 0:
        labels[:] = "lag"

    # Hysteresis: ensure short segments are merged away to avoid phase flicker.
    labels = _apply_min_duration(labels, time_h, min_phase_duration_h)

    # If the stationary phase was not robust, leave it out; otherwise keep the low-mu segment.
    if stationary_start_index is None:
        labels[labels == "stationary"] = "transition"

    result = pd.Series(labels, index=working.index, name="phase")
    return result


def summarize_phases(df: pd.DataFrame, labels: pd.Series, *, window_h: float = 4.0, ammonia_threshold_g_l: float = 0.10, nutrient_tolerance_h: float = 3.0) -> Dict[str, Optional[PhaseSummary]]:
    """Summarize each phase segment in hours and biomass, with nutrient-limit checks."""
    working = _prepare_df(df)
    if working.empty:
        return {}

    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series indexed like df")

    if len(labels) != len(working):
        raise ValueError("labels must match the length of df")

    mu_raw = specific_growth_rate_series(working, window_h=window_h)
    mu_smooth = _smooth_mu(mu_raw, window_length=9, polyorder=2)
    mu_smooth = mu_smooth.ffill().bfill().fillna(0.0)

    nutrient_time = None
    if NH3_N_G_L in working.columns:
        nutrient_times = working[TIME_H][working[NH3_N_G_L] <= ammonia_threshold_g_l]
        if not nutrient_times.empty:
            nutrient_time = float(nutrient_times.iloc[0])

    phase_summary: Dict[str, Optional[PhaseSummary]] = {}
    for phase_name in ["lag", "exponential", "stationary", "transition"]:
        mask = labels == phase_name
        if not mask.any():
            phase_summary[phase_name] = None
            continue

        idx = mask[mask].index
        start_idx = idx[0]
        end_idx = idx[-1]
        start_h = float(working.loc[start_idx, TIME_H])
        end_h = float(working.loc[end_idx, TIME_H])
        duration_h = end_h - start_h

        mu_values = mu_smooth.loc[start_idx:end_idx].to_numpy(dtype=float)
        mu_mean = float(np.nanmean(mu_values)) if mu_values.size else 0.0

        biomass_start = float(working.loc[start_idx, BIOMASS_G_L])
        biomass_end = float(working.loc[end_idx, BIOMASS_G_L])

        nutrient_matches = False
        if phase_name == "stationary" and nutrient_time is not None:
            stationary_boundary_h = float(start_h)
            nutrient_matches = abs(nutrient_time - stationary_boundary_h) <= nutrient_tolerance_h

        phase_summary[phase_name] = PhaseSummary(
            phase=phase_name,
            start_h=start_h,
            end_h=end_h,
            duration_h=duration_h,
            mu_mean_1_per_h=mu_mean,
            biomass_start_g_L=biomass_start,
            biomass_end_g_L=biomass_end,
            nutrient_limitation_time_h=nutrient_time,
            nutrient_matches_boundary=nutrient_matches,
        )

    return phase_summary


__all__ = [
    "PhaseSummary",
    "detect_phases",
    "summarize_phases",
]
