"""Batch hydrogen-fermentation kinetics utilities for the sample bioreactor CSV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


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

H2_MOLAR_MASS_G_PER_MOL = 2.016
CO2_MOLAR_MASS_G_PER_MOL = 44.01
O2_MOLAR_MASS_G_PER_MOL = 32.00
EPSILON = 1e-9


@dataclass(frozen=True)
class KineticsSummary:
    """Headline batch-fermentation metrics in standard units."""

    mu_max_1_per_h: float
    mu_max_time_h: float
    doubling_time_h: float
    yield_h2_g_per_g: float
    yield_co2_g_per_g: float
    yield_nitrogen_g_per_g: float
    productivity_g_L_h: float
    peak_productivity_g_L_h: float
    peak_productivity_time_h: float
    peak_h2_uptake_mmol_L_h: float
    peak_co2_uptake_mmol_L_h: float
    peak_o2_uptake_mmol_L_h: float
    final_biomass_g_L: float


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, time-sorted copy with required columns and numeric data."""
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


def specific_growth_rate_series(df: pd.DataFrame, window_h: float = 4.0) -> pd.Series:
    """Rolling log-linear growth rate mu (1/h) for each time point using a time window."""
    if window_h <= 0:
        raise ValueError("window_h must be positive")

    working = _prepare_df(df)
    if working.empty:
        return pd.Series(dtype=float, name="mu_1_per_h")

    time_h = working[TIME_H].to_numpy(dtype=float)
    biomass = working[BIOMASS_G_L].to_numpy(dtype=float)
    biomass = np.clip(biomass, EPSILON, None)
    log_biomass = np.log(biomass)
    mu_values = np.full(len(working), np.nan, dtype=float)

    for i in range(len(working)):
        time_start = time_h[i] - window_h
        mask = (time_h >= time_start) & (time_h <= time_h[i])
        if mask.sum() < 2:
            continue

        x = time_h[mask]
        y = log_biomass[mask]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue

        slope = linregress(x, y).slope
        if np.isfinite(slope):
            mu_values[i] = float(slope)

    return pd.Series(mu_values, index=working.index, name="mu_1_per_h")


def max_specific_growth_rate(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (mu_max_1_per_h, time_h_of_mu_max, doubling_time_h)."""
    mu_series = specific_growth_rate_series(df)
    valid_mu = mu_series.dropna()
    if valid_mu.empty:
        return 0.0, float("nan"), float("nan")

    mu_max = float(valid_mu.max())
    time_h = float(_prepare_df(df).loc[valid_mu.idxmax(), TIME_H])
    doubling_time_h = float(np.log(2.0) / mu_max) if mu_max > 0 else float("nan")
    return mu_max, time_h, doubling_time_h


def mu_in_window(df: pd.DataFrame, t_start: float, t_end: float) -> Tuple[float, float]:
    """Return (mu_1_per_h, R_squared) for biomass growth in a time window."""
    if t_end <= t_start:
        return 0.0, 0.0

    working = _prepare_df(df)
    mask = (working[TIME_H] >= t_start) & (working[TIME_H] <= t_end)
    subset = working.loc[mask].copy()
    if len(subset) < 2:
        return 0.0, 0.0

    time_h = subset[TIME_H].to_numpy(dtype=float)
    biomass = np.clip(subset[BIOMASS_G_L].to_numpy(dtype=float), EPSILON, None)
    log_biomass = np.log(biomass)
    if not np.all(np.isfinite(log_biomass)):
        return 0.0, 0.0

    result = linregress(time_h, log_biomass)
    mu = float(result.slope) if np.isfinite(result.slope) else 0.0
    r_squared = float(result.rvalue ** 2) if np.isfinite(result.rvalue) else 0.0
    return mu, r_squared


def gas_uptake_totals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Integrate gas uptake rates over time; return mmol/L and g/L totals."""
    working = _prepare_df(df)
    if working.empty:
        return {
            "H2": {"mmol_L": 0.0, "g_L": 0.0},
            "CO2": {"mmol_L": 0.0, "g_L": 0.0},
            "O2": {"mmol_L": 0.0, "g_L": 0.0},
        }

    time_h = working[TIME_H].to_numpy(dtype=float)
    totals: Dict[str, Dict[str, float]] = {}
    for label, column, molar_mass in [
        ("H2", H2_UPTAKE_MMOL_L_H, H2_MOLAR_MASS_G_PER_MOL),
        ("CO2", CO2_UPTAKE_MMOL_L_H, CO2_MOLAR_MASS_G_PER_MOL),
        ("O2", O2_UPTAKE_MMOL_L_H, O2_MOLAR_MASS_G_PER_MOL),
    ]:
        rates = pd.to_numeric(working[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total_mmol_L = float(np.trapezoid(rates, time_h))
        total_g_L = total_mmol_L * molar_mass / 1000.0
        totals[label] = {"mmol_L": total_mmol_L, "g_L": total_g_L}

    return totals


def biomass_yield_on_h2(df: pd.DataFrame, t_start: float, t_end: float) -> Dict[str, float]:
    """Return biomass yield on H2, CO2, and ammonia-nitrogen for a time window in g/g."""
    if t_end <= t_start:
        return {
            "yield_h2_g_per_g": 0.0,
            "yield_co2_g_per_g": 0.0,
            "yield_nitrogen_g_per_g": 0.0,
        }

    working = _prepare_df(df)
    mask = (working[TIME_H] >= t_start) & (working[TIME_H] <= t_end)
    subset = working.loc[mask].copy()
    if subset.empty:
        return {
            "yield_h2_g_per_g": 0.0,
            "yield_co2_g_per_g": 0.0,
            "yield_nitrogen_g_per_g": 0.0,
        }

    x_start = float(subset[BIOMASS_G_L].iloc[0])
    x_end = float(subset[BIOMASS_G_L].iloc[-1])
    delta_biomass = x_end - x_start

    gas_totals = gas_uptake_totals(subset)
    h2_consumed_g_L = gas_totals["H2"]["g_L"]
    co2_consumed_g_L = gas_totals["CO2"]["g_L"]

    n_start = float(subset[NH3_N_G_L].iloc[0])
    n_end = float(subset[NH3_N_G_L].iloc[-1])
    nitrogen_consumed_g_L = max(0.0, n_start - n_end)

    yield_h2 = delta_biomass / h2_consumed_g_L if h2_consumed_g_L > 0 else 0.0
    yield_co2 = delta_biomass / co2_consumed_g_L if co2_consumed_g_L > 0 else 0.0
    yield_nitrogen = delta_biomass / nitrogen_consumed_g_L if nitrogen_consumed_g_L > 0 else 0.0

    return {
        "yield_h2_g_per_g": float(yield_h2),
        "yield_co2_g_per_g": float(yield_co2),
        "yield_nitrogen_g_per_g": float(yield_nitrogen),
    }


def volumetric_productivity(df: pd.DataFrame, t_start: float, t_end: float) -> Dict[str, float]:
    """Return volumetric productivity in g/L/h and the peak smoothed dX/dt."""
    if t_end <= t_start:
        return {
            "productivity_g_L_h": 0.0,
            "peak_productivity_g_L_h": 0.0,
            "peak_productivity_time_h": float("nan"),
        }

    working = _prepare_df(df)
    mask = (working[TIME_H] >= t_start) & (working[TIME_H] <= t_end)
    subset = working.loc[mask].copy()
    if subset.empty:
        return {
            "productivity_g_L_h": 0.0,
            "peak_productivity_g_L_h": 0.0,
            "peak_productivity_time_h": float("nan"),
        }

    x_start = float(subset[BIOMASS_G_L].iloc[0])
    x_end = float(subset[BIOMASS_G_L].iloc[-1])
    delta_t_h = float(t_end - t_start)
    productivity = (x_end - x_start) / delta_t_h if delta_t_h > 0 else 0.0

    time_h = subset[TIME_H].to_numpy(dtype=float)
    biomass = subset[BIOMASS_G_L].to_numpy(dtype=float)
    window = max(3, min(9, len(subset)))
    smoothed = pd.Series(biomass).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    dxd_t = np.gradient(smoothed, time_h)
    finite = np.isfinite(dxd_t)
    if not np.any(finite):
        peak_productivity = 0.0
        peak_time_h = float("nan")
    else:
        peak_index = int(np.nanargmax(dxd_t[finite])) if np.any(finite) else 0
        peak_values = dxd_t[finite]
        peak_productivity = float(np.max(peak_values))
        peak_time_h = float(time_h[np.flatnonzero(finite)[peak_index]])

    return {
        "productivity_g_L_h": float(productivity),
        "peak_productivity_g_L_h": float(peak_productivity),
        "peak_productivity_time_h": peak_time_h,
    }


def specific_uptake_rates(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return specific uptake rates q_H2, q_CO2, q_O2 in mmol/g/h."""
    working = _prepare_df(df)
    biomass = working[BIOMASS_G_L].clip(lower=EPSILON)

    uptake_rates = {
        "q_H2_mmol_g_h": working[H2_UPTAKE_MMOL_L_H] / biomass,
        "q_CO2_mmol_g_h": working[CO2_UPTAKE_MMOL_L_H] / biomass,
        "q_O2_mmol_g_h": working[O2_UPTAKE_MMOL_L_H] / biomass,
    }
    return uptake_rates


def summarize_run(df: pd.DataFrame) -> KineticsSummary:
    """Summarize the batch run with the headline kinetic metrics."""
    working = _prepare_df(df)
    if working.empty:
        return KineticsSummary(
            mu_max_1_per_h=0.0,
            mu_max_time_h=float("nan"),
            doubling_time_h=float("nan"),
            yield_h2_g_per_g=0.0,
            yield_co2_g_per_g=0.0,
            yield_nitrogen_g_per_g=0.0,
            productivity_g_L_h=0.0,
            peak_productivity_g_L_h=0.0,
            peak_productivity_time_h=float("nan"),
            peak_h2_uptake_mmol_L_h=0.0,
            peak_co2_uptake_mmol_L_h=0.0,
            peak_o2_uptake_mmol_L_h=0.0,
            final_biomass_g_L=0.0,
        )

    mu_max, mu_time_h, doubling_time_h = max_specific_growth_rate(working)

    t_start = float(working[TIME_H].iloc[0])
    t_end = float(working[TIME_H].iloc[-1])
    overall_yield = biomass_yield_on_h2(working, t_start, t_end)
    productivity = volumetric_productivity(working, t_start, t_end)

    peak_h2 = float(working[H2_UPTAKE_MMOL_L_H].max())
    peak_co2 = float(working[CO2_UPTAKE_MMOL_L_H].max())
    peak_o2 = float(working[O2_UPTAKE_MMOL_L_H].max())
    final_biomass = float(working[BIOMASS_G_L].iloc[-1])

    return KineticsSummary(
        mu_max_1_per_h=float(mu_max),
        mu_max_time_h=float(mu_time_h),
        doubling_time_h=float(doubling_time_h),
        yield_h2_g_per_g=float(overall_yield["yield_h2_g_per_g"]),
        yield_co2_g_per_g=float(overall_yield["yield_co2_g_per_g"]),
        yield_nitrogen_g_per_g=float(overall_yield["yield_nitrogen_g_per_g"]),
        productivity_g_L_h=float(productivity["productivity_g_L_h"]),
        peak_productivity_g_L_h=float(productivity["peak_productivity_g_L_h"]),
        peak_productivity_time_h=float(productivity["peak_productivity_time_h"]),
        peak_h2_uptake_mmol_L_h=peak_h2,
        peak_co2_uptake_mmol_L_h=peak_co2,
        peak_o2_uptake_mmol_L_h=peak_o2,
        final_biomass_g_L=final_biomass,
    )


__all__ = [
    "KineticsSummary",
    "specific_growth_rate_series",
    "max_specific_growth_rate",
    "mu_in_window",
    "gas_uptake_totals",
    "biomass_yield_on_h2",
    "volumetric_productivity",
    "specific_uptake_rates",
    "summarize_run",
]
