from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bioprocess.kinetics import (
    BIOMASS_G_L,
    CO2_UPTAKE_MMOL_L_H,
    H2_UPTAKE_MMOL_L_H,
    NH3_N_G_L,
    O2_UPTAKE_MMOL_L_H,
    TIME_H,
    gas_uptake_totals,
    max_specific_growth_rate,
    mu_in_window,
    specific_growth_rate_series,
)


def _make_exponential_growth_df():
    """Small synthetic batch run with known exponential growth rate."""
    mu_expected = 0.20
    x0 = 0.5
    time_h = np.linspace(0.0, 20.0, 401)
    biomass = x0 * np.exp(mu_expected * time_h)

    df = pd.DataFrame(
        {
            TIME_H: time_h,
            BIOMASS_G_L: biomass,
            "OD600": 0.8 * biomass + 0.05,
            NH3_N_G_L: 2.5 - 0.02 * time_h,
            "PO4_gL": 1.5 - 0.015 * time_h,
            H2_UPTAKE_MMOL_L_H: 12.0 * np.ones_like(time_h),
            CO2_UPTAKE_MMOL_L_H: 2.0 * np.ones_like(time_h),
            O2_UPTAKE_MMOL_L_H: 4.0 * np.ones_like(time_h),
            "offgas_H2_pct": 65.0 * np.ones_like(time_h),
            "offgas_CO2_pct": 5.0 * np.ones_like(time_h),
            "offgas_O2_pct": 20.0 * np.ones_like(time_h),
            "DO_percent": 20.0 * np.ones_like(time_h),
            "pH": 7.0 * np.ones_like(time_h),
            "temperature_C": 30.0 * np.ones_like(time_h),
            "agitation_rpm": 180.0 * np.ones_like(time_h),
            "pressure_bar": 1.2 * np.ones_like(time_h),
        }
    )
    return df, mu_expected


@pytest.fixture
def exponential_growth_df():
    """Pytest fixture for the synthetic exponential batch data."""
    return _make_exponential_growth_df()


def test_specific_growth_rate_recovers_known_mu(exponential_growth_df):
    """Rolling log-linear regression should recover the known exponential mu within 1%."""
    df, mu_expected = exponential_growth_df

    mu_series = specific_growth_rate_series(df, window_h=4.0)
    recovered_mu = float(mu_series.dropna().median())
    assert np.isfinite(recovered_mu)
    assert abs(recovered_mu - mu_expected) / mu_expected < 0.01

    mu_max, _, _ = max_specific_growth_rate(df)
    assert np.isfinite(mu_max)
    assert abs(mu_max - mu_expected) / mu_expected < 0.01

    mu_window, r2 = mu_in_window(df, 0.0, 20.0)
    assert np.isfinite(mu_window)
    assert np.isfinite(r2)
    assert abs(mu_window - mu_expected) / mu_expected < 0.01


def test_constant_gas_uptake_totals_are_correct(exponential_growth_df):
    """Trapezoid integration should match constant uptake rate totals exactly."""
    df, _ = exponential_growth_df
    t_start = float(df[TIME_H].iloc[0])
    t_end = float(df[TIME_H].iloc[-1])
    duration_h = t_end - t_start

    totals = gas_uptake_totals(df)

    expected_h2_mmol_l = 12.0 * duration_h
    expected_co2_mmol_l = 2.0 * duration_h
    expected_o2_mmol_l = 4.0 * duration_h

    assert np.isclose(totals["H2"]["mmol_L"], expected_h2_mmol_l, rtol=1e-12, atol=1e-9)
    assert np.isclose(totals["CO2"]["mmol_L"], expected_co2_mmol_l, rtol=1e-12, atol=1e-9)
    assert np.isclose(totals["O2"]["mmol_L"], expected_o2_mmol_l, rtol=1e-12, atol=1e-9)

    expected_h2_g_l = expected_h2_mmol_l * 2.016 / 1000.0
    expected_co2_g_l = expected_co2_mmol_l * 44.01 / 1000.0
    expected_o2_g_l = expected_o2_mmol_l * 32.00 / 1000.0

    assert np.isclose(totals["H2"]["g_L"], expected_h2_g_l, rtol=1e-12, atol=1e-12)
    assert np.isclose(totals["CO2"]["g_L"], expected_co2_g_l, rtol=1e-12, atol=1e-12)
    assert np.isclose(totals["O2"]["g_L"], expected_o2_g_l, rtol=1e-12, atol=1e-12)


def test_zero_biomass_edge_case_is_safe():
    """Zero biomass must not cause log(0), divide-by-zero, or NaN propagation."""
    time_h = np.linspace(0.0, 10.0, 41)
    df = pd.DataFrame(
        {
            TIME_H: time_h,
            BIOMASS_G_L: np.zeros_like(time_h),
            "OD600": np.zeros_like(time_h),
            NH3_N_G_L: np.linspace(2.5, 0.0, len(time_h)),
            "PO4_gL": np.linspace(1.5, 0.0, len(time_h)),
            H2_UPTAKE_MMOL_L_H: np.zeros_like(time_h),
            CO2_UPTAKE_MMOL_L_H: np.zeros_like(time_h),
            O2_UPTAKE_MMOL_L_H: np.zeros_like(time_h),
            "offgas_H2_pct": np.zeros_like(time_h),
            "offgas_CO2_pct": np.zeros_like(time_h),
            "offgas_O2_pct": np.zeros_like(time_h),
            "DO_percent": np.zeros_like(time_h),
            "pH": 7.0 * np.ones_like(time_h),
            "temperature_C": 30.0 * np.ones_like(time_h),
            "agitation_rpm": 180.0 * np.ones_like(time_h),
            "pressure_bar": 1.2 * np.ones_like(time_h),
        }
    )

    mu_series = specific_growth_rate_series(df, window_h=4.0)
    assert np.all(np.isfinite(mu_series.fillna(0.0)))

    mu_max, time_h_of_max, doubling_time_h = max_specific_growth_rate(df)
    assert mu_max == 0.0
    assert np.isfinite(time_h_of_max)
    assert np.isnan(doubling_time_h)

    mu_window, r2 = mu_in_window(df, 0.0, 10.0)
    assert mu_window == 0.0
    assert np.isfinite(r2)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
