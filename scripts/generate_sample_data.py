"""Synthetic Knallgas batch reactor data for hydrogen-oxidizing bacteria."""

from pathlib import Path

import numpy as np
import pandas as pd



# Time and operating conditions
START_HOUR = 0.0
END_HOUR = 72.0
TIME_STEP_H = 0.5


# Initial concentrations and stoichiometry
INITIAL_BIOMASS_G_L = 0.08          # g/L cell dry weight
INITIAL_OD600 = 0.08                # absorbance at 600 nm
INITIAL_NH3_N_G_L = 2.5             # g/L as ammonia-nitrogen
INITIAL_PO4_G_L = 1.5               # g/L phosphate

BIOMASS_NITROGEN_FRACTION = 0.12    # g N / g biomass
PHOSPHATE_TO_BIOMASS_RATIO = 0.02   # g PO4 / g biomass
MU_MAX_1_PER_H = 0.20               # maximum specific growth rate
NITROGEN_HALF_SATURATION_G_L = 0.25 # Monod half-saturation for NH3-N
H2_YIELD_G_PER_G = 1.5              # g biomass per g H2 consumed (approximate)

# Deterministic RNG for reproducible noise.
RNG = np.random.default_rng(42)


def generate_time_hours():
    """Return the hourly time vector for the full run."""
    return np.arange(START_HOUR, END_HOUR + TIME_STEP_H, TIME_STEP_H)


def simulate_bioprocess(time_h):
    """Simulate biomass, NH3-N, and phosphate with lag, growth, and nitrogen-limited stationary phase."""
    biomass = np.zeros_like(time_h, dtype=float)
    nh3_n = np.zeros_like(time_h, dtype=float)
    phosphate = np.zeros_like(time_h, dtype=float)
    mu = np.zeros_like(time_h, dtype=float)

    biomass[0] = INITIAL_BIOMASS_G_L
    nh3_n[0] = INITIAL_NH3_N_G_L
    phosphate[0] = INITIAL_PO4_G_L

    for i in range(1, len(time_h)):
        previous_biomass = biomass[i - 1]
        previous_nh3 = nh3_n[i - 1]

        if time_h[i] < 8.0:
            # Lag phase: almost flat, cells adapt rather than growing fast.
            mu[i] = 0.02
        elif previous_nh3 <= 0.05:
            # Nitrogen depletion triggers the stationary phase.
            mu[i] = 0.0
        else:
            nitrogen_limitation = previous_nh3 / (NITROGEN_HALF_SATURATION_G_L + previous_nh3)
            mu[i] = MU_MAX_1_PER_H * nitrogen_limitation

        # Slight decay after the stationary period to add realism.
        if time_h[i] > 60.0:
            mu[i] = min(mu[i], 0.01)

        biomass_step = previous_biomass * np.exp(mu[i] * TIME_STEP_H)
        biomass[i] = max(0.0, biomass_step)

        if previous_nh3 <= 0.05 and time_h[i] >= 50.0:
            biomass[i] = max(previous_biomass, biomass[i] * 0.995)

        biomass_delta = max(0.0, biomass[i] - previous_biomass)
        nh3_n[i] = max(0.0, previous_nh3 - BIOMASS_NITROGEN_FRACTION * biomass_delta)
        phosphate[i] = max(0.0, phosphate[i - 1] - PHOSPHATE_TO_BIOMASS_RATIO * biomass_delta)

    return biomass, nh3_n, phosphate, mu


def simulate_gas_uptake(time_h, biomass, growth_rate):
    """Model H2, CO2, and O2 uptake rates as functions of biomass and growth rate."""
    h2_uptake = np.zeros_like(time_h, dtype=float)
    co2_uptake = np.zeros_like(time_h, dtype=float)
    o2_uptake = np.zeros_like(time_h, dtype=float)

    for i, hour in enumerate(time_h):
        if biomass[i] <= 0.0:
            growth_factor = 0.0
        else:
            growth_factor = np.clip(growth_rate[i] / MU_MAX_1_PER_H, 0.0, 1.0)

        # H2 uptake is the dominant gas demand and scales strongly with active growth.
        h2_uptake[i] = max(0.0, 24.0 * biomass[i] * growth_factor + 0.3)

        # Approximate respiratory stoichiometry: CO2 and O2 consumption scale below H2.
        co2_uptake[i] = h2_uptake[i] / 6.0
        o2_uptake[i] = h2_uptake[i] / 3.0

        # When growth slows, gas demand drops substantially as cells enter lag or stationary phase.
        if hour < 8.0 or hour > 50.0:
            h2_uptake[i] *= 0.25
            co2_uptake[i] *= 0.25
            o2_uptake[i] *= 0.25

    return h2_uptake, co2_uptake, o2_uptake


def simulate_offgas(time_h, h2_uptake, co2_uptake):
    """Off-gas composition changes inversely with gas uptake during growth and recovery in slow phases."""
    utilization = np.clip(h2_uptake / 30.0, 0.0, 1.0)

    offgas_h2 = 72.0 - 40.0 * utilization + 1.2 * np.sin(2.0 * np.pi * time_h / 15.0)
    offgas_o2 = 22.0 - 18.0 * utilization + 0.8 * np.sin(2.0 * np.pi * time_h / 11.0)
    offgas_co2 = 4.0 + 2.5 * utilization + 0.5 * np.sin(2.0 * np.pi * time_h / 18.0)

    # Data should stay within physically realistic gas composition bounds.
    offgas_h2 = np.clip(offgas_h2, 0.0, 100.0)
    offgas_o2 = np.clip(offgas_o2, 0.0, 100.0)
    offgas_co2 = np.clip(offgas_co2, 0.0, 30.0)

    return offgas_h2, offgas_co2, offgas_o2


def simulate_do_and_ph(time_h, growth_rate):
    """DO stays very low during gas-transfer-limited growth; pH is near neutral with a brief dip."""
    growth_fraction = np.clip(growth_rate / MU_MAX_1_PER_H, 0.0, 1.0)

    do_percent = 4.0 + 22.0 * (1.0 - growth_fraction) + 1.5 * np.sin(2.0 * np.pi * time_h / 12.0)
    do_percent = np.clip(do_percent, 0.0, 60.0)

    ph = 7.0 + 0.10 * np.sin(2.0 * np.pi * time_h / 18.0)
    ph -= 0.85 * np.exp(-((time_h - 40.0) / 2.0) ** 2)
    ph = np.clip(ph, 5.8, 7.5)

    return do_percent, ph


def simulate_process_controls(time_h):
    """Temperature, agitation, and pressure mimic controlled bioreactor operation."""
    temperature_c = 30.0 + 0.25 * np.sin(2.0 * np.pi * time_h / 16.0)
    temperature_c += 2.8 * np.exp(-0.5 * ((time_h - 59.0) / 1.4) ** 2)
    temperature_c = np.clip(temperature_c, 28.0, 36.0)

    agitation_rpm = 170.0 + 130.0 / (1.0 + np.exp(-(time_h - 8.0) / 6.0))
    agitation_rpm += 10.0 * np.sin(2.0 * np.pi * time_h / 10.0)
    agitation_rpm = np.clip(agitation_rpm, 120.0, 400.0)

    pressure_bar = 1.2 + 0.04 * np.sin(2.0 * np.pi * time_h / 20.0)
    pressure_bar = np.clip(pressure_bar, 0.9, 1.6)

    return temperature_c, agitation_rpm, pressure_bar


def add_gaussian_noise(data_frame):
    """Add small Gaussian noise to every measurement column for realistic process variability."""
    noise_levels = {
        "biomass_gL": 0.05,
        "OD600": 0.04,
        "NH3_N_gL": 0.05,
        "PO4_gL": 0.04,
        "H2_uptake_mmol_L_h": 0.8,
        "CO2_uptake_mmol_L_h": 0.5,
        "O2_uptake_mmol_L_h": 0.8,
        "offgas_H2_pct": 1.2,
        "offgas_CO2_pct": 0.5,
        "offgas_O2_pct": 1.0,
        "DO_percent": 1.0,
        "pH": 0.04,
        "temperature_C": 0.12,
        "agitation_rpm": 5.0,
        "pressure_bar": 0.02,
    }

    for column_name, sigma in noise_levels.items():
        if column_name in data_frame.columns:
            data_frame[column_name] = data_frame[column_name] + RNG.normal(0.0, sigma, size=len(data_frame))

    # Concentrations must remain physically valid after noise addition.
    data_frame["biomass_gL"] = np.clip(data_frame["biomass_gL"], 0.0, None)
    data_frame["OD600"] = np.clip(data_frame["OD600"], 0.0, None)
    data_frame["NH3_N_gL"] = np.clip(data_frame["NH3_N_gL"], 0.0, None)
    data_frame["PO4_gL"] = np.clip(data_frame["PO4_gL"], 0.0, None)

    data_frame["H2_uptake_mmol_L_h"] = np.clip(data_frame["H2_uptake_mmol_L_h"], 0.0, None)
    data_frame["CO2_uptake_mmol_L_h"] = np.clip(data_frame["CO2_uptake_mmol_L_h"], 0.0, None)
    data_frame["O2_uptake_mmol_L_h"] = np.clip(data_frame["O2_uptake_mmol_L_h"], 0.0, None)
    data_frame["offgas_H2_pct"] = np.clip(data_frame["offgas_H2_pct"], 0.0, 100.0)
    data_frame["offgas_CO2_pct"] = np.clip(data_frame["offgas_CO2_pct"], 0.0, 30.0)
    data_frame["offgas_O2_pct"] = np.clip(data_frame["offgas_O2_pct"], 0.0, 100.0)
    data_frame["DO_percent"] = np.clip(data_frame["DO_percent"], 0.0, 100.0)
    data_frame["pH"] = np.clip(data_frame["pH"], 5.8, 7.5)
    data_frame["temperature_C"] = np.clip(data_frame["temperature_C"], 28.0, 36.0)
    data_frame["agitation_rpm"] = np.clip(data_frame["agitation_rpm"], 120.0, 420.0)
    data_frame["pressure_bar"] = np.clip(data_frame["pressure_bar"], 0.9, 1.6)

    return data_frame


def main():
    """Generate and save the synthetic Knallgas bioreactor dataset."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    time_h = generate_time_hours()
    biomass, nh3_n, phosphate, growth_rate = simulate_bioprocess(time_h)
    h2_uptake, co2_uptake, o2_uptake = simulate_gas_uptake(time_h, biomass, growth_rate)
    offgas_h2, offgas_co2, offgas_o2 = simulate_offgas(time_h, h2_uptake, co2_uptake)
    do_percent, ph = simulate_do_and_ph(time_h, growth_rate)
    temperature_c, agitation_rpm, pressure_bar = simulate_process_controls(time_h)

    od600 = 0.8 * biomass + 0.03

    df = pd.DataFrame(
        {
            "time_h": time_h,
            "biomass_gL": biomass,
            "OD600": od600,
            "NH3_N_gL": nh3_n,
            "PO4_gL": phosphate,
            "H2_uptake_mmol_L_h": h2_uptake,
            "CO2_uptake_mmol_L_h": co2_uptake,
            "O2_uptake_mmol_L_h": o2_uptake,
            "offgas_H2_pct": offgas_h2,
            "offgas_CO2_pct": offgas_co2,
            "offgas_O2_pct": offgas_o2,
            "DO_percent": do_percent,
            "pH": ph,
            "temperature_C": temperature_c,
            "agitation_rpm": agitation_rpm,
            "pressure_bar": pressure_bar,
        }
    )

    df = add_gaussian_noise(df)
    output_path = data_dir / "sample_run.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved Knallgas batch run to {output_path}")


if __name__ == "__main__":
    main()

