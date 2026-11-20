# Bioprocess Kinetic Analyzer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Kinetic analysis and executive PDF reports for Knallgas (aerobic hydrogen-oxidizing) gas-fermentation bioreactor runs.

## Why this exists

Bioreactor runs produce large CSV logs of biomass, nutrient levels, gas flows and process conditions. Scientists often spend hours parsing these files by hand to get basic growth metrics. This tool loads a run, validates the data, calculates the key kinetic metrics, detects the fermentation phases and builds a PDF report automatically.

The process context is hydrogen-oxidizing bacteria growing on H2 (energy), CO2 (carbon) and O2 (electron acceptor) in a mineral medium, with ammonia as the nitrogen source. There is no sugar, and the product is the biomass itself (single-cell protein), so there is no product titer.

## Features

- Specific growth rate (mu) and doubling time
- Biomass yield on H2, CO2 and nitrogen
- Volumetric productivity (g/L/h), overall and peak
- Gas uptake rates (H2, CO2, O2), totals and specific rates
- Phase detection: lag, exponential, transition, stationary
- Cross-check of nitrogen depletion against the end of the exponential phase
- Critical process parameter (CPP) excursion alerts for pH, temperature and pressure
- CSV validation with clear error messages
- Executive-ready PDF report with metric tables and charts

## Example output

- [Example report (PDF)](docs/example_report.pdf)


## Quick start

```bash
git clone https://github.com/kaarlo-kinnunen/bioprocess-kinetic-analyzer.git
cd bioprocess-kinetic-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generate a report from the sample data:

```bash
bioprocess data/sample_run.csv --out output/report.pdf --run-id RUN-001
```

Run the tests:

```bash
pytest
```

To regenerate the simulated sample data:

```bash
python scripts/generate_sample_data.py
```

## Input data format

One row per time point, as a CSV file. Required columns must be present, optional columns enable the corresponding analyses and charts.

| Column | Unit | Required |
|---|---|---|
| `time_h` | h | yes |
| `biomass_gL` | g/L (cell dry weight) | yes |
| `NH3_N_gL` | g/L (nitrogen) | yes |
| `OD600` | - | no |
| `PO4_gL` | g/L | no |
| `H2_uptake_mmol_L_h` | mmol/L/h | no |
| `CO2_uptake_mmol_L_h` | mmol/L/h | no |
| `O2_uptake_mmol_L_h` | mmol/L/h | no |
| `offgas_H2_pct`, `offgas_CO2_pct`, `offgas_O2_pct` | % | no |
| `DO_percent` | % | no |
| `pH` | - | no |
| `temperature_C` | deg C | no |
| `agitation_rpm` | rpm | no |
| `pressure_bar` | bar | no |

The loader sorts by time, drops duplicate timestamps, interpolates short gaps (up to 3 points), and flags physically implausible values as missing instead of deleting rows.

## How it works

**Specific growth rate.** mu is the slope of ln(biomass) against time, computed with a rolling linear regression (default window 4 h). Doubling time is ln(2) / mu.

**Yields and productivity.** Gas uptake rates are integrated over time (trapezoid rule) and converted to g/L with molar masses. Biomass yield is the biomass gained in a time window divided by the H2 (or CO2, or nitrogen) consumed in that window. Volumetric productivity is the biomass gained per liter and hour.

**Phase detection.** The smoothed mu curve is compared with its maximum:
- lag: before mu first reaches 20% of mu_max
- exponential: until mu falls below 50% of mu_max after the peak
- stationary: mu below 10% of mu_max after the exponential phase
- transition: anything in between

A minimum phase duration (default 2 h) prevents flickering between phases from noise. The report also checks whether nitrogen depletion falls close to the end of the exponential phase, which is the expected growth-limiting event in the sample data.

**CPP alerts.** Out-of-range stretches are grouped into excursions with start, end, duration and extreme value. Default limits: pH 6.5-7.5, temperature 28-33 deg C, pressure 1.0-1.5 bar. Excursions shorter than 1 h are warnings, longer ones are critical.

## Project structure

```
bioprocess-kinetic-analyzer/
├── src/bioprocess/
│   ├── loader.py      # CSV reading and validation
│   ├── kinetics.py    # growth rate, yields, productivity, uptake rates
│   ├── phases.py      # lag / exponential / stationary detection
│   ├── plots.py       # matplotlib charts
│   ├── report.py      # ReportLab PDF builder, CPP alerts
│   └── cli.py         # command-line entry point
├── scripts/
│   └── generate_sample_data.py   # simulated batch run
├── data/
│   ├── sample_run.csv
│   └── visualize_sample_run.py
├── tests/
├── docs/              # example report
├── output/            # generated reports (gitignored)
├── pyproject.toml
└── README.md
```

## Assumptions and limitations

- The sample data is simulated. No real production data is used. Inspiration for sample data generation acquired from [1].
- Yields and gas ratios in the simulator are plausible ballparks, not verified values.
- Batch mode only. Real production is batch-fed, so continuous mode (dilution rate, steady state) is a possible future feature, along with regression models to predict culture response to changes in the environment.
- Phase thresholds and CPP limits are defaults and should be tuned for a real process.

## References

1. Jesenofsky, V., Gescher, J., Edel, M. (2026). Limits of Knallgas bacteria for autotrophic CO₂ conversion: physiology, H₂ kinetics and reactor constraints. *Current Opinion in Chemical Engineering*, 54, 101302. [https://doi.org/10.1016/j.coche.2026.101302](https://doi.org/10.1016/j.coche.2026.101302)

## Roadmap

- Continuous mode (dilution rate, steady-state detection)
- Import of real-world data formats
- More CPP rules and configurable limits via a file
- Continuous integration with automated tests
- Supervised learning models trained to predict culture response to changes in the environment

## License

MIT
