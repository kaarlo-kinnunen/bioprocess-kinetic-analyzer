"""Script to visualize a sample run of the sample data generator."""

import pandas as pd
import matplotlib.pyplot as plt


# Load once and use the dataframe directly for plotting.
df = pd.read_csv('/Users/kallekinnunen/Documents/bioprocess-kinetic-analyzer/data/sample_run.csv')

# Group variables by meaning so each subplot is readable without crowded labels.
biomass_group = ['biomass_gL', 'OD600']
substrate_group = ['NH3_N_gL', 'PO4_gL']
gas_group = ['H2_uptake_mmol_L_h', 'CO2_uptake_mmol_L_h', 'O2_uptake_mmol_L_h']
offgas_group = ['offgas_H2_pct', 'offgas_CO2_pct', 'offgas_O2_pct']
process_group = ['DO_percent', 'pH', 'temperature_C', 'agitation_rpm', 'pressure_bar']

all_groups = [biomass_group, substrate_group, gas_group, offgas_group, process_group]

fig, axes = plt.subplots(len(all_groups), 1, figsize=(12, 14), sharex=True)
fig.suptitle('Bioreactor time series', fontsize=14, y=0.98)

for ax, group in zip(axes, all_groups):
    for column in group:
        ax.plot(df['time_h'], df[column], linewidth=1.6)
    ax.set_ylabel('value')
    ax.grid(True, alpha=0.3)
    ax.legend(group, loc='upper right', fontsize='small')
    ax.set_xticks(df['time_h'].unique()[::2])
    ax.tick_params(axis='x', rotation=45)

axes[-1].set_xlabel('time_h (h)')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()