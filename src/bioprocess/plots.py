"""Report-ready plotting helpers for the batch hydrogen-fermentation dataset."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .kinetics import specific_growth_rate_series

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
}

PHASE_COLORS: Dict[str, str] = {
    "lag": "#d9edf7",
    "exponential": "#dff0d8",
    "transition": "#fcf8e3",
    "stationary": "#f2dede",
}

FIGSIZE = (7.0, 3.5)
DPI = 200


def _prepare_df(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Return a cleaned, non-mutating DataFrame subset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    working = df.copy()
    for column in required:
        if column not in working.columns:
            raise KeyError(f"Missing required column: {column}")

    working[TIME_H] = pd.to_numeric(working[TIME_H], errors="coerce")
    for column in [column for column in working.columns if column != TIME_H]:
        if np.issubdtype(working[column].dtype, np.number):
            continue
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.dropna(subset=[TIME_H]).sort_values(TIME_H).reset_index(drop=True)
    return working


def _save_and_close(fig: plt.Figure, out_path: str | Path) -> Path:
    """Save a figure as PNG and close it, returning the output path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, format="png", bbox_inches="tight")
    plt.close(fig)
    return path


def _apply_style(ax: plt.Axes, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    """Apply the project’s plotting style."""
    ax.grid(True, which="major", linewidth=0.6, alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#444444")
    ax.tick_params(labelsize=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9)


def _phase_overlay(ax: plt.Axes, time_h: pd.Series, labels: Optional[pd.Series]) -> list:
    """Shade optional phase bands and return any phase legend handles."""
    if labels is None:
        return []

    if not isinstance(labels, pd.Series):
        warnings.warn("Ignoring phase labels: expected pandas Series.")
        return []

    if len(labels) != len(time_h):
        warnings.warn("Ignoring phase labels: label length does not match time series length.")
        return []

    phase_series = labels.reset_index(drop=True).astype(str)
    plot_order = [name for name in ["lag", "exponential", "transition", "stationary"] if name in phase_series.unique()]
    if not plot_order:
        return []

    last_idx = 0
    last_phase = phase_series.iloc[0]
    for idx in range(1, len(phase_series)):
        current_phase = phase_series.iloc[idx]
        if current_phase != last_phase:
            start_x = float(time_h.iloc[last_idx])
            end_x = float(time_h.iloc[idx - 1])
            if last_phase in PHASE_COLORS:
                ax.axvspan(start_x, end_x, color=PHASE_COLORS[last_phase], alpha=0.18, zorder=0)
            last_idx = idx
            last_phase = current_phase

    start_x = float(time_h.iloc[last_idx])
    end_x = float(time_h.iloc[-1])
    if last_phase in PHASE_COLORS:
        ax.axvspan(start_x, end_x, color=PHASE_COLORS[last_phase], alpha=0.18, zorder=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_COLORS[phase], edgecolor="none", label=phase)
        for phase in plot_order
    ]
    return handles


def plot_growth(df: pd.DataFrame, labels: Optional[pd.Series] = None, out_path: str | Path = "growth.png", *, log_scale: bool = False) -> Path:
    """Plot biomass and OD600 over time."""
    try:
        working = _prepare_df(df, [TIME_H, BIOMASS_G_L, OD600])
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_growth: missing required columns in input DataFrame.")
        return Path(out_path)

    time_h = working[TIME_H]
    biomass = working[BIOMASS_G_L]
    od600 = working[OD600]

    fig, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax2 = ax1.twinx()

    phase_handles = _phase_overlay(ax1, time_h, labels)
    ax1.plot(time_h, biomass, color=PALETTE["biomass"], linewidth=1.8, label="biomass_gL")
    ax2.plot(time_h, od600, color=PALETTE["od600"], linewidth=1.6, linestyle="--", label="OD600")

    if log_scale:
        ax1.set_yscale("log")

    ax1.set_xlabel("time_h (h)")
    ax1.set_ylabel("biomass_gL")
    ax2.set_ylabel("OD600")
    _apply_style(ax1, xlabel="time_h (h)", ylabel="biomass_gL")
    _apply_style(ax2, ylabel="OD600")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    final_handles = handles1 + handles2 + phase_handles
    final_labels = labels1 + labels2 + [handle.get_label() for handle in phase_handles]
    ax1.legend(final_handles, final_labels, loc="upper right", frameon=False, fontsize=7)

    return _save_and_close(fig, out_path)


def plot_growth_rate(df: pd.DataFrame, labels: Optional[pd.Series] = None, out_path: str | Path = "growth_rate.png") -> Path:
    """Plot the specific growth rate and mark its maximum."""
    try:
        working = _prepare_df(df, [TIME_H, BIOMASS_G_L])
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_growth_rate: missing required columns in input DataFrame.")
        return Path(out_path)

    mu = specific_growth_rate_series(working)
    if mu.empty:
        warnings.warn("Skipping plot_growth_rate: no valid growth-rate values were computed.")
        return Path(out_path)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    phase_handles = _phase_overlay(ax, working[TIME_H], labels)

    ax.plot(working[TIME_H], mu.to_numpy(), color=PALETTE["mu"], linewidth=1.8)
    mu_max = float(np.nanmax(mu.to_numpy())) if mu.notna().any() else 0.0
    if np.isfinite(mu_max) and mu_max > 0:
        idx = mu.idxmax()
        ax.axhline(mu_max, color=PALETTE["mu"], linestyle="--", linewidth=1.0, alpha=0.8)
        ax.annotate(
            f"mu_max = {mu_max:.3f} 1/h\n@ {working.loc[idx, TIME_H]:.1f} h",
            xy=(working.loc[idx, TIME_H], mu_max),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color=PALETTE["mu"],
        )

    ax.set_xlabel("time_h (h)")
    ax.set_ylabel("mu (1/h)")
    _apply_style(ax, xlabel="time_h (h)", ylabel="mu (1/h)")

    if phase_handles:
        phase_labels = [handle.get_label() for handle in phase_handles]
        ax.legend(phase_handles + ax.get_legend_handles_labels()[0], phase_labels + ax.get_legend_handles_labels()[1], loc="upper right", frameon=False, fontsize=7)

    return _save_and_close(fig, out_path)


def plot_nutrients(df: pd.DataFrame, labels: Optional[pd.Series] = None, out_path: str | Path = "nutrients.png") -> Path:
    """Plot ammonia nitrogen and phosphate with a depletion marker."""
    try:
        working = _prepare_df(df, [TIME_H, NH3_N_G_L, PO4_G_L])
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_nutrients: missing required columns in input DataFrame.")
        return Path(out_path)

    fig, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax2 = ax1.twinx()
    phase_handles = _phase_overlay(ax1, working[TIME_H], labels)

    ax1.plot(working[TIME_H], working[NH3_N_G_L], color=PALETTE["nh3"], linewidth=1.8, label="NH3-N (g/L)")
    ax2.plot(working[TIME_H], working[PO4_G_L], color=PALETTE["phosphate"], linewidth=1.6, linestyle="--", label="PO4 (g/L)")

    depletion_idx = np.where(working[NH3_N_G_L].to_numpy() <= 0.1)[0]
    if depletion_idx.size:
        idx = int(depletion_idx[0])
        x = float(working.loc[idx, TIME_H])
        y = float(working.loc[idx, NH3_N_G_L])
        ax1.axvline(x, color=PALETTE["bad"], linestyle=":", linewidth=1.0, alpha=0.8)
        ax1.annotate("N depletion", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color=PALETTE["bad"])

    ax1.set_xlabel("time_h (h)")
    ax1.set_ylabel("NH3-N (g/L)")
    ax2.set_ylabel("PO4 (g/L)")
    _apply_style(ax1, xlabel="time_h (h)", ylabel="NH3-N (g/L)")
    _apply_style(ax2, ylabel="PO4 (g/L)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    final_handles = handles1 + handles2 + phase_handles
    final_labels = labels1 + labels2 + [handle.get_label() for handle in phase_handles]
    ax1.legend(final_handles, final_labels, loc="upper right", frameon=False, fontsize=7)

    return _save_and_close(fig, out_path)


def plot_gas_uptake(df: pd.DataFrame, labels: Optional[pd.Series] = None, out_path: str | Path = "gas_uptake.png") -> Path:
    """Plot H2, CO2, and O2 uptake rates over time."""
    try:
        working = _prepare_df(df, [TIME_H, H2_UPTAKE_MMOL_L_H, CO2_UPTAKE_MMOL_L_H, O2_UPTAKE_MMOL_L_H])
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_gas_uptake: missing required columns in input DataFrame.")
        return Path(out_path)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    phase_handles = _phase_overlay(ax, working[TIME_H], labels)

    ax.plot(working[TIME_H], working[H2_UPTAKE_MMOL_L_H], color=PALETTE["h2"], linewidth=1.8, label="H2")
    ax.plot(working[TIME_H], working[CO2_UPTAKE_MMOL_L_H], color=PALETTE["co2"], linewidth=1.6, label="CO2")
    ax.plot(working[TIME_H], working[O2_UPTAKE_MMOL_L_H], color=PALETTE["o2"], linewidth=1.6, linestyle="--", label="O2")

    ax.set_xlabel("time_h (h)")
    ax.set_ylabel("uptake (mmol/L/h)")
    _apply_style(ax, xlabel="time_h (h)", ylabel="uptake (mmol/L/h)")

    handles, labels_out = ax.get_legend_handles_labels()
    final_handles = handles + phase_handles
    final_labels = labels_out + [handle.get_label() for handle in phase_handles]
    ax.legend(final_handles, final_labels, loc="upper right", frameon=False, fontsize=7)

    return _save_and_close(fig, out_path)


def plot_offgas_do(df: pd.DataFrame, labels: Optional[pd.Series] = None, out_path: str | Path = "offgas_do.png") -> Path:
    """Plot off-gas composition and dissolved oxygen in stacked panels."""
    try:
        working = _prepare_df(df, [TIME_H, OFFGAS_H2_PCT, OFFGAS_CO2_PCT, OFFGAS_O2_PCT, DO_PERCENT])
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_offgas_do: missing required columns in input DataFrame.")
        return Path(out_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.5), dpi=DPI, sharex=True)

    phase_handles = _phase_overlay(ax1, working[TIME_H], labels)
    ax1.plot(working[TIME_H], working[OFFGAS_H2_PCT], color=PALETTE["offgas_h2"], linewidth=1.8, label="H2")
    ax1.plot(working[TIME_H], working[OFFGAS_CO2_PCT], color=PALETTE["offgas_co2"], linewidth=1.6, label="CO2")
    ax1.plot(working[TIME_H], working[OFFGAS_O2_PCT], color=PALETTE["offgas_o2"], linewidth=1.6, linestyle="--", label="O2")
    ax1.set_ylabel("off-gas (%)")
    _apply_style(ax1, ylabel="off-gas (%)")
    handles, labels_out = ax1.get_legend_handles_labels()
    final_handles = handles + phase_handles
    final_labels = labels_out + [handle.get_label() for handle in phase_handles]
    ax1.legend(final_handles, final_labels, loc="upper right", frameon=False, fontsize=7)

    ax2.plot(working[TIME_H], working[DO_PERCENT], color=PALETTE["do"], linewidth=1.8)
    ax2.set_ylabel("DO_percent")
    ax2.set_xlabel("time_h (h)")
    _apply_style(ax2, xlabel="time_h (h)", ylabel="DO_percent")

    return _save_and_close(fig, out_path)


def plot_cpp(df: pd.DataFrame, limits: Mapping[str, tuple[float, float]], out_path: str | Path = "cpp.png") -> Path:
    """Plot process-control variables with green acceptable bands and red out-of-range points."""
    if not isinstance(limits, Mapping):
        raise TypeError("limits must be a mapping of variable names to (lower, upper) bounds")

    required = [
        TIME_H,
        PH,
        TEMPERATURE_C,
        AGITATION_RPM,
        PRESSURE_BAR,
    ]
    try:
        working = _prepare_df(df, required)
    except (KeyError, TypeError):
        warnings.warn(f"Skipping plot_cpp: missing required columns in input DataFrame.")
        return Path(out_path)

    variables = [
        (PH, "pH", PALETTE["pH"]),
        (TEMPERATURE_C, "temperature_C", PALETTE["temperature"]),
        (AGITATION_RPM, "agitation_rpm", PALETTE["agitation"]),
        (PRESSURE_BAR, "pressure_bar", PALETTE["pressure"]),
    ]

    fig, axes = plt.subplots(len(variables), 1, figsize=(7.0, 7.0), dpi=DPI, sharex=True)
    if len(variables) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.45)

    for ax, (column, label, color) in zip(axes, variables):
        if column not in working.columns:
            continue
        if label not in limits:
            continue

        lower, upper = limits[label]
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue

        ax.axhspan(float(lower), float(upper), color="#2ca02c", alpha=0.10, zorder=0)

        series = working[column].astype(float)
        ax.plot(working[TIME_H], series, color=color, linewidth=1.6)

        out_of_range = (series < lower) | (series > upper)
        if out_of_range.any():
            ax.scatter(working.loc[out_of_range, TIME_H], series[out_of_range], color=PALETTE["bad"], s=14, zorder=3)

        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        _apply_style(ax, ylabel=label)

    axes[-1].set_xlabel("time_h (h)")
    return _save_and_close(fig, out_path)


def make_all_plots(df: pd.DataFrame, labels: Optional[pd.Series], limits: Mapping[str, tuple[float, float]], out_dir: str | Path) -> Dict[str, Path]:
    """Generate all report plots in the requested output directory."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map: Dict[str, Path] = {
        "growth": plot_growth(df, labels=labels, out_path=output_dir / "growth.png"),
        "growth_rate": plot_growth_rate(df, labels=labels, out_path=output_dir / "growth_rate.png"),
        "nutrients": plot_nutrients(df, labels=labels, out_path=output_dir / "nutrients.png"),
        "gas_uptake": plot_gas_uptake(df, labels=labels, out_path=output_dir / "gas_uptake.png"),
        "offgas_do": plot_offgas_do(df, labels=labels, out_path=output_dir / "offgas_do.png"),
        "cpp": plot_cpp(df, limits=limits, out_path=output_dir / "cpp.png"),
    }
    return file_map


__all__ = [
    "plot_growth",
    "plot_growth_rate",
    "plot_nutrients",
    "plot_gas_uptake",
    "plot_offgas_do",
    "plot_cpp",
    "make_all_plots",
]
