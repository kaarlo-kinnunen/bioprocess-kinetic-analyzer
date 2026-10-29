from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from bioprocess.phases import PhaseSummary, detect_phases, summarize_phases


def test_detect_phases_produces_valid_labels():
    """Phase detection should classify the synthetic run without NaNs or invalid labels."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "sample_run.csv")

    labels = detect_phases(df)

    assert isinstance(labels, pd.Series)
    assert not labels.isna().any()
    assert set(labels.unique()).issubset({"lag", "exponential", "stationary", "transition"})


def test_summarize_phases_returns_phase_summary():
    """Phase summaries should include a per-phase record with numeric timing and biomass values."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "sample_run.csv")
    labels = detect_phases(df)

    summaries = summarize_phases(df, labels)

    assert set(summaries).issuperset({"lag", "exponential", "stationary", "transition"})
    for phase_name, summary in summaries.items():
        if summary is None:
            continue
        assert isinstance(summary, PhaseSummary)
        assert summary.phase == phase_name
        assert summary.duration_h >= 0.0
        assert summary.biomass_start_g_L >= 0.0
        assert summary.biomass_end_g_L >= 0.0
