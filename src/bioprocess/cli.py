"""Command-line entry point for generating the bioprocess executive PDF report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .loader import DataValidationError, load_run_with_report
from .report import build_report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the report generator."""
    parser = argparse.ArgumentParser(
        description="Load a sample bioprocess CSV and build the executive PDF report."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the input CSV file, such as data/sample_run.csv",
    )
    parser.add_argument(
        "--out",
        default="output/report.pdf",
        help="Output PDF path (default: output/report.pdf)",
    )
    parser.add_argument(
        "--run-id",
        default="RUN-001",
        help="Run identifier embedded in the report header (default: RUN-001)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate a report from a CSV input file and print the output path."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        df, validation_report = load_run_with_report(args.csv_path)
    except DataValidationError as exc:
        parser.error(str(exc))

    for warning in validation_report.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    output_path = Path(args.out)
    output_path = build_report(df, output_path, run_id=args.run_id)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
