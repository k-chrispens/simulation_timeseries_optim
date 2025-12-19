from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
import reciprocalspaceship as rs
from loguru import logger


def load_structure_factors(path: str, amp_label: str, phase_label: str) -> pd.DataFrame:
    ds = rs.read_mtz(path).expand_to_p1()
    ds = ds.dropna(subset=[amp_label, phase_label])

    sf = ds.to_structurefactor(amp_label, phase_label)
    df = ds.reset_index()[["H", "K", "L", amp_label, phase_label]].copy()
    df = df.rename(columns={amp_label: "amplitude", phase_label: "phase_deg"})
    df["f_real"] = sf.to_numpy().real
    df["f_imag"] = sf.to_numpy().imag
    df["source"] = Path(path).name
    return df


def stack_mtz(pattern: str, labels: tuple[str, str]) -> pd.DataFrame:
    mtz_files = sorted(glob.glob(pattern))
    if not mtz_files:
        raise FileNotFoundError(f"No MTZ files found for pattern {pattern!r}")
    frames = [load_structure_factors(path, *labels) for path in mtz_files]
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack MTZ structure factors into a parquet file."
    )
    parser.add_argument(
        "--glob", required=True, help="Glob pattern for input MTZ files."
    )
    parser.add_argument("--output", required=True, help="Destination parquet file.")
    parser.add_argument(
        "--labels",
        default="FC,PHIC",
        help="Comma-separated amplitude and phase column names (default: FC,PHIC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(part.strip() for part in args.labels.split(","))
    if len(labels) != 2:
        raise ValueError("Expected two comma-separated labels: amplitude,phase")

    stacked = stack_mtz(args.glob, labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(stacked)} rows to {output_path}")


if __name__ == "__main__":
    main()
