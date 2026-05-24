"""Data preparation pipeline: raw ingestion and cleaning stages."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_data  # noqa: E402

RAW_INPUT = "data/MachineLearningRating_v3.txt"
RAW_OUTPUT = "data/insurance_data.csv"
CLEAN_OUTPUT = "data/insurance_data_cleaned.csv"

DROP_COLS = [
    "CrossBorder", "NewVehicle", "WrittenOff", "Rebuilt",
    "Converted", "NumberOfVehiclesInFleet", "CustomValueEstimate",
]

FILL_COLS = ["Bank", "AccountType", "Gender", "MaritalStatus"]

VEHICLE_COLS = [
    "mmcode", "VehicleType", "make", "Model", "VehicleIntroDate",
    "NumberOfDoors", "bodytype", "kilowatts", "cubiccapacity", "Cylinders",
]


def prepare_raw():
    """Read the pipe-delimited source file and write as standard CSV."""
    df = load_data(RAW_INPUT)
    # Cast nullable Int64 to float so CSV round-trips cleanly
    for col in df.select_dtypes("Int64").columns:
        df[col] = df[col].astype("float64")
    for col in df.select_dtypes("boolean").columns:
        df[col] = df[col].astype("object")
    df.to_csv(RAW_OUTPUT, index=False)
    print(f"[raw]     {df.shape[0]:,} rows × {df.shape[1]} cols  →  {RAW_OUTPUT}")


def prepare_cleaned():
    """Apply EDA-documented cleaning strategy and write cleaned CSV."""
    df = pd.read_csv(RAW_OUTPUT, low_memory=False)

    before = len(df)

    # Drop columns with >50% missing or zero information
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Impute low-missing categoricals with a sentinel value
    for col in FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Not specified")

    # Drop rows missing vehicle attributes (0.06% of data)
    vehicle_present = [c for c in VEHICLE_COLS if c in df.columns]
    df = df.dropna(subset=vehicle_present)

    # Drop 2 rows missing CapitalOutstanding
    if "CapitalOutstanding" in df.columns:
        df = df.dropna(subset=["CapitalOutstanding"])

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    df.to_csv(CLEAN_OUTPUT, index=False)
    print(
        f"[cleaned] {before:,} → {len(df):,} rows × {df.shape[1]} cols  →  {CLEAN_OUTPUT}"
    )
    print(f"          Dropped {before - len(df):,} rows and {7} high-missing columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance data preparation pipeline")
    parser.add_argument(
        "--stage",
        choices=["raw", "clean", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    if args.stage in ("raw", "all"):
        prepare_raw()
    if args.stage in ("clean", "all"):
        prepare_cleaned()
