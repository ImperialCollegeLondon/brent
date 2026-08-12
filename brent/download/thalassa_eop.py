"""Convert IERS finals2000A.all records to THALASSA's EOP format.

THALASSA expects the fixed-width fields parsed by NSGRAV_INITIALIZE_EOP:
MJD, polar motion X/Y, UT1-UTC, LOD, dX, and dY.  By default the converter
keeps data through 31 December 2025 plus seven days of padding for THALASSA's
seven-day rotation precomputation.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import tempfile


MJD_EPOCH = date(1858, 11, 17)
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "orekit"
    / "Earth-Orientation-Parameters"
    / "IAU-2000"
    / "finals2000A.all"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "thalassa"
    / "data"
    / "eop_data.txt"
)


def date_to_mjd(value: str) -> int:
    """Convert an ISO calendar date to its integer MJD."""

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from error
    return (parsed - MJD_EPOCH).days


def parse_record(
    line: str,
) -> tuple[float, float, float, float, float, float, float] | None:
    """Read the seven fields at the offsets used by THALASSA's Fortran parser."""

    if len(line) < 125 or not line.strip():
        return None

    fields = (
        line[7:15],
        line[18:27],
        line[37:46],
        line[58:68],
        line[79:86],
        line[97:106],
        line[116:125],
    )
    try:
        return tuple(float(field) for field in fields)  # type: ignore[return-value]
    except ValueError:
        return None


def format_record(
    values: tuple[float, float, float, float, float, float, float]
) -> str:
    """Write a record accepted by NSGRAV_INITIALIZE_EOP's fixed-width READ."""

    mjd, pmx, pmy, ut1_utc, lod, dx, dy = values
    return (
        f"{'':7}{mjd:8.2f}{'':3}{pmx:9.6f}{'':10}{pmy:9.6f}"
        f"{'':12}{ut1_utc:10.7f}{'':11}{lod:7.4f}"
        f"{'':11}{dx:9.3f}{'':10}{dy:9.3f}\n"
    )


def convert(
    input_path: Path, output_path: Path, end_mjd: int, padding_days: int
) -> tuple[int, int]:
    """Convert records through ``end_mjd + padding_days`` atomically."""

    last_mjd = end_mjd + padding_days
    records: list[str] = []
    expected_mjd: int | None = None

    with input_path.open("r", encoding="ascii") as source:
        for line_number, line in enumerate(source, start=1):
            values = parse_record(line)
            if values is None:
                continue

            mjd = int(values[0])
            if expected_mjd is None:
                expected_mjd = mjd
            if mjd < expected_mjd:
                raise ValueError(f"MJD order decreases at source line {line_number}")
            if mjd > last_mjd:
                break
            if mjd != expected_mjd:
                raise ValueError(
                    f"missing MJD {expected_mjd} before source line {line_number}"
                )

            records.append(format_record(values))
            expected_mjd += 1

    if not records:
        raise ValueError(f"no complete EOP records found in {input_path}")
    if int(float(records[-1][7:15])) < last_mjd:
        raise ValueError(
            f"{input_path} does not contain complete records through MJD {last_mjd}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.writelines(records)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return int(float(records[0][7:15])), int(float(records[-1][7:15]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--end-date",
        type=date_to_mjd,
        default=date_to_mjd("2025-12-31"),
        help="last requested calendar date (default: 2025-12-31)",
    )
    parser.add_argument(
        "--padding-days",
        type=int,
        default=7,
        help="extra records retained after end-date (default: 7)",
    )
    args = parser.parse_args()
    if args.padding_days < 0:
        parser.error("--padding-days must be non-negative")

    first_mjd, last_mjd = convert(
        args.input, args.output, args.end_date, args.padding_days
    )
    print(f"Wrote {last_mjd - first_mjd + 1} records to {args.output}")
    print(f"MJD range: {first_mjd} through {last_mjd}")
    print(
        "Calendar range: "
        f"{MJD_EPOCH + timedelta(days=first_mjd)} through "
        f"{MJD_EPOCH + timedelta(days=last_mjd)}"
    )


if __name__ == "__main__":
    main()
