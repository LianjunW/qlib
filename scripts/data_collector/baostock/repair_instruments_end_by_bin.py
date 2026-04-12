#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair instruments end date based on actual bin length.

This is a lightweight fix for cases where instruments/all.txt end date is ahead of
the last date in the feature bin files, which blocks incremental dump_update.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


def _read_calendar(calendar_path: Path) -> list[str]:
    lines = [x.strip() for x in calendar_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"calendar file is empty: {calendar_path}")
    return lines


def _read_instruments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["symbol", "start", "end"])
    if df.empty:
        raise ValueError(f"instruments file is empty: {path}")
    return df


def _bin_last_date(
    bin_path: Path, calendar: list[str]
) -> Optional[str]:
    if not bin_path.exists():
        return None
    data = bin_path.read_bytes()
    if not data:
        return None
    n = len(data) // 4
    if n < 2:
        return None
    # First float is date_index, remaining are values.
    date_index = int(struct.unpack("<f", data[:4])[0])
    last_idx = date_index + (n - 2)
    if last_idx < 0 or last_idx >= len(calendar):
        return None
    return calendar[last_idx]


def _resolve_symbols(scope_file: Optional[Path]) -> Optional[set[str]]:
    if scope_file is None:
        return None
    df = pd.read_csv(scope_file, sep="\t", header=None, names=["symbol", "start", "end"])
    return set(df["symbol"].astype(str).str.strip())


def repair(
    qlib_dir: Path,
    field: str,
    freq: str,
    instruments_path: Path,
    calendar_path: Path,
    scope_file: Optional[Path],
    dry_run: bool,
) -> Tuple[int, int]:
    calendar = _read_calendar(calendar_path)
    inst_df = _read_instruments(instruments_path)
    scope = _resolve_symbols(scope_file)

    changed = 0
    missing = 0

    end_map: Dict[str, str] = {}
    for _, row in inst_df.iterrows():
        sym = str(row["symbol"]).strip().upper()
        if scope is not None and sym not in scope:
            continue
        bin_path = qlib_dir / "features" / sym.lower() / f"{field}.{freq}.bin"
        last_date = _bin_last_date(bin_path, calendar)
        if last_date is None:
            missing += 1
            continue
        end_map[sym] = last_date

    def _maybe_update(row: pd.Series) -> pd.Series:
        sym = str(row["symbol"]).strip().upper()
        if sym in end_map:
            current_end = str(row["end"])
            new_end = end_map[sym]
            if current_end != new_end:
                nonlocal changed
                changed += 1
                row["end"] = new_end
        return row

    inst_df = inst_df.apply(_maybe_update, axis=1)

    if not dry_run:
        backup = instruments_path.with_suffix(".txt.bak")
        instruments_path.replace(backup)
        inst_df.to_csv(instruments_path, sep="\t", header=False, index=False)
        print(f"backup: {backup}")
        print(f"written: {instruments_path}")

    return changed, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qlib_dir", required=True)
    p.add_argument("--field", default="close")
    p.add_argument("--freq", default="day")
    p.add_argument("--instruments_path", default=None)
    p.add_argument("--calendar_path", default=None)
    p.add_argument("--scope_file", default=None, help="optional instruments file (e.g. csi300.txt)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    instruments_path = (
        Path(args.instruments_path).expanduser().resolve()
        if args.instruments_path
        else qlib_dir / "instruments" / "all.txt"
    )
    calendar_path = (
        Path(args.calendar_path).expanduser().resolve()
        if args.calendar_path
        else qlib_dir / "calendars" / "day.txt"
    )
    scope_file = Path(args.scope_file).expanduser().resolve() if args.scope_file else None

    changed, missing = repair(
        qlib_dir=qlib_dir,
        field=args.field,
        freq=args.freq,
        instruments_path=instruments_path,
        calendar_path=calendar_path,
        scope_file=scope_file,
        dry_run=args.dry_run,
    )
    print(f"changed={changed} missing_bin={missing} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
