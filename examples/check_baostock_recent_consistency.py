#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check recent consistency between local baostock csv data and current qlib bin data.

The goal is to verify whether baostock itself has recent-date issues, even when
historical source csv files may already have been overwritten.

This script compares:
1. `scripts/data_collector/baostock/source/*.csv` raw rows
2. `scripts/data_collector/baostock/normalize/*.csv` normalized rows
3. current qlib bin data under `~/.qlib/qlib_data/cn_data`

It focuses on recent trading days and reports whether:
- raw close tracks the same day-level direction;
- normalized `change` matches qlib `$change`;
- normalized price/volume/factor differs only by a near-constant scale.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def init_qlib() -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=str(Path("~/.qlib/qlib_data/cn_data").expanduser()), region=REG_CN)


def read_scope_symbols(scope_file: Path) -> List[str]:
    df = pd.read_csv(scope_file, sep="\t", header=None, usecols=[0], names=["instrument"])
    return sorted(set(df["instrument"].astype(str).str.upper()))


def recent_calendar_days(calendar_path: Path, days: int) -> List[str]:
    lines = [x.strip() for x in calendar_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[-days:]


def to_csv_name(instrument: str) -> str:
    symbol = str(instrument).strip().upper()
    return symbol[:2].lower() + symbol[2:].lower() + ".csv"


def load_source_rows(source_dir: Path, instruments: List[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    for inst in instruments:
        path = source_dir / to_csv_name(inst)
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume", "amount"])
        df["instrument"] = inst
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "instrument"])
    return pd.concat(rows, ignore_index=True)


def load_normalize_rows(normalize_dir: Path, instruments: List[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    for inst in instruments:
        path = normalize_dir / to_csv_name(inst)
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume", "factor", "change"])
        df["instrument"] = inst
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "instrument"])
    return pd.concat(rows, ignore_index=True)


def load_qlib_rows(instruments: List[str], start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor", "$change"]
    df = D.features(instruments, fields, start_time=start, end_time=end, freq="day")
    df.columns = [c[1:] for c in df.columns]
    df = df.reset_index()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    return df.drop(columns=["datetime"])


def summarize_pairwise(normalize_df: pd.DataFrame, qlib_df: pd.DataFrame) -> Dict[str, object]:
    merged = normalize_df.merge(qlib_df, on=["instrument", "date"], suffixes=("_norm", "_qlib"), how="inner")
    summary: Dict[str, object] = {"merged_rows": int(len(merged))}
    field_summary = {}
    for col in ["open", "high", "low", "close", "volume", "factor", "change"]:
        a = pd.to_numeric(merged[f"{col}_norm"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_qlib"], errors="coerce")
        diff = (a - b).abs()
        denom = b.abs().replace(0, np.nan)
        rel = diff / denom
        field_summary[col] = {
            "max_abs_diff": float(diff.max(skipna=True) if len(diff) else np.nan),
            "mean_abs_diff": float(diff.mean(skipna=True) if len(diff) else np.nan),
            "max_rel_diff": float(rel.max(skipna=True) if len(rel) else np.nan),
            "rows_gt_1e-6": int((diff > 1e-6).sum()),
            "rows_gt_1e-4": int((diff > 1e-4).sum()),
        }
    summary["field_summary"] = field_summary

    # Per-instrument ratio stability check: if ratios are stable across recent days,
    # recent path is likely fine and the difference is mainly a scaling anchor issue.
    ratio_rows = []
    for inst, group in merged.groupby("instrument"):
        item = {"instrument": inst}
        ok = True
        for col in ["open", "close", "volume", "factor"]:
            denom = pd.to_numeric(group[f"{col}_qlib"], errors="coerce").replace(0, np.nan)
            ratio = pd.to_numeric(group[f"{col}_norm"], errors="coerce") / denom
            item[f"{col}_ratio_mean"] = float(ratio.mean(skipna=True))
            item[f"{col}_ratio_std"] = float(ratio.std(skipna=True))
            if np.isfinite(item[f"{col}_ratio_std"]) and item[f"{col}_ratio_std"] > 1e-3:
                ok = False
        change_diff = (
            pd.to_numeric(group["change_norm"], errors="coerce")
            - pd.to_numeric(group["change_qlib"], errors="coerce")
        ).abs()
        item["change_max_abs_diff"] = float(change_diff.max(skipna=True))
        item["recent_shape_consistent"] = bool(ok and item["change_max_abs_diff"] <= 1e-6)
        ratio_rows.append(item)
    ratio_df = pd.DataFrame(ratio_rows).sort_values("change_max_abs_diff", ascending=False)
    summary["ratio_stability"] = {
        "instrument_count": int(len(ratio_df)),
        "shape_consistent_count": int(ratio_df["recent_shape_consistent"].sum()) if not ratio_df.empty else 0,
        "examples_inconsistent": ratio_df.loc[~ratio_df["recent_shape_consistent"]].head(20).to_dict("records"),
    }

    top_close_diff = merged.copy()
    top_close_diff["close_abs_diff"] = (
        pd.to_numeric(top_close_diff["close_norm"], errors="coerce")
        - pd.to_numeric(top_close_diff["close_qlib"], errors="coerce")
    ).abs()
    summary["largest_close_diffs"] = (
        top_close_diff.sort_values("close_abs_diff", ascending=False)
        .head(20)[["instrument", "date", "close_norm", "close_qlib", "close_abs_diff"]]
        .to_dict("records")
    )
    return summary


def compare_source_change(source_df: pd.DataFrame, normalize_df: pd.DataFrame) -> Dict[str, object]:
    merged = source_df.merge(
        normalize_df[["instrument", "date", "change", "close"]],
        on=["instrument", "date"],
        suffixes=("_src", "_norm"),
        how="inner",
    )
    merged = merged.sort_values(["instrument", "date"]).reset_index(drop=True)
    result = {"merged_rows": int(len(merged))}
    if merged.empty:
        result["status"] = "empty"
        return result

    # Rebuild source-side daily return from raw close within each instrument.
    merged["raw_change_rebuilt"] = merged.groupby("instrument")["close_src"].pct_change()
    valid = merged["raw_change_rebuilt"].notna() & pd.to_numeric(merged["change"], errors="coerce").notna()
    diff = (merged.loc[valid, "raw_change_rebuilt"] - merged.loc[valid, "change"]).abs()
    result["raw_change_vs_normalized_change"] = {
        "valid_rows": int(valid.sum()),
        "max_abs_diff": float(diff.max(skipna=True) if len(diff) else np.nan),
        "mean_abs_diff": float(diff.mean(skipna=True) if len(diff) else np.nan),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check recent baostock data consistency against qlib bin data.")
    parser.add_argument("--days", type=int, default=4, help="How many recent trading days to compare.")
    parser.add_argument("--scope-file", default="~/.qlib/qlib_data/cn_data/instruments/csi300.txt")
    parser.add_argument("--source-dir", default="scripts/data_collector/baostock/source")
    parser.add_argument("--normalize-dir", default="scripts/data_collector/baostock/normalize")
    parser.add_argument("--output", default="examples/daily_report/baostock_recent_consistency.json")
    args = parser.parse_args()

    init_qlib()
    qlib_dir = Path("~/.qlib/qlib_data/cn_data").expanduser()
    calendar_days = recent_calendar_days(qlib_dir / "calendars" / "day.txt", args.days)
    start, end = calendar_days[0], calendar_days[-1]
    scope_file = Path(args.scope_file).expanduser()
    instruments = read_scope_symbols(scope_file)

    source_df = load_source_rows(Path(args.source_dir), instruments, start, end)
    normalize_df = load_normalize_rows(Path(args.normalize_dir), instruments, start, end)
    qlib_df = load_qlib_rows(instruments, start, end)

    report = {
        "scope_file": str(scope_file.resolve()),
        "date_range": {"start": start, "end": end, "calendar_days": calendar_days},
        "instrument_count_in_scope_file": int(len(instruments)),
        "source_rows": int(len(source_df)),
        "normalize_rows": int(len(normalize_df)),
        "qlib_rows": int(len(qlib_df)),
        "source_vs_normalize": compare_source_change(source_df, normalize_df),
        "normalize_vs_qlib": summarize_pairwise(normalize_df, qlib_df),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Compared recent trading days: {start} -> {end}")
    print(f"Scope instruments: {len(instruments)}")
    print(f"Source rows: {len(source_df)}  Normalize rows: {len(normalize_df)}  Qlib rows: {len(qlib_df)}")
    change_stats = report["normalize_vs_qlib"]["field_summary"]["change"]
    print(
        "Normalize vs Qlib change diff: "
        f"max={change_stats['max_abs_diff']:.3e} mean={change_stats['mean_abs_diff']:.3e}"
    )
    ratio_stats = report["normalize_vs_qlib"]["ratio_stability"]
    print(
        "Recent-shape consistent instruments: "
        f"{ratio_stats['shape_consistent_count']}/{ratio_stats['instrument_count']}"
    )
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
