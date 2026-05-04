#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep-dive stocks flagged by `check_baostock_recent_consistency.py`.

This script extends the inspection window and measures how stable the
normalize-vs-qlib ratios remain over time. It helps separate:
1. benign constant-scale differences;
2. mild ratio drift likely caused by different normalization anchors;
3. actual recent inconsistencies worth manual inspection.
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


def load_candidates(report_path: Path, limit: int) -> List[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report["normalize_vs_qlib"]["ratio_stability"]["examples_inconsistent"]
    out = []
    for item in items[:limit]:
        out.append(str(item["instrument"]).upper())
    return out


def to_csv_name(instrument: str) -> str:
    symbol = str(instrument).strip().upper()
    return symbol[:2].lower() + symbol[2:].lower() + ".csv"


def load_normalize(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    base = Path("scripts/data_collector/baostock/normalize")
    for sym in symbols:
        path = base / to_csv_name(sym)
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date", "close", "volume", "factor", "change"])
        df["instrument"] = sym
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "instrument"])


def load_qlib(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    fields = ["$close", "$volume", "$factor", "$change"]
    df = D.features(symbols, fields, start_time=start, end_time=end, freq="day")
    df.columns = [c[1:] for c in df.columns]
    df = df.reset_index()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    return df.drop(columns=["datetime"])


def classify_item(row: Dict[str, float]) -> str:
    close_std = row.get("close_ratio_std", np.nan)
    factor_std = row.get("factor_ratio_std", np.nan)
    volume_std = row.get("volume_ratio_std", np.nan)
    change_max = row.get("change_max_abs_diff", np.nan)
    if np.isfinite(change_max) and change_max > 1e-6:
        return "recent_change_mismatch"
    if np.isfinite(close_std) and close_std <= 5e-4 and np.isfinite(factor_std) and factor_std <= 5e-4:
        return "stable_scale_difference"
    if np.isfinite(close_std) and close_std <= 2e-3 and np.isfinite(factor_std) and factor_std <= 2e-3:
        return "mild_ratio_drift"
    if np.isfinite(volume_std) and volume_std > 1e-2:
        return "volume_ratio_unstable"
    return "needs_manual_check"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep dive ratio drift for baostock-vs-qlib inconsistent stocks.")
    parser.add_argument(
        "--report",
        default="examples/daily_report/baostock_recent_consistency.json",
        help="Path to the recent consistency report.",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-10")
    parser.add_argument("--output", default="examples/daily_report/baostock_ratio_drift_deep_dive.json")
    args = parser.parse_args()

    init_qlib()
    report_path = Path(args.report).expanduser().resolve()
    symbols = load_candidates(report_path, args.limit)

    norm = load_normalize(symbols, args.start, args.end)
    qlib_df = load_qlib(symbols, args.start, args.end)
    merged = norm.merge(qlib_df, on=["instrument", "date"], suffixes=("_norm", "_qlib"), how="inner")

    items = []
    for sym, group in merged.groupby("instrument"):
        item: Dict[str, float] = {"instrument": sym, "rows": int(len(group))}
        for col in ["close", "volume", "factor"]:
            ratio = pd.to_numeric(group[f"{col}_norm"], errors="coerce") / pd.to_numeric(
                group[f"{col}_qlib"], errors="coerce"
            ).replace(0, np.nan)
            item[f"{col}_ratio_mean"] = float(ratio.mean(skipna=True))
            item[f"{col}_ratio_std"] = float(ratio.std(skipna=True))
            item[f"{col}_ratio_min"] = float(ratio.min(skipna=True))
            item[f"{col}_ratio_max"] = float(ratio.max(skipna=True))
        change_diff = (
            pd.to_numeric(group["change_norm"], errors="coerce")
            - pd.to_numeric(group["change_qlib"], errors="coerce")
        ).abs()
        item["change_max_abs_diff"] = float(change_diff.max(skipna=True))
        item["classification"] = classify_item(item)
        items.append(item)

    items = sorted(items, key=lambda x: (x["classification"], x["close_ratio_std"]), reverse=False)
    summary = {
        "window": {"start": args.start, "end": args.end},
        "candidate_count": len(symbols),
        "items": items,
        "classification_counts": pd.Series([x["classification"] for x in items]).value_counts().to_dict(),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Deep-dive window: {args.start} -> {args.end}")
    print(f"Candidates: {len(symbols)}")
    print("Classification counts:", summary["classification_counts"])
    print(f"Saved deep-dive report to {out}")


if __name__ == "__main__":
    main()
