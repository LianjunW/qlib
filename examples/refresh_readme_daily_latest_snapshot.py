#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rewrite the ``Latest score snapshot`` block in README_daily_update_latest.md.

Picks the newest ``examples/daily_report/YYYY-MM-DD.json`` whose date is on or before the
last CN trading day from baostock ``query_trade_dates`` (1d), then replaces the HTML-comment
marked region in the README.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BEGIN = "<!-- DAILY_SCORE_SNAPSHOT_BEGIN -->"
END = "<!-- DAILY_SCORE_SNAPSHOT_END -->"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_baostock_collector(repo_root: Path):
    path = repo_root / "scripts" / "data_collector" / "baostock" / "collector.py"
    spec = importlib.util.spec_from_file_location("baostock_collector_dynamic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baostock_last_trade_date(repo_root: Path) -> Optional[str]:
    try:
        mod = _load_baostock_collector(repo_root)
        return mod.query_latest_trading_day_cn_baostock()
    except Exception as e:
        print(f"warning: baostock calendar unavailable ({e}); using newest local report only.", file=sys.stderr)
        return None


def _list_report_files(report_dir: Path) -> List[Tuple[pd.Timestamp, Path]]:
    out: List[Tuple[pd.Timestamp, Path]] = []
    for p in sorted(report_dir.glob("*.json")):
        m = re.match(r"^(\d{4}-\d{2}-\d{2})\.json$", p.name)
        if not m:
            continue
        out.append((pd.Timestamp(m.group(1)), p))
    return out


def pick_report_path(
    report_dir: Path,
    *,
    repo_root: Path,
    explicit: Optional[Path],
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Returns (json_path, baostock_last_or_none).

    Chooses max report date D with D <= baostock_last when auto-picking; if no baostock
    response, uses the newest dated report. ``explicit`` forces a specific JSON file.
    """
    bs_last = _baostock_last_trade_date(repo_root)
    files = _list_report_files(report_dir)
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            return None, bs_last
        return p, bs_last

    if not files:
        return None, bs_last

    cap = pd.Timestamp(bs_last) if bs_last else None
    candidates = [(t, p) for t, p in files if cap is None or t <= cap]
    if not candidates:
        t, p = max(files, key=lambda x: x[0])
        return p, bs_last
    t, p = max(candidates, key=lambda x: x[0])
    return p, bs_last


def _fmt_top5(top_scores: List[Dict[str, Any]]) -> str:
    parts = []
    for row in top_scores[:5]:
        sym = row.get("instrument", "")
        name = row.get("name", "")
        sc = float(row.get("score", 0.0))
        parts.append(f"{sym} {name} ({sc:.4f})")
    return ", ".join(parts)


def build_snapshot_markdown(data: Dict[str, Any], *, report_rel: str, bs_last: Optional[str]) -> str:
    date = data.get("date") or ""
    gen = data.get("generated_at") or ""
    n = data.get("total_instruments_scored", "")
    sig = data.get("market_signal", "")
    desc = (data.get("market_signal_description") or "").strip()
    stats = data.get("score_stats") or {}
    mean = float(stats.get("mean", 0.0))
    med = float(stats.get("median", 0.0))
    tkm = float(stats.get("top_k_mean", 0.0))
    bkm = float(stats.get("bottom_k_mean", 0.0))
    spr = float(stats.get("spread", 0.0))
    top5 = _fmt_top5(list(data.get("top_scores") or []))

    lines = [
        BEGIN,
        "",
        f"_(Generated from `{report_rel}`; refresh with `examples/refresh_readme_daily_latest_snapshot.py`.)_",
    ]
    if bs_last:
        lines.append(f"Baostock last CN trade date (1d calendar): `{bs_last}`.")
    lines += [
        "",
        f"Saved report: `{report_rel}` (generated `{gen}`).",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Score date | {date} |",
        f"| Instruments scored | {n} |",
        f"| Market signal | {sig} — {desc} |",
        f"| Mean / median score | {mean:.4f} / {med:.4f} |",
        f"| Top-K mean / bottom-K mean | {tkm:.4f} / {bkm:.4f} |",
        f"| Spread (top − bottom) | {spr:.4f} |",
        "",
        f"Top 5 by model score: {top5}.",
        "",
        END,
    ]
    return "\n".join(lines) + "\n"


def replace_marked_block(readme_text: str, new_block: str) -> str:
    if BEGIN not in readme_text or END not in readme_text:
        raise ValueError(
            f"README must contain {BEGIN!r} and {END!r} (wrap the snapshot body to replace)."
        )
    pre, rest = readme_text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    return pre + new_block + post


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to README_daily_update_latest.md (default: examples/README_daily_update_latest.md).",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory of daily_report JSON files (default: examples/daily_report).",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Use this report JSON instead of auto-picking by baostock calendar.",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    readme = (args.readme or (repo_root / "examples" / "README_daily_update_latest.md")).expanduser().resolve()
    report_dir = (args.report_dir or (repo_root / "examples" / "daily_report")).expanduser().resolve()

    path, bs_last = pick_report_path(report_dir, repo_root=repo_root, explicit=args.json)
    if path is None:
        print(f"No suitable JSON under {report_dir}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    report_rel = path.relative_to(repo_root).as_posix()
    block = build_snapshot_markdown(data, report_rel=report_rel, bs_last=bs_last)

    text = readme.read_text(encoding="utf-8")
    updated = replace_marked_block(text, block)
    readme.write_text(updated, encoding="utf-8")
    print(f"Updated {readme} from {path}" + (f" (baostock cap {bs_last})" if bs_last else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
