#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end: baostock data update → prediction-union instrument list → tail bin rewrite → daily score JSON.

Designed to match ``README_daily_update_latest.md`` (§2 incremental update + §3–4 union + ``rewrite_scope_bins_from_normalize``)
and ``daily_update_and_score.py`` (120-day prediction window).

Examples:
  python examples/baostock_daily_update_full_pipeline.py
  python examples/baostock_daily_update_full_pipeline.py --skip-collector   # only union + tail + score
  python examples/baostock_daily_update_full_pipeline.py --dry-run-score
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Same window as ``daily_update_and_score.load_model_and_predict`` (start = latest - 120d).
LOOKBACK_DAYS = 120
UNION_SCOPE_NAME = "csi300_score_union_latest"
CSI300_SEGMENT_START = "2005-04-08"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _examples_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_examples_on_path() -> None:
    ex = str(_examples_dir())
    if ex not in sys.path:
        sys.path.insert(0, ex)


def _run_subprocess(cmd: list, *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def _latest_calendar_date(qlib_data_dir: Path) -> Optional[str]:
    cal = qlib_data_dir / "calendars" / "day.txt"
    if not cal.exists():
        return None
    lines = [ln.strip() for ln in cal.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[-1] if lines else None


def step_baostock_incremental(
    *,
    repo_root: Path,
    qlib_data_dir: Path,
    python_exe: str,
) -> None:
    """README §2: download/normalize/dump for CSI300 snapshot scope."""
    collector = repo_root / "scripts" / "data_collector" / "baostock" / "collector.py"
    source_dir = repo_root / "scripts" / "data_collector" / "baostock" / "source"
    normalize_dir = repo_root / "scripts" / "data_collector" / "baostock" / "normalize"
    cmd = [
        python_exe,
        str(collector),
        "update_data_to_bin",
        "--qlib_data_1d_dir",
        str(qlib_data_dir),
        "--source_dir",
        str(source_dir),
        "--normalize_dir",
        str(normalize_dir),
        "--region",
        "CN",
        "--interval",
        "1d",
        "--instrument_scope",
        "csi300",
        "--source_skip_existing",
        "True",
        "--normalize_skip_existing",
        "True",
        "--refresh_csi300_instruments_baostock",
        "True",
        "--start_from_bin_end",
        "True",
    ]
    _run_subprocess(cmd, cwd=repo_root)


def build_prediction_union_instrument_file(
    *,
    qlib_data_dir: Path,
) -> Tuple[str, int]:
    """README §3 style: CSI300 constituent union over the same period as scoring (latest - 120d … latest)."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    latest = _latest_calendar_date(qlib_data_dir)
    if latest is None:
        raise FileNotFoundError(f"Missing calendar: {qlib_data_dir / 'calendars' / 'day.txt'}")

    start = (pd.Timestamp(latest) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    qlib.init(
        provider_uri=str(qlib_data_dir.expanduser().resolve()),
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
    )
    union = sorted(
        D.list_instruments(
            D.instruments("csi300"),
            start_time=start,
            end_time=latest,
            as_list=True,
            freq="day",
        )
    )
    inst_dir = qlib_data_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    out = inst_dir / f"{UNION_SCOPE_NAME}.txt"
    out.write_text(
        "".join(f"{s}\t{CSI300_SEGMENT_START}\t{latest}\n" for s in union),
        encoding="utf-8",
    )
    print(f"\nWrote {out}  (window {start} .. {latest}, n={len(union)})", flush=True)
    return latest, len(union)


def step_tail_rewrite_from_normalize(
    *,
    repo_root: Path,
    qlib_data_dir: Path,
    python_exe: str,
    latest_trade_date: str,
) -> None:
    """README §4: rewrite bin tails from local normalize csv for the union scope."""
    collector = repo_root / "scripts" / "data_collector" / "baostock" / "collector.py"
    source_dir = repo_root / "scripts" / "data_collector" / "baostock" / "source"
    normalize_dir = repo_root / "scripts" / "data_collector" / "baostock" / "normalize"
    # Open-interval end_date: include bars through ``latest_trade_date``.
    end_open = (pd.Timestamp(latest_trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [
        python_exe,
        str(collector),
        "update_data_to_bin",
        "--qlib_data_1d_dir",
        str(qlib_data_dir),
        "--source_dir",
        str(source_dir),
        "--normalize_dir",
        str(normalize_dir),
        "--region",
        "CN",
        "--interval",
        "1d",
        "--instrument_scope",
        UNION_SCOPE_NAME,
        "--skip_download",
        "True",
        "--skip_normalize",
        "True",
        "--rewrite_scope_bins_from_normalize",
        "True",
        "--end_date",
        end_open,
    ]
    _run_subprocess(cmd, cwd=repo_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baostock update + union tail patch + daily score (see README_daily_update_latest.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qlib-data-dir",
        default=os.path.expanduser("~/.qlib/qlib_data/cn_data"),
        help="Active qlib CN daily data directory.",
    )
    p.add_argument("--python", default=sys.executable, help="Python used to invoke collector.")
    p.add_argument("--skip-collector", action="store_true", help="Skip §2 baostock incremental update.")
    p.add_argument(
        "--skip-tail-rewrite",
        action="store_true",
        help="Skip §4 rewrite_scope_bins_from_normalize for the union scope.",
    )
    p.add_argument("--skip-score", action="store_true", help="Stop after data steps (no JSON report).")
    p.add_argument(
        "--dry-run-score",
        action="store_true",
        help="Run scoring logic but do not write the JSON report.",
    )
    # Forwarded to ``daily_update_and_score.run_pipeline``
    p.add_argument("--recorder-id", default=None)
    p.add_argument("--experiment-name", default="workflow")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-drop", type=int, default=2)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--score-date", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--refresh-stock-name-map", action="store_true")
    p.add_argument(
        "--stock-name-map",
        default=None,
        help="Same as daily_update_and_score.py (default: examples/local_stock_names.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    qlib_data_dir = Path(os.path.expanduser(args.qlib_data_dir))

    if args.refresh_stock_name_map:
        _ensure_examples_on_path()
        from daily_update_and_score import refresh_stock_name_map_via_baostock

        refresh_stock_name_map_via_baostock()

    if not args.skip_collector:
        step_baostock_incremental(
            repo_root=repo_root,
            qlib_data_dir=qlib_data_dir,
            python_exe=args.python,
        )

    latest_trade_date, _n_union = build_prediction_union_instrument_file(qlib_data_dir=qlib_data_dir)

    if not args.skip_tail_rewrite:
        step_tail_rewrite_from_normalize(
            repo_root=repo_root,
            qlib_data_dir=qlib_data_dir,
            python_exe=args.python,
            latest_trade_date=latest_trade_date,
        )

    if args.skip_score:
        print("\nDone (score skipped).", flush=True)
        return

    _ensure_examples_on_path()
    from daily_update_and_score import run_pipeline

    run_pipeline(
        recorder_id=args.recorder_id,
        experiment_name=args.experiment_name,
        topk=args.topk,
        n_drop=args.n_drop,
        previous_holdings=None,
        output_dir=args.output_dir,
        dry_run=args.dry_run_score,
        qlib_data_dir=str(qlib_data_dir),
        score_date=args.score_date,
        stock_name_map_path=args.stock_name_map,
    )


if __name__ == "__main__":
    main()
