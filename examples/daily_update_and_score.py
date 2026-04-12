#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily scoring pipeline.

Workflow:
  1. Load the trained model from a qlib recorder
  2. Generate predictions for the latest trading date
  3. Output top-K stock scores with stock names and strategy recommendations
  4. Save a daily report (JSON + console)

Usage:
  python examples/daily_update_and_score.py
  python examples/daily_update_and_score.py --recorder-id <id>
  python examples/daily_update_and_score.py --recorder-id <id> --score-date 2026-04-10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "daily_report")
DEFAULT_STOCK_NAME_MAP_PATH = os.path.join(os.path.dirname(__file__), "local_stock_names.json")
BAOSTOCK_SOURCE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "data_collector", "baostock", "source"
)
BAOSTOCK_NORMALIZE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "data_collector", "baostock", "normalize"
)


def get_latest_calendar_date(qlib_data_dir: Optional[str] = None) -> Optional[str]:
    base = Path(qlib_data_dir or QLIB_DATA_DIR)
    cal = base.joinpath("calendars", "day.txt")
    if not cal.exists():
        return None
    with cal.open("r", encoding="utf-8") as f:
        days = [line.strip() for line in f if line.strip()]
    return days[-1] if days else None


def init_qlib(provider_uri: Optional[str] = None) -> None:
    import qlib
    from qlib.constant import REG_CN
    from qlib.tests.data import GetData

    provider_uri = provider_uri or QLIB_DATA_DIR
    if not os.path.isdir(provider_uri):
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def load_stock_name_map(map_path: Optional[str] = None) -> Dict[str, str]:
    resolved = os.path.abspath(os.path.expanduser(map_path or DEFAULT_STOCK_NAME_MAP_PATH))
    if not os.path.isfile(resolved):
        return {}

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(k).upper(): str(v) for k, v in data.items()}


def stock_label(symbol: str, stock_name_map: Dict[str, str]) -> str:
    name = stock_name_map.get(str(symbol).upper())
    if name:
        return f"{symbol}({name})"
    return str(symbol)


def qlib_symbol_from_baostock(bs_code: str) -> Optional[str]:
    code = str(bs_code).strip().lower()
    if "." not in code:
        return None
    market, number = code.split(".", 1)
    if market not in {"sh", "sz", "bj"}:
        return None
    if not number:
        return None
    return f"{market.upper()}{number.upper()}"


def refresh_stock_name_map_via_baostock(map_path: Optional[str] = None) -> Dict[str, str]:
    import baostock as bs

    resolved = os.path.abspath(os.path.expanduser(map_path or DEFAULT_STOCK_NAME_MAP_PATH))
    existing_map = load_stock_name_map(resolved)

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login_result.error_msg}")

    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_basic failed: {rs.error_msg}")

        baostock_map: Dict[str, str] = {}
        while rs.error_code == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            if row.get("type") != "1":
                continue
            symbol = qlib_symbol_from_baostock(row.get("code", ""))
            name = str(row.get("code_name", "")).strip()
            if not symbol or not name:
                continue
            baostock_map[symbol] = name
    finally:
        bs.logout()

    merged_map = dict(existing_map)
    merged_map.update(baostock_map)
    merged_map = dict(sorted(merged_map.items(), key=lambda item: item[0]))

    with open(resolved, "w", encoding="utf-8") as f:
        json.dump(merged_map, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Refreshed stock name map via baostock: {len(baostock_map)} A-share names")
    print(f"Saved merged stock name map to {resolved} (total {len(merged_map)} entries)")
    return merged_map


def resolve_latest_recorder_id(experiment_name: str) -> str:
    from qlib.workflow import R

    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders(rtype=exp.RT_L, max_results=5000)

    for rec in recorders:
        info = getattr(rec, "info", {}) or {}
        if info.get("status") != "FINISHED":
            continue

        rid = getattr(rec, "id", None)
        if rid is None:
            continue

        recorder = exp.get_recorder(recorder_id=rid)
        try:
            pred = recorder.load_object("pred.pkl")
            recorder.load_object("params.pkl")
        except Exception:
            continue

        if pred is not None and len(pred) > 0:
            print(f"Using latest finished recorder with predictions: {rid}")
            return rid

    raise ValueError(
        f"No finished recorder with params.pkl and pred.pkl was found in experiment '{experiment_name}'."
    )


def load_model_and_predict(
    recorder_id: str,
    experiment_name: str,
    latest_date: str,
    target_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    from qlib.workflow import R
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    examples_dir = os.path.dirname(os.path.abspath(__file__))
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from workflow_by_code_v2 import get_extended_dataset_config, resolve_prediction_instruments
    from qlib.tests.config import CSI300_MARKET

    exp = R.get_exp(experiment_name=experiment_name)
    recorder = exp.get_recorder(recorder_id=recorder_id)

    print(f"\nLoading model from recorder {recorder_id}...")
    model = recorder.load_object("params.pkl")

    start_date = (pd.Timestamp(latest_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    prediction_instruments = resolve_prediction_instruments(
        instruments=CSI300_MARKET,
        test_start_time=start_date,
        test_end_time=latest_date,
        use_union_constituents=True,
    )

    dataset_config = get_extended_dataset_config(
        test=(start_date, latest_date),
        instruments=prediction_instruments,
        end_time=latest_date,
    )
    dataset = init_instance_by_config(dataset_config)

    print("Generating predictions...")
    pred = model.predict(dataset)

    if isinstance(pred, np.ndarray):
        test_data = dataset.prepare("test", col_set=["feature"], data_key=DataHandlerLP.DK_I)
        pred = pd.Series(pred, index=test_data.index, name="score")

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "score"})

    pred = pred.sort_index()
    available_dates = pd.DatetimeIndex(pred.index.get_level_values("datetime").unique()).sort_values()

    if target_date is not None:
        target_ts = pd.Timestamp(target_date).normalize()
        matching_dates = available_dates[available_dates.normalize() <= target_ts]
        if len(matching_dates) == 0:
            raise ValueError(
                f"No prediction rows are available on or before requested date {target_date}."
            )
        selected_ts = matching_dates[-1]
        if selected_ts.normalize() != target_ts:
            print(
                f"Requested report date {target_date} is not directly available; "
                f"using latest prediction date {selected_ts.strftime('%Y-%m-%d')} instead."
            )
    else:
        selected_ts = available_dates[-1]

    latest_pred = pred.loc[selected_ts].copy()

    if isinstance(latest_pred.index, pd.MultiIndex):
        latest_pred = latest_pred.droplevel("datetime")

    latest_pred = latest_pred.sort_values("score", ascending=False)
    return latest_pred, selected_ts


def _baostock_csv_name(instrument: str) -> str:
    symbol = str(instrument).strip().lower()
    if symbol.startswith(("sh", "sz", "bj")):
        return f"{symbol}.csv"
    raise ValueError(f"Unsupported instrument format: {instrument}")


def _get_price_from_baostock_source(instrument: str, date: str) -> Optional[float]:
    path = os.path.abspath(os.path.join(BAOSTOCK_SOURCE_DIR, _baostock_csv_name(instrument)))
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["date", "close"])
        row = df.loc[df["date"] == date]
        if row.empty:
            return None
        return float(row.iloc[-1]["close"])
    except Exception:
        return None


def _get_price_from_baostock_normalize(instrument: str, date: str) -> Optional[float]:
    path = os.path.abspath(os.path.join(BAOSTOCK_NORMALIZE_DIR, _baostock_csv_name(instrument)))
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, usecols=["date", "adjclose"])
        row = df.loc[df["date"] == date]
        if row.empty:
            return None
        return float(row.iloc[-1]["adjclose"])
    except Exception:
        return None


def get_latest_prices(instruments: List[str], date: str) -> pd.Series:
    prices: Dict[str, float] = {}
    for instrument in instruments:
        raw_price = _get_price_from_baostock_source(instrument, date)
        if raw_price is None:
            raw_price = _get_price_from_baostock_normalize(instrument, date)
        if raw_price is not None:
            prices[str(instrument)] = raw_price
    return pd.Series(prices, dtype=float)


def generate_strategy_recommendations(
    scores: pd.DataFrame,
    topk: int = 20,
    n_drop: int = 2,
    previous_holdings: Optional[List[str]] = None,
) -> Dict:
    ranked = scores["score"].sort_values(ascending=False)

    if previous_holdings is None:
        previous_holdings = []
    previous_holdings = list(dict.fromkeys(previous_holdings))

    if not previous_holdings:
        new_topk = list(ranked.head(topk).index)
        buys = list(new_topk)
        sells = []
        holds = []
    else:
        rank_lookup = {inst: pos for pos, inst in enumerate(ranked.index)}
        current_holdings = previous_holdings[:topk]
        drop_count = min(max(n_drop, 0), len(current_holdings))
        holdings_by_rank = sorted(
            current_holdings,
            key=lambda inst: rank_lookup.get(inst, len(ranked)),
            reverse=True,
        )
        sells = holdings_by_rank[:drop_count]
        retained = [inst for inst in current_holdings if inst not in sells]

        buys = []
        for inst in ranked.index:
            if inst in retained or inst in buys:
                continue
            buys.append(inst)
            if len(retained) + len(buys) >= topk:
                break

        new_topk = retained + buys
        holds = [s for s in new_topk if s in current_holdings]

    mean_score = float(ranked.mean())
    median_score = float(ranked.median())
    topk_mean = float(ranked.head(topk).mean())
    bottomk_mean = float(ranked.tail(topk).mean()) if len(ranked) >= topk else float(ranked.mean())
    spread = topk_mean - bottomk_mean

    if topk_mean > 0.05 and spread > 0.08:
        signal = "BULLISH"
        signal_desc = "Strong positive score spread, multiple high-conviction opportunities"
    elif topk_mean > 0.02:
        signal = "MILDLY_BULLISH"
        signal_desc = "Moderate positive scores, some differentiation among stocks"
    elif topk_mean < -0.02:
        signal = "BEARISH"
        signal_desc = "Negative top-K scores, consider reducing exposure"
    else:
        signal = "NEUTRAL"
        signal_desc = "Low score spread, limited alpha opportunities"

    return {
        "buys": buys,
        "sells": sells,
        "holds": holds,
        "target_portfolio": new_topk,
        "market_signal": signal,
        "market_signal_description": signal_desc,
        "score_stats": {
            "mean": mean_score,
            "median": median_score,
            "top_k_mean": topk_mean,
            "bottom_k_mean": bottomk_mean,
            "spread": spread,
        },
    }


def format_symbol_list(symbols: List[str], stock_name_map: Dict[str, str], limit: int = 10) -> List[str]:
    items = [stock_label(str(symbol), stock_name_map) for symbol in symbols[:limit]]
    return items


def format_report(
    date_str: str,
    requested_date_str: Optional[str],
    scores: pd.DataFrame,
    prices: pd.Series,
    recommendations: Dict,
    topk: int,
    stock_name_map: Dict[str, str],
    recorder_id: str,
    experiment_name: str,
) -> Tuple[str, Dict]:
    lines = []
    lines.append("=" * 96)
    lines.append(f"Daily Score Report - {date_str}")
    lines.append("=" * 96)
    lines.append(f"Experiment: {experiment_name}")
    lines.append(f"Recorder ID: {recorder_id}")
    lines.append(f"Stock Name Map: {os.path.basename(DEFAULT_STOCK_NAME_MAP_PATH)}")
    if requested_date_str is not None:
        lines.append(f"Requested Report Date: {requested_date_str}")
        if requested_date_str != date_str:
            lines.append(f"Actual Prediction Date: {date_str}")

    lines.append(f"\nMarket Signal: {recommendations['market_signal']}")
    lines.append(f"  {recommendations['market_signal_description']}")

    stats = recommendations["score_stats"]
    lines.append("\nScore Statistics:")
    lines.append(f"  Mean: {stats['mean']:.4f}  Median: {stats['median']:.4f}")
    lines.append(f"  Top-{topk} mean: {stats['top_k_mean']:.4f}  Bottom-{topk} mean: {stats['bottom_k_mean']:.4f}")
    lines.append(f"  Spread: {stats['spread']:.4f}")

    lines.append(f"\nTop-{topk} Stock Scores:")
    lines.append(f"{'Rank':<6}{'Code':<12}{'Name':<16}{'Score':<12}{'Close':<12}")
    lines.append("-" * 58)

    top_scores = []
    for rank, (inst, row) in enumerate(scores.head(topk).iterrows(), 1):
        inst_str = str(inst)
        score_val = float(row["score"])
        price_val = prices.get(inst, float('nan'))
        price_str = f"{price_val:.2f}" if pd.notna(price_val) else "N/A"
        name = stock_name_map.get(inst_str, "-")
        lines.append(f"{rank:<6}{inst_str:<12}{name:<16}{score_val:<12.6f}{price_str:<12}")
        top_scores.append(
            {
                "rank": rank,
                "instrument": inst_str,
                "name": None if name == "-" else name,
                "score": score_val,
                "close": float(price_val) if pd.notna(price_val) else None,
            }
        )

    lines.append(f"\nStrategy Recommendations (topk={topk}):")
    if recommendations["buys"]:
        buys = ", ".join(format_symbol_list(recommendations["buys"], stock_name_map))
        lines.append(f"  BUY  ({len(recommendations['buys'])}): {buys}")
        if len(recommendations["buys"]) > 10:
            lines.append(f"        ... and {len(recommendations['buys']) - 10} more")
    if recommendations["sells"]:
        sells = ", ".join(format_symbol_list(recommendations["sells"], stock_name_map))
        lines.append(f"  SELL ({len(recommendations['sells'])}): {sells}")
        if len(recommendations["sells"]) > 10:
            lines.append(f"        ... and {len(recommendations['sells']) - 10} more")
    if recommendations["holds"]:
        holds = ", ".join(format_symbol_list(recommendations["holds"], stock_name_map))
        lines.append(f"  HOLD ({len(recommendations['holds'])}): {holds}")

    lines.append("=" * 96)

    json_report = {
        "date": date_str,
        "requested_date": requested_date_str,
        "generated_at": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "recorder_id": recorder_id,
        "market_signal": recommendations["market_signal"],
        "market_signal_description": recommendations["market_signal_description"],
        "score_stats": {k: float(v) for k, v in stats.items()},
        "top_scores": top_scores,
        "strategy": {
            "buys": [
                {"instrument": str(s), "name": stock_name_map.get(str(s).upper())}
                for s in recommendations["buys"]
            ],
            "sells": [
                {"instrument": str(s), "name": stock_name_map.get(str(s).upper())}
                for s in recommendations["sells"]
            ],
            "holds": [
                {"instrument": str(s), "name": stock_name_map.get(str(s).upper())}
                for s in recommendations["holds"]
            ],
            "target_portfolio": [
                {"instrument": str(s), "name": stock_name_map.get(str(s).upper())}
                for s in recommendations["target_portfolio"]
            ],
        },
        "total_instruments_scored": len(scores),
        "stock_name_map_path": os.path.abspath(DEFAULT_STOCK_NAME_MAP_PATH),
    }

    return "\n".join(lines), json_report


def run_pipeline(
    recorder_id: Optional[str],
    experiment_name: str = "workflow",
    topk: int = 20,
    n_drop: int = 2,
    previous_holdings: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    qlib_data_dir: Optional[str] = None,
    score_date: Optional[str] = None,
    stock_name_map_path: Optional[str] = None,
):
    global QLIB_DATA_DIR  # pylint: disable=global-statement
    QLIB_DATA_DIR = os.path.expanduser(qlib_data_dir or QLIB_DATA_DIR)

    init_qlib(provider_uri=QLIB_DATA_DIR)

    cal_last = get_latest_calendar_date(QLIB_DATA_DIR)
    latest_date = score_date or cal_last
    if latest_date is None:
        raise ValueError("Cannot determine latest trading date (calendar missing and no --score-date).")

    if score_date:
        print(f"\nUsing --score-date as window end: {latest_date}  (calendar file last day: {cal_last})")
    else:
        print(f"\nLatest trading date in data (calendars/day.txt): {latest_date}")

    if recorder_id is None:
        recorder_id = resolve_latest_recorder_id(experiment_name=experiment_name)

    stock_name_map = load_stock_name_map(stock_name_map_path)

    scores, pred_date = load_model_and_predict(
        recorder_id=recorder_id,
        experiment_name=experiment_name,
        latest_date=latest_date,
        target_date=score_date,
    )
    date_str = str(pred_date)[:10]
    print(f"Predictions generated for: {date_str}")
    print(f"Total instruments scored: {len(scores)}")

    top_instruments = list(scores.head(topk).index)
    prices = get_latest_prices([str(i) for i in top_instruments], date_str)
    recommendations = generate_strategy_recommendations(
        scores=scores,
        topk=topk,
        n_drop=n_drop,
        previous_holdings=previous_holdings,
    )

    console_text, json_report = format_report(
        date_str=date_str,
        requested_date_str=score_date,
        scores=scores,
        prices=prices,
        recommendations=recommendations,
        topk=topk,
        stock_name_map=stock_name_map,
        recorder_id=recorder_id,
        experiment_name=experiment_name,
    )
    print("\n" + console_text)

    report_dir = output_dir or REPORT_DIR
    report_path = os.path.join(report_dir, f"{date_str}.json")
    if dry_run:
        print("\n[DRY RUN] Report would be saved to", report_path)
    else:
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {report_path}")

    return json_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Daily scoring pipeline based on a trained qlib recorder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--recorder-id",
        default=None,
        help="Recorder ID of the trained model. If omitted, use the latest finished recorder with predictions.",
    )
    parser.add_argument("--experiment-name", default="workflow")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n-drop", type=int, default=2)
    parser.add_argument("--output-dir", default=None, help="Directory for saved JSON reports.")
    parser.add_argument("--dry-run", action="store_true", help="Run scoring without saving report.")
    parser.add_argument(
        "--qlib-data-dir",
        default=None,
        help="Qlib data directory (default: ~/.qlib/qlib_data/cn_data).",
    )
    parser.add_argument(
        "--score-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Generate a daily report for the specified date. If that date is unavailable, fallback to the nearest earlier prediction date.",
    )
    parser.add_argument(
        "--stock-name-map",
        default=DEFAULT_STOCK_NAME_MAP_PATH,
        help="Local JSON file used to map stock code to stock name.",
    )
    parser.add_argument(
        "--refresh-stock-name-map",
        action="store_true",
        help="Refresh the local stock name map from baostock before scoring.",
    )
    parser.add_argument(
        "--refresh-stock-name-map-only",
        action="store_true",
        help="Only refresh the local stock name map from baostock, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_stock_name_map or args.refresh_stock_name_map_only:
        refresh_stock_name_map_via_baostock(args.stock_name_map)
        if args.refresh_stock_name_map_only:
            return

    run_pipeline(
        recorder_id=args.recorder_id,
        experiment_name=args.experiment_name,
        topk=args.topk,
        n_drop=args.n_drop,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        qlib_data_dir=args.qlib_data_dir,
        score_date=args.score_date,
        stock_name_map_path=args.stock_name_map,
    )


if __name__ == "__main__":
    main()
