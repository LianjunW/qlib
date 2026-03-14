#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from analyze_missing_signal_alpha import build_matrices, load_inputs, simulate_strategy
from workflow_backtrader_bridge import annualize_returns, init_qlib
from qlib.data import D


def build_membership_matrix(index: pd.Index, columns: pd.Index, market: str) -> pd.DataFrame:
    init_qlib()
    spans: Dict[str, list] = D.list_instruments(
        D.instruments(market),
        start_time=index.min(),
        end_time=index.max(),
        as_list=False,
        freq="day",
    )
    membership = pd.DataFrame(False, index=index, columns=columns)
    for instrument in columns:
        for start, end in spans.get(instrument, []):
            membership.loc[(membership.index >= start) & (membership.index <= end), instrument] = True
    return membership


def simulate_hybrid_strategy(
    close_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    risk_degree: float,
    missing_signal_sell_after: int,
):
    dates = close_df.index
    instruments = list(close_df.columns)
    weights = pd.DataFrame(0.0, index=dates, columns=instruments)

    current_weights = {}
    hold_days = {}
    missing_signal_days = {}

    for date in dates:
        close_row = close_df.loc[date]
        signal_row = signal_df.loc[date]
        valid_mask = signal_row.notna() & close_row.notna() & (close_row > 0)
        score_map = signal_row[valid_mask].astype(float).to_dict()

        holdings = [name for name, weight in current_weights.items() if weight > 0]
        for name in holdings:
            hold_days[name] = hold_days.get(name, 0) + 1
            if name in score_map:
                missing_signal_days.pop(name, None)
                continue

            # Hybrid rule: keep holdings after they leave the universe,
            # but still force-sell prolonged missing signals inside the universe.
            if membership_df.at[date, name]:
                missing_signal_days[name] = missing_signal_days.get(name, 0) + 1
            else:
                missing_signal_days.pop(name, None)

        stale_holdings = [
            name
            for name in holdings
            if membership_df.at[date, name]
            and missing_signal_days.get(name, 0) >= missing_signal_sell_after
        ]

        if score_map:
            ranked = pd.Series(score_map).sort_values(ascending=False)
            current_rank = pd.Series({name: score_map.get(name, np.nan) for name in holdings}).dropna()
            last = pd.Index(current_rank.sort_values(ascending=False).index)

            candidate_count = max(n_drop + topk - len(last), 0)
            today = pd.Index(ranked.index[~ranked.index.isin(last)][:candidate_count])
            combined_names = list(last) + [name for name in today if name not in last]
            combined_rank = ranked.reindex(combined_names).dropna().sort_values(ascending=False)
            bottom = pd.Index(combined_rank.index[-n_drop:]) if len(combined_rank) else pd.Index([])
            sell_candidates = last[last.isin(bottom)]
            allowed_sells = [name for name in sell_candidates if hold_days.get(name, 0) >= hold_thresh]
            retained = [name for name in holdings if name not in allowed_sells and name not in stale_holdings]
            buy_slots = max(topk - len(retained), 0)
            buy_names = [name for name in today if name not in retained][:buy_slots]
            target_names = retained + [name for name in buy_names if name not in retained]
            target_weight = risk_degree / len(target_names) if target_names else 0.0
            current_weights = {name: target_weight for name in target_names}
        elif stale_holdings:
            for name in stale_holdings:
                current_weights.pop(name, None)
                hold_days.pop(name, None)
                missing_signal_days.pop(name, None)

        if current_weights:
            for name, weight in current_weights.items():
                weights.at[date, name] = weight

        exited = [name for name in list(hold_days.keys()) if name not in current_weights]
        for name in exited:
            hold_days.pop(name, None)
            missing_signal_days.pop(name, None)

    next_returns = close_df.pct_change(fill_method=None).shift(-1).fillna(0.0)
    portfolio_returns = (weights * next_returns).sum(axis=1)
    return {"weights": weights, "next_returns": next_returns, "portfolio_returns": portfolio_returns}


def summarize(label: str, returns: pd.Series):
    total = float((1.0 + returns).prod() - 1.0)
    annual = annualize_returns(returns)
    return {"label": label, "total_return": total, "annual_return": annual}


def run_analysis(args):
    recorder, feed_frames = load_inputs(
        experiment_name=args.experiment_name,
        recorder_id=args.recorder_id,
        topk=args.topk,
        start_time=args.start_time,
        end_time=args.end_time,
        max_instruments=args.max_instruments,
    )
    close_df, signal_df = build_matrices(feed_frames)
    membership_df = build_membership_matrix(close_df.index, close_df.columns, args.market)

    keep_result = simulate_strategy(
        close_df=close_df,
        signal_df=signal_df,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        risk_degree=args.risk_degree,
        sell_missing_signal=False,
        missing_signal_sell_after=args.sell_missing_signal_after,
    )
    sell_result = simulate_strategy(
        close_df=close_df,
        signal_df=signal_df,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        risk_degree=args.risk_degree,
        sell_missing_signal=True,
        missing_signal_sell_after=args.sell_missing_signal_after,
    )
    hybrid_result = simulate_hybrid_strategy(
        close_df=close_df,
        signal_df=signal_df,
        membership_df=membership_df,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        risk_degree=args.risk_degree,
        missing_signal_sell_after=args.sell_missing_signal_after,
    )

    diff_weights = (keep_result["weights"] - sell_result["weights"]).clip(lower=0.0)
    contribution = diff_weights * keep_result["next_returns"]
    reason_df = pd.concat(
        [
            contribution.stack().rename("excess_return_contrib"),
            diff_weights.stack().rename("weight_gap"),
            membership_df.stack().rename("in_universe"),
        ],
        axis=1,
    ).reset_index()
    reason_df = reason_df.rename(columns={"level_0": "date", "level_1": "instrument"})
    reason_df = reason_df[reason_df["weight_gap"] > 1e-12].copy()
    reason_df["reason"] = np.where(reason_df["in_universe"], "in_universe_missing", "out_of_universe_missing")

    reason_summary = (
        reason_df.groupby("reason")
        .agg(
            excess_return_contrib=("excess_return_contrib", "sum"),
            stock_date_pairs=("instrument", "count"),
            instrument_count=("instrument", "nunique"),
        )
        .reset_index()
        .sort_values("excess_return_contrib", ascending=False)
    )

    stock_reason_summary = (
        reason_df.pivot_table(
            index="instrument",
            columns="reason",
            values="excess_return_contrib",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    if "out_of_universe_missing" not in stock_reason_summary.columns:
        stock_reason_summary["out_of_universe_missing"] = 0.0
    if "in_universe_missing" not in stock_reason_summary.columns:
        stock_reason_summary["in_universe_missing"] = 0.0
    stock_reason_summary["total_excess"] = (
        stock_reason_summary["out_of_universe_missing"] + stock_reason_summary["in_universe_missing"]
    )
    stock_reason_summary = stock_reason_summary.sort_values("total_excess", ascending=False)

    perf_summary = pd.DataFrame(
        [
            summarize("keep_all_missing", keep_result["portfolio_returns"]),
            summarize("sell_all_missing", sell_result["portfolio_returns"]),
            summarize("hybrid_keep_out_of_universe", hybrid_result["portfolio_returns"]),
        ]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    reason_path = os.path.join(args.output_dir, "missing_signal_reason_summary.csv")
    stock_reason_path = os.path.join(args.output_dir, "missing_signal_stock_reason_summary.csv")
    perf_path = os.path.join(args.output_dir, "missing_signal_reason_performance.csv")
    reason_df_path = os.path.join(args.output_dir, "missing_signal_reason_pairs.csv")

    reason_summary.to_csv(reason_path, index=False)
    stock_reason_summary.to_csv(stock_reason_path, index=False)
    perf_summary.to_csv(perf_path, index=False)
    reason_df.to_csv(reason_df_path, index=False)

    print("=" * 80)
    print("Missing signal reason analysis")
    print("=" * 80)
    print(f"Recorder ID: {recorder.id}")
    print(f"Universe market checked: {args.market}")
    print()
    print("Performance comparison:")
    print(perf_summary.to_string(index=False))
    print()
    print("Contribution by reason:")
    print(reason_summary.to_string(index=False))
    print()
    print("Top stocks by reason split:")
    print(stock_reason_summary.head(args.top_rows).to_string(index=False))
    print()
    print(f"Saved: {reason_path}")
    print(f"Saved: {stock_reason_path}")
    print(f"Saved: {perf_path}")
    print(f"Saved: {reason_df_path}")


def parse_args():
    from workflow_backtrader_bridge import BACKTEST_CONFIG

    parser = argparse.ArgumentParser(description="Split missing-signal alpha by out-of-universe vs in-universe causes.")
    parser.add_argument("--experiment-name", default="workflow")
    parser.add_argument("--recorder-id", default=None)
    parser.add_argument("--start-time", default=BACKTEST_CONFIG.get("start_time"))
    parser.add_argument("--end-time", default=BACKTEST_CONFIG.get("end_time"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n-drop", type=int, default=2)
    parser.add_argument("--hold-thresh", type=int, default=1)
    parser.add_argument("--risk-degree", type=float, default=0.95)
    parser.add_argument("--max-instruments", type=int, default=None)
    parser.add_argument("--sell-missing-signal-after", type=int, default=20)
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--top-rows", type=int, default=15)
    parser.add_argument("--output-dir", default="backtest_log/missing_signal_analysis")
    return parser.parse_args()


def main():
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
