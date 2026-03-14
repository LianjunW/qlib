#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from workflow_backtrader_bridge import (
    BACKTEST_CONFIG,
    build_feed_frames,
    init_qlib,
    resolve_recorder,
)


def load_inputs(
    experiment_name: str,
    recorder_id: Optional[str],
    topk: int,
    start_time: Optional[str],
    end_time: Optional[str],
    max_instruments: Optional[int],
) -> Tuple[object, Dict[str, pd.DataFrame]]:
    init_qlib()
    recorder = resolve_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
    pred = recorder.load_object("pred.pkl")
    feed_frames = build_feed_frames(
        pred=pred,
        start_time=start_time or BACKTEST_CONFIG.get("start_time"),
        end_time=end_time or BACKTEST_CONFIG.get("end_time"),
        topk=topk,
        max_instruments=max_instruments,
    )
    return recorder, feed_frames


def build_matrices(feed_frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close_dict = {}
    signal_dict = {}
    for instrument, frame in feed_frames.items():
        close_dict[instrument] = frame["close"]
        signal_dict[instrument] = frame["signal_score"]

    close_df = pd.DataFrame(close_dict).sort_index()
    signal_df = pd.DataFrame(signal_dict).sort_index().reindex(close_df.index)
    close_df = close_df.reindex(sorted(close_df.columns), axis=1)
    signal_df = signal_df.reindex(close_df.columns, axis=1)
    return close_df, signal_df


def simulate_strategy(
    close_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    risk_degree: float,
    sell_missing_signal: bool,
    missing_signal_sell_after: int,
) -> Dict[str, pd.DataFrame]:
    dates = close_df.index
    instruments = list(close_df.columns)
    weights = pd.DataFrame(0.0, index=dates, columns=instruments)
    active_mask = pd.DataFrame(False, index=dates, columns=instruments)
    missing_mask = pd.DataFrame(False, index=dates, columns=instruments)
    sell_mask = pd.DataFrame(False, index=dates, columns=instruments)
    score_rank = pd.DataFrame(np.nan, index=dates, columns=instruments)

    current_weights: Dict[str, float] = {}
    hold_days: Dict[str, int] = {}
    missing_signal_days: Dict[str, int] = {}

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
            else:
                missing_signal_days[name] = missing_signal_days.get(name, 0) + 1
                missing_mask.at[date, name] = True

        stale_holdings: List[str] = []
        if sell_missing_signal:
            stale_holdings = [
                name
                for name in holdings
                if missing_signal_days.get(name, 0) >= missing_signal_sell_after
            ]
            for name in stale_holdings:
                sell_mask.at[date, name] = True

        if score_map:
            ranked = pd.Series(score_map).sort_values(ascending=False)
            score_rank.loc[date, ranked.index] = np.arange(1, len(ranked) + 1)

            current_rank = pd.Series({name: score_map.get(name, np.nan) for name in holdings}).dropna()
            last = pd.Index(current_rank.sort_values(ascending=False).index)

            candidate_count = max(n_drop + topk - len(last), 0)
            today = pd.Index(ranked.index[~ranked.index.isin(last)][:candidate_count])

            combined_names = list(last) + [name for name in today if name not in last]
            combined_rank = ranked.reindex(combined_names).dropna().sort_values(ascending=False)
            bottom = pd.Index(combined_rank.index[-n_drop:]) if len(combined_rank) else pd.Index([])
            sell_candidates = last[last.isin(bottom)]
            allowed_sells = [name for name in sell_candidates if hold_days.get(name, 0) >= hold_thresh]
            retained = [
                name for name in holdings if name not in allowed_sells and name not in stale_holdings
            ]
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
            active_mask.loc[date, list(current_weights.keys())] = True
            for name, weight in current_weights.items():
                weights.at[date, name] = weight

        exited = [name for name in list(hold_days.keys()) if name not in current_weights]
        for name in exited:
            hold_days.pop(name, None)
            missing_signal_days.pop(name, None)

    next_returns = close_df.pct_change(fill_method=None).shift(-1).fillna(0.0)
    portfolio_returns = (weights * next_returns).sum(axis=1)
    return {
        "weights": weights,
        "active_mask": active_mask,
        "missing_mask": missing_mask,
        "sell_mask": sell_mask,
        "score_rank": score_rank,
        "next_returns": next_returns,
        "portfolio_returns": portfolio_returns,
    }


def build_event_table(
    diff_weights: pd.DataFrame,
    keep_result: Dict[str, pd.DataFrame],
    contribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dates = list(diff_weights.index)
    for instrument in diff_weights.columns:
        active = diff_weights[instrument] > 1e-12
        if not active.any():
            continue
        start_idx = None
        for i, is_active in enumerate(active):
            if is_active and start_idx is None:
                start_idx = i
            if start_idx is not None and (not is_active or i == len(active) - 1):
                end_idx = i if is_active and i == len(active) - 1 else i - 1
                event_dates = dates[start_idx : end_idx + 1]
                event_contrib = contribution.loc[event_dates, instrument]
                missing_days = int(keep_result["missing_mask"].loc[event_dates, instrument].sum())
                rows.append(
                    {
                        "instrument": instrument,
                        "start_date": event_dates[0],
                        "end_date": event_dates[-1],
                        "days": len(event_dates),
                        "missing_days": missing_days,
                        "start_weight_gap": float(diff_weights.loc[event_dates[0], instrument]),
                        "excess_return_contrib": float(event_contrib.sum()),
                        "peak_daily_contrib": float(event_contrib.max()),
                        "worst_daily_contrib": float(event_contrib.min()),
                    }
                )
                start_idx = None
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows)
    return events.sort_values("excess_return_contrib", ascending=False).reset_index(drop=True)


def save_html_report(
    output_path: str,
    daily_summary: pd.DataFrame,
    stock_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=("Cumulative Excess Contribution", "Top Stock Contributions"),
    )
    fig.add_trace(
        go.Scatter(
            x=daily_summary.index,
            y=daily_summary["cum_excess_contrib"],
            mode="lines",
            name="cum_excess_contrib",
        ),
        row=1,
        col=1,
    )

    top_stocks = stock_summary.head(15).sort_values("excess_return_contrib")
    fig.add_trace(
        go.Bar(
            x=top_stocks["excess_return_contrib"],
            y=top_stocks["instrument"],
            orientation="h",
            name="top_stock_contrib",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=900, width=1200, title_text="Missing Signal Excess Return Analysis")

    summary_html = [
        "<h2>Top Events</h2>",
        event_summary.head(20).to_html(index=False),
        "<h2>Top Stocks</h2>",
        stock_summary.head(20).to_html(index=False),
    ]
    html = fig.to_html(include_plotlyjs="cdn", full_html=False) + "\n".join(summary_html)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


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

    diff_weights = (keep_result["weights"] - sell_result["weights"]).clip(lower=0.0)
    contribution = diff_weights * keep_result["next_returns"]
    daily_summary = pd.DataFrame(
        {
            "keep_return": keep_result["portfolio_returns"],
            "sell_return": sell_result["portfolio_returns"],
            "excess_return_contrib": contribution.sum(axis=1),
        }
    )
    daily_summary["cum_keep"] = (1.0 + daily_summary["keep_return"]).cumprod()
    daily_summary["cum_sell"] = (1.0 + daily_summary["sell_return"]).cumprod()
    daily_summary["cum_excess_contrib"] = daily_summary["excess_return_contrib"].cumsum()

    stock_summary = pd.DataFrame(
        {
            "instrument": contribution.columns,
            "excess_return_contrib": contribution.sum(axis=0).values,
            "days_with_gap": (diff_weights > 1e-12).sum(axis=0).values,
            "missing_days_while_held": keep_result["missing_mask"].sum(axis=0).reindex(contribution.columns).values,
            "avg_weight_gap": diff_weights.replace(0.0, np.nan).mean(axis=0).fillna(0.0).values,
            "max_weight_gap": diff_weights.max(axis=0).values,
        }
    )
    stock_summary = stock_summary[stock_summary["days_with_gap"] > 0]
    stock_summary = stock_summary.sort_values("excess_return_contrib", ascending=False).reset_index(drop=True)

    day_top = (
        daily_summary[["excess_return_contrib"]]
        .rename_axis("date")
        .sort_values("excess_return_contrib", ascending=False)
        .reset_index()
    )
    day_bottom = (
        daily_summary[["excess_return_contrib"]]
        .rename_axis("date")
        .sort_values("excess_return_contrib", ascending=True)
        .reset_index()
    )
    event_summary = build_event_table(diff_weights, keep_result, contribution)
    pair_summary = pd.concat(
        [
            contribution.stack().rename("excess_return_contrib"),
            diff_weights.stack().rename("weight_gap"),
            keep_result["next_returns"].stack().rename("next_return"),
            keep_result["missing_mask"].stack().rename("missing_while_held"),
        ],
        axis=1,
    ).reset_index()
    pair_summary = pair_summary.rename(columns={"level_0": "date", "level_1": "instrument"})
    pair_summary = pair_summary[pair_summary["weight_gap"] > 1e-12].sort_values(
        "excess_return_contrib", ascending=False
    )

    os.makedirs(args.output_dir, exist_ok=True)
    daily_path = os.path.join(args.output_dir, "missing_signal_daily_summary.csv")
    stock_path = os.path.join(args.output_dir, "missing_signal_stock_summary.csv")
    event_path = os.path.join(args.output_dir, "missing_signal_event_summary.csv")
    pair_path = os.path.join(args.output_dir, "missing_signal_pair_summary.csv")
    html_path = os.path.join(args.output_dir, "missing_signal_analysis.html")

    daily_summary.to_csv(daily_path, index_label="date")
    stock_summary.to_csv(stock_path, index=False)
    event_summary.to_csv(event_path, index=False)
    pair_summary.to_csv(pair_path, index=False)
    save_html_report(html_path, daily_summary, stock_summary, event_summary)

    print("=" * 80)
    print("Missing signal excess return analysis")
    print("=" * 80)
    print(f"Recorder ID: {recorder.id}")
    print(f"Compared against sell-after threshold: {args.sell_missing_signal_after} days")
    print(f"Trading dates: {len(daily_summary)}")
    print(f"Instrument universe: {len(close_df.columns)}")
    print(f"Keep simulated total return: {(daily_summary['cum_keep'].iloc[-1] - 1.0):.2%}")
    print(f"Sell simulated total return: {(daily_summary['cum_sell'].iloc[-1] - 1.0):.2%}")
    print(f"Cumulative excess contribution: {daily_summary['cum_excess_contrib'].iloc[-1]:.4f}")
    print()
    print("Top contributing stocks:")
    print(stock_summary.head(args.top_rows).to_string(index=False))
    print()
    print("Top contribution dates:")
    print(day_top.head(args.top_rows).to_string(index=False))
    print()
    print("Top stock-date contribution pairs:")
    print(pair_summary.head(args.top_rows).to_string(index=False))
    print()
    print("Worst contribution dates:")
    print(day_bottom.head(args.top_rows).to_string(index=False))
    print()
    print("Top missing-signal holding events:")
    print(event_summary.head(args.top_rows).to_string(index=False))
    print()
    print(f"Saved: {daily_path}")
    print(f"Saved: {stock_path}")
    print(f"Saved: {event_path}")
    print(f"Saved: {pair_path}")
    if os.path.exists(html_path):
        print(f"Saved: {html_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze excess returns from keeping missing-signal holdings.")
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
    parser.add_argument("--top-rows", type=int, default=15)
    parser.add_argument("--output-dir", default="backtest_log/missing_signal_analysis")
    return parser.parse_args()


def main():
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
