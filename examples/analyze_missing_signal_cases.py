#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from analyze_missing_signal_alpha import (
    build_matrices,
    load_inputs,
    simulate_strategy,
)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    return float(drawdown.min())


def analyze_cases(args):
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
    stock_summary = contribution.sum().sort_values(ascending=False)
    instruments: List[str] = stock_summary.head(args.top_n).index.tolist()

    rows = []
    for instrument in instruments:
        gap_mask = diff_weights[instrument] > 1e-12
        if not gap_mask.any():
            continue
        start_date = gap_mask[gap_mask].index[0]
        end_date = gap_mask[gap_mask].index[-1]
        signal_series = signal_df[instrument]
        close_series = close_df[instrument]
        contrib_series = contribution[instrument]
        next_ret_series = keep_result["next_returns"][instrument]
        score_rank = keep_result["score_rank"][instrument]

        prior_signal = signal_series.loc[:start_date].dropna()
        last_signal_date = prior_signal.index[-1] if not prior_signal.empty else pd.NaT
        last_signal_value = float(prior_signal.iloc[-1]) if not prior_signal.empty else np.nan
        last_rank = float(score_rank.loc[last_signal_date]) if pd.notna(last_signal_date) else np.nan

        window_close = close_series.loc[start_date:end_date].dropna()
        window_returns = next_ret_series.loc[start_date:end_date]
        after_start_signals = signal_series.loc[start_date:end_date]
        after_end_signals = signal_series.loc[start_date:]

        if len(window_close) >= 2:
            hold_return = float(window_close.iloc[-1] / window_close.iloc[0] - 1.0)
        else:
            hold_return = np.nan

        if after_end_signals.notna().sum() == 0:
            missing_pattern = "permanent_missing_after_start"
        elif after_start_signals.notna().sum() == 0:
            missing_pattern = "missing_until_event_end"
        else:
            missing_pattern = "intermittent_missing"

        rows.append(
            {
                "instrument": instrument,
                "event_start": start_date,
                "event_end": end_date,
                "event_days": int(gap_mask.sum()),
                "last_signal_date": last_signal_date,
                "last_signal_value": last_signal_value,
                "last_rank_before_missing": last_rank,
                "missing_days_during_event": int(keep_result["missing_mask"][instrument].loc[start_date:end_date].sum()),
                "price_return_during_event": hold_return,
                "event_excess_contribution": float(contrib_series.loc[start_date:end_date].sum()),
                "best_day_contribution": float(contrib_series.loc[start_date:end_date].max()),
                "worst_day_contribution": float(contrib_series.loc[start_date:end_date].min()),
                "max_drawdown_during_event": _max_drawdown_from_returns(window_returns),
                "missing_pattern": missing_pattern,
                "close_data_days_after_start": int(close_series.loc[start_date:].notna().sum()),
                "signal_days_after_start": int(after_end_signals.notna().sum()),
            }
        )

    case_df = pd.DataFrame(rows).sort_values("event_excess_contribution", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "missing_signal_case_studies.csv")
    html_path = os.path.join(args.output_dir, "missing_signal_case_studies.html")
    case_df.to_csv(csv_path, index=False)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=len(instruments),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.06,
            subplot_titles=[f"{inst}" for inst in instruments],
        )
        for idx, instrument in enumerate(instruments, start=1):
            price = close_df[instrument].dropna()
            if price.empty:
                continue
            price_norm = price / price.iloc[0]
            fig.add_trace(
                go.Scatter(x=price_norm.index, y=price_norm, mode="lines", name=f"{instrument} price"),
                row=idx,
                col=1,
            )
            gap_mask = diff_weights[instrument] > 1e-12
            gap_dates = gap_mask[gap_mask].index
            if len(gap_dates) > 0:
                fig.add_vrect(
                    x0=gap_dates[0],
                    x1=gap_dates[-1],
                    fillcolor="LightSalmon",
                    opacity=0.2,
                    line_width=0,
                    row=idx,
                    col=1,
                )
                contrib = contribution[instrument].loc[gap_dates]
                fig.add_trace(
                    go.Bar(
                        x=contrib.index,
                        y=contrib.values,
                        name=f"{instrument} contribution",
                        marker_color="rgba(31, 119, 180, 0.35)",
                    ),
                    row=idx,
                    col=1,
                )
        fig.update_layout(height=320 * max(len(instruments), 1), width=1300, title_text="Missing Signal Case Studies")
        html_parts = [
            fig.to_html(include_plotlyjs="cdn", full_html=False),
            "<h2>Case Summary</h2>",
            case_df.to_html(index=False),
        ]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
    except ImportError:
        pass

    print("=" * 80)
    print("Missing signal case studies")
    print("=" * 80)
    print(f"Recorder ID: {recorder.id}")
    print(case_df.to_string(index=False))
    print()
    print(f"Saved: {csv_path}")
    if os.path.exists(html_path):
        print(f"Saved: {html_path}")


def parse_args():
    from workflow_backtrader_bridge import BACKTEST_CONFIG

    parser = argparse.ArgumentParser(description="Case study for top missing-signal contributors.")
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
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--output-dir", default="backtest_log/missing_signal_analysis")
    return parser.parse_args()


def main():
    analyze_cases(parse_args())


if __name__ == "__main__":
    main()
