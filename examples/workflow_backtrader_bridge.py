#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qlib -> Backtrader bridge example.

This script keeps qlib responsible for training/prediction generation and uses
the recorder artifact `pred.pkl` as the signal source for a backtrader run.
"""

import argparse
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.tests.data import GetData
from qlib.workflow import R

try:
    import backtrader as bt
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "backtrader is not installed in the current environment. "
        "Please install it first, for example:\n"
        "  /root/projects/qlib/.venv/bin/pip install -e /root/projects/backtrader"
    ) from exc

try:
    from workflow_by_code_v2 import BACKTEST_CONFIG
except ImportError:
    BACKTEST_CONFIG = {
        "start_time": "2024-01-01",
        "end_time": "2025-08-01",
        "account": 100000000,
        "exchange_kwargs": {
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }


PRICE_FIELDS = ["$open", "$high", "$low", "$close", "$volume"]


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return None
    return float((1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0)


def load_benchmark_returns(
    benchmark: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
) -> Tuple[Optional[str], pd.Series]:
    if not benchmark:
        return None, pd.Series(dtype=float)

    benchmark_data = D.features(
        [benchmark],
        ["$close"],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    if benchmark_data is None or benchmark_data.empty:
        return benchmark, pd.Series(dtype=float)

    benchmark_frame = benchmark_data.xs(benchmark, level="instrument").sort_index()
    close_column = next((col for col in benchmark_frame.columns if col in ("$close", "close")), None)
    if close_column is None:
        return benchmark, pd.Series(dtype=float)

    benchmark_returns = benchmark_frame[close_column].astype(float).pct_change().dropna()
    return benchmark, benchmark_returns


def build_plot_frame(summary: Dict[str, object]) -> pd.DataFrame:
    curves = {}
    equity_curve = summary.get("equity_curve")
    if equity_curve is not None and len(equity_curve) > 0:
        curves["strategy"] = equity_curve / summary["initial_cash"]

    benchmark_curve = summary.get("benchmark_curve")
    if benchmark_curve is not None and len(benchmark_curve) > 0:
        curves[str(summary.get("benchmark_name") or "benchmark")] = benchmark_curve / summary["initial_cash"]

    if not curves:
        return pd.DataFrame()

    return pd.DataFrame(curves).dropna(how="all")


def plot_summary(summary: Dict[str, object], output_path: Optional[str] = None):
    plot_df = build_plot_frame(summary)
    if plot_df.empty:
        print("No equity data available for plotting.")
        return

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        suffix = os.path.splitext(output_path)[1].lower()
        if suffix == ".html":
            try:
                import plotly.graph_objects as go
            except ImportError:
                print("plotly is not installed; cannot save HTML plot.")
                return

            fig = go.Figure()
            for column in plot_df.columns:
                fig.add_trace(
                    go.Scatter(x=plot_df.index, y=plot_df[column], mode="lines", name=column)
                )
            fig.update_layout(
                title="Strategy vs Benchmark NAV",
                xaxis_title="Date",
                yaxis_title="Normalized NAV",
                template="plotly_white",
            )
            fig.write_html(output_path)
            print(f"Saved plot to {output_path}")
            return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot output.")
        return

    ax = plot_df.plot(figsize=(12, 5), title="Strategy vs Benchmark NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")
        return

    plt.show()


class QlibPredictionData(bt.feeds.PandasData):
    lines = ("signal_score",)
    params = (("signal_score", -1),)


class CNStockCommission(bt.CommInfoBase):
    params = (
        ("buy_commission", 0.0005),
        ("sell_commission", 0.0015),
        ("min_commission", 5.0),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if not size:
            return 0.0
        rate = self.p.buy_commission if size > 0 else self.p.sell_commission
        return max(abs(size) * price * rate, self.p.min_commission)


class QlibTopKStrategy(bt.Strategy):
    params = (
        ("topk", 20),
        ("n_drop", 2),
        ("hold_thresh", 1),
        ("risk_degree", 0.95),
        ("sell_missing_signal", False),
        ("missing_signal_sell_after", 3),
        ("verbose", False),
    )

    def __init__(self):
        self.data_by_name = {data._name: data for data in self.datas}
        self.hold_days: Dict[str, int] = {}
        self.missing_signal_days: Dict[str, int] = {}
        self.last_rebalance_date = None

    def log(self, message: str):
        if self.p.verbose:
            print(f"[{self.datetime.date(0)}] {message}")

    def prenext(self):
        self.next()

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self.hold_days[order.data._name] = 0
            self.log(
                f"BUY {order.data._name} size={order.executed.size:.0f} "
                f"price={order.executed.price:.4f}"
            )
        elif order.status == order.Completed and order.issell():
            self.log(
                f"SELL {order.data._name} size={abs(order.executed.size):.0f} "
                f"price={order.executed.price:.4f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER {order.getstatusname()} {order.data._name}")

    def _get_current_scores(self) -> Dict[str, float]:
        current_date = self.datetime.date(0)
        score_map = {}
        for data in self.datas:
            if not len(data):
                continue
            data_date = bt.num2date(data.datetime[0]).date()
            if data_date != current_date:
                continue
            score = float(data.signal_score[0])
            close = float(data.close[0])
            if math.isfinite(score) and math.isfinite(close) and close > 0:
                score_map[data._name] = score
        return score_map

    def _get_current_holdings(self) -> List[str]:
        holdings = []
        for data in self.datas:
            position = self.getposition(data)
            if position.size > 0:
                holdings.append(data._name)
                self.hold_days[data._name] = self.hold_days.get(data._name, 0) + 1
            else:
                self.hold_days.pop(data._name, None)
                self.missing_signal_days.pop(data._name, None)
        return holdings

    def next(self):
        current_date = self.datetime.date(0)
        if self.last_rebalance_date == current_date:
            return
        self.last_rebalance_date = current_date

        holdings = self._get_current_holdings()
        score_map = self._get_current_scores()
        for name in holdings:
            if name in score_map:
                self.missing_signal_days.pop(name, None)
            else:
                self.missing_signal_days[name] = self.missing_signal_days.get(name, 0) + 1

        stale_holdings = []
        if self.p.sell_missing_signal:
            stale_holdings = [
                name
                for name in holdings
                if self.missing_signal_days.get(name, 0) >= self.p.missing_signal_sell_after
            ]
        if not score_map:
            for name in stale_holdings:
                data = self.data_by_name.get(name)
                if data is not None:
                    self.log(
                        f"SELL {name} due to missing signal "
                        f"({self.missing_signal_days.get(name, 0)} days)"
                    )
                    self.order_target_percent(data=data, target=0.0)
            return

        current_rank = pd.Series({name: score_map.get(name, np.nan) for name in holdings}).dropna()
        last = pd.Index(current_rank.sort_values(ascending=False).index)
        ranked = pd.Series(score_map).sort_values(ascending=False)

        candidate_count = max(self.p.n_drop + self.p.topk - len(last), 0)
        today = pd.Index(ranked.index[~ranked.index.isin(last)][:candidate_count])

        combined_names = list(last) + [name for name in today if name not in last]
        combined_rank = ranked.reindex(combined_names).dropna().sort_values(ascending=False)
        bottom = pd.Index(combined_rank.index[-self.p.n_drop :]) if len(combined_rank) else pd.Index([])
        sell_candidates = last[last.isin(bottom)]

        allowed_sells = [
            name for name in sell_candidates if self.hold_days.get(name, 0) >= self.p.hold_thresh
        ]
        retained = [name for name in holdings if name not in allowed_sells and name not in stale_holdings]
        buy_slots = max(self.p.topk - len(retained), 0)
        buy_names = [name for name in today if name not in retained][:buy_slots]
        target_names = retained + [name for name in buy_names if name not in retained]

        target_weight = self.p.risk_degree / len(target_names) if target_names else 0.0
        target_map = {}
        for name in holdings:
            if name not in target_names:
                target_map[name] = 0.0
        for name in target_names:
            target_map[name] = target_weight

        if self.p.verbose:
            preview = ", ".join(target_names[: min(10, len(target_names))])
            self.log(
                f"rebalance holdings={len(holdings)} target={len(target_names)} "
                f"sells={allowed_sells} buys={buy_names} preview=[{preview}]"
            )

        for name, target in sorted(target_map.items(), key=lambda item: item[1]):
            data = self.data_by_name.get(name)
            if data is not None:
                self.order_target_percent(data=data, target=target)


def init_qlib():
    provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    try:
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)


def _iter_recorders(exp) -> Iterable[Tuple[str, object]]:
    recorders = exp.list_recorders(rtype=exp.RT_L)
    if isinstance(recorders, dict):
        items = list(recorders.items())
    else:
        items = [(getattr(rec, "id", rec), rec) for rec in recorders]
    return sorted(
        items,
        key=lambda item: getattr(item[1], "info", {}).get("start_time", ""),
        reverse=True,
    )


def resolve_recorder(experiment_name: str, recorder_id: Optional[str] = None):
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id is not None:
        return exp.get_recorder(recorder_id=recorder_id)

    for rid, rec in _iter_recorders(exp):
        info = getattr(rec, "info", {}) or {}
        if info.get("status") != "FINISHED":
            continue
        recorder = exp.get_recorder(recorder_id=rid)
        try:
            pred = recorder.load_object("pred.pkl")
        except Exception:
            continue
        if pred is not None and len(pred) > 0:
            print(f"Using latest finished recorder with predictions: {rid}")
            return recorder

    raise ValueError(
        f"No finished recorder with pred.pkl was found in experiment '{experiment_name}'."
    )


def normalize_prediction_frame(pred: pd.DataFrame) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "score"})

    index_names = list(pred.index.names)
    if set(index_names) != {"datetime", "instrument"}:
        raise ValueError(
            f"pred.pkl index must contain datetime/instrument, got index names: {index_names}"
        )
    if index_names != ["datetime", "instrument"]:
        pred = pred.reorder_levels(["datetime", "instrument"])

    pred = pred.sort_index()
    pred.index = pred.index.set_levels(
        [
            pd.to_datetime(pred.index.levels[0]),
            pred.index.levels[1],
        ]
    )
    return pred


def select_instruments(score_pivot: pd.DataFrame, topk: int, max_instruments: Optional[int]) -> List[str]:
    available = score_pivot.columns[score_pivot.notna().any()].tolist()
    if max_instruments is None or max_instruments >= len(available):
        return available

    cutoff = min(len(available), max(topk * 3, max_instruments))
    top_flags = score_pivot.rank(axis=1, ascending=False, method="first") <= cutoff
    top_hits = top_flags.sum().sort_values(ascending=False)
    selected = top_hits.head(max_instruments).index.tolist()
    return selected


def build_feed_frames(
    pred: pd.DataFrame,
    start_time: Optional[str],
    end_time: Optional[str],
    topk: int,
    max_instruments: Optional[int],
) -> Dict[str, pd.DataFrame]:
    pred = normalize_prediction_frame(pred)
    score_pivot = pred["score"].unstack("instrument").sort_index()

    start_ts = pd.Timestamp(start_time) if start_time is not None else score_pivot.index.min()
    end_ts = pd.Timestamp(end_time) if end_time is not None else score_pivot.index.max()
    shifted_scores = score_pivot.loc[:end_ts].shift(1)
    shifted_scores = shifted_scores.loc[(shifted_scores.index >= start_ts) & (shifted_scores.index <= end_ts)]

    instruments = select_instruments(shifted_scores, topk=topk, max_instruments=max_instruments)
    if not instruments:
        raise ValueError("No instruments were selected from pred.pkl.")

    price_data = D.features(
        instruments,
        PRICE_FIELDS,
        start_time=start_ts,
        end_time=end_ts,
        freq="day",
    )
    if price_data is None or price_data.empty:
        raise ValueError("Qlib did not return price data for the selected instruments.")

    price_data = price_data.sort_index().rename(columns={field: field[1:] for field in PRICE_FIELDS})
    feed_frames: Dict[str, pd.DataFrame] = {}
    available_instruments = set(price_data.index.get_level_values("instrument"))

    for instrument in instruments:
        if instrument not in available_instruments:
            continue
        inst_price = price_data.xs(instrument, level="instrument").copy()
        inst_signal = shifted_scores[instrument] if instrument in shifted_scores.columns else pd.Series(dtype=float)
        frame = inst_price.join(inst_signal.rename("signal_score"), how="left")
        frame["volume"] = frame["volume"].fillna(0.0)
        frame["openinterest"] = 0.0
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        if not frame.empty:
            feed_frames[instrument] = frame

    if not feed_frames:
        raise ValueError("No valid OHLCV frames were constructed for backtrader.")
    return feed_frames


def summarize_results(strategy, initial_cash: float) -> Dict[str, object]:
    returns = pd.Series(strategy.analyzers.time_return.get_analysis(), dtype=float).sort_index()
    if returns.empty:
        final_value = strategy.broker.getvalue()
        return {
            "initial_cash": initial_cash,
            "final_value": final_value,
            "total_return": final_value / initial_cash - 1.0,
            "annual_return": None,
            "max_drawdown": None,
            "daily_returns": returns,
        }

    equity_curve = (1.0 + returns).cumprod() * initial_cash
    drawdown_info = strategy.analyzers.drawdown.get_analysis()
    return {
        "initial_cash": initial_cash,
        "final_value": float(equity_curve.iloc[-1]),
        "total_return": float(equity_curve.iloc[-1] / initial_cash - 1.0),
        "annual_return": annualize_returns(returns),
        "max_drawdown": drawdown_info.get("max", {}).get("drawdown"),
        "daily_returns": returns,
        "equity_curve": equity_curve,
    }


def run_backtrader_bridge(
    recorder_id: Optional[str] = None,
    experiment_name: str = "workflow",
    topk: int = 20,
    n_drop: int = 2,
    hold_thresh: int = 1,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    account: Optional[float] = None,
    max_instruments: Optional[int] = None,
    sell_missing_signal: bool = False,
    missing_signal_sell_after: int = 3,
    verbose: bool = False,
    plot: bool = False,
    plot_output: Optional[str] = None,
):
    init_qlib()
    recorder = resolve_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
    pred = recorder.load_object("pred.pkl")

    config = dict(BACKTEST_CONFIG)
    benchmark = config.get("benchmark")
    exchange_kwargs = dict(config.get("exchange_kwargs", {}))
    start_time = start_time or config.get("start_time")
    end_time = end_time or config.get("end_time")
    account = float(account or config.get("account", 100000000))

    print("=" * 80)
    print("Building backtrader feeds from qlib artifacts")
    print("=" * 80)
    print(f"Recorder ID: {recorder.id}")
    print(f"Experiment: {experiment_name}")
    print(f"Date range: {start_time} -> {end_time}")
    print(f"Strategy: topk={topk}, n_drop={n_drop}, hold_thresh={hold_thresh}")
    print(f"Sell missing signal: {sell_missing_signal}")
    if sell_missing_signal:
        print(f"Missing signal sell after: {missing_signal_sell_after} days")
    if max_instruments is not None:
        print(f"Max instruments: {max_instruments}")

    feed_frames = build_feed_frames(
        pred=pred,
        start_time=start_time,
        end_time=end_time,
        topk=topk,
        max_instruments=max_instruments,
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(account)
    cerebro.broker.set_coc(True)
    cerebro.broker.addcommissioninfo(
        CNStockCommission(
            buy_commission=exchange_kwargs.get("open_cost", 0.0005),
            sell_commission=exchange_kwargs.get("close_cost", 0.0015),
            min_commission=exchange_kwargs.get("min_cost", 5),
        )
    )

    for instrument, frame in feed_frames.items():
        feed = QlibPredictionData(dataname=frame)
        cerebro.adddata(feed, name=instrument)

    cerebro.addstrategy(
        QlibTopKStrategy,
        topk=topk,
        n_drop=n_drop,
        hold_thresh=hold_thresh,
        risk_degree=0.95,
        sell_missing_signal=sell_missing_signal,
        missing_signal_sell_after=missing_signal_sell_after,
        verbose=verbose,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"Loaded {len(feed_frames)} instrument feeds into backtrader")
    results = cerebro.run(runonce=False, preload=True)
    strategy = results[0]
    summary = summarize_results(strategy, initial_cash=account)
    benchmark_name, benchmark_returns = load_benchmark_returns(benchmark, start_time, end_time)
    if not benchmark_returns.empty:
        summary["benchmark_name"] = benchmark_name
        summary["benchmark_daily_returns"] = benchmark_returns
        summary["benchmark_curve"] = (1.0 + benchmark_returns).cumprod() * account
        summary["benchmark_annual_return"] = annualize_returns(benchmark_returns)
    else:
        summary["benchmark_name"] = benchmark_name
        summary["benchmark_daily_returns"] = benchmark_returns
        summary["benchmark_annual_return"] = None

    print("\n" + "=" * 80)
    print("Backtrader run completed")
    print("=" * 80)
    print(f"Initial cash: {summary['initial_cash']:.2f}")
    print(f"Final value:  {summary['final_value']:.2f}")
    print(f"Total return: {summary['total_return']:.2%}")
    if summary.get("annual_return") is not None:
        print(f"Annual return: {summary['annual_return']:.2%}")
    if summary.get("max_drawdown") is not None:
        print(f"Max drawdown: {summary['max_drawdown']:.2f}%")
    if summary.get("benchmark_annual_return") is not None:
        print(f"Benchmark ({summary['benchmark_name']}) annual return: {summary['benchmark_annual_return']:.2%}")
        print(
            f"Excess annual return: "
            f"{summary['annual_return'] - summary['benchmark_annual_return']:.2%}"
        )
    if "equity_curve" in summary:
        print(f"Trading days: {len(summary['equity_curve'])}")
    if plot or plot_output:
        plot_summary(summary, output_path=plot_output)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run qlib prediction artifacts through backtrader."
    )
    parser.add_argument("--experiment-name", default="workflow")
    parser.add_argument("--recorder-id", default=None)
    parser.add_argument("--start-time", default=BACKTEST_CONFIG.get("start_time"))
    parser.add_argument("--end-time", default=BACKTEST_CONFIG.get("end_time"))
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n-drop", type=int, default=2)
    parser.add_argument("--hold-thresh", type=int, default=1)
    parser.add_argument("--account", type=float, default=BACKTEST_CONFIG.get("account", 100000000))
    parser.add_argument("--max-instruments", type=int, default=None)
    parser.add_argument(
        "--sell-missing-signal",
        action="store_true",
        help="Force-sell holdings that have missed signals for too many consecutive days.",
    )
    parser.add_argument(
        "--missing-signal-sell-after",
        type=int,
        default=3,
        help="Force-sell holdings only after this many consecutive missing-signal days.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Plot strategy and benchmark NAV curves.")
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Save plot to a file, e.g. result.png or result.html.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_backtrader_bridge(
        recorder_id=args.recorder_id,
        experiment_name=args.experiment_name,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        start_time=args.start_time,
        end_time=args.end_time,
        account=args.account,
        max_instruments=args.max_instruments,
        sell_missing_signal=args.sell_missing_signal,
        missing_signal_sell_after=max(args.missing_signal_sell_after, 1),
        verbose=args.verbose,
        plot=args.plot,
        plot_output=args.plot_output,
    )


if __name__ == "__main__":
    main()
