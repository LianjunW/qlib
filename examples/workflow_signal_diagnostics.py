#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostics for workflow prediction signals and saved backtest artifacts.

This script reads an existing qlib recorder. It does not train a model and does
not rerun a backtest. The goal is to make signal quality and performance metric
definitions reproducible:

- portfolio absolute return from ``report_normal_1day.pkl``
- excess return from ``port_analysis_1day.pkl``
- IC / RankIC from ``pred.pkl`` and ``label.pkl``
- TopK / Top1 trend matching diagnostics
- score quantile monotonicity diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _score_frame(pred: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        return pred.to_frame("score")
    if "score" not in pred.columns:
        return pred.rename(columns={pred.columns[0]: "score"})
    return pred[["score"]]


def _label_series(label: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(label, pd.DataFrame):
        return label.iloc[:, 0].rename("label")
    return label.rename("label")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _annualized_from_nav(nav: pd.Series, periods_per_year: int = 252) -> float:
    years = len(nav) / periods_per_year
    if years <= 0:
        raise ValueError("Cannot annualize an empty NAV series.")
    return float(nav.iloc[-1] ** (1 / years) - 1)


def _annualized_from_return(ret: pd.Series, periods_per_year: int = 252) -> float:
    clean = ret.dropna()
    if clean.empty:
        return float("nan")
    nav = (1 + clean).cumprod()
    return _annualized_from_nav(nav, periods_per_year=periods_per_year)


def _max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def _return_path_metrics(ret: pd.Series) -> Dict[str, Optional[float]]:
    clean = ret.dropna()
    if clean.empty:
        return {
            "trading_days": 0,
            "total_return": None,
            "annualized_return": None,
            "max_drawdown": None,
            "positive_day_rate": None,
        }
    nav = (1 + clean).cumprod()
    return {
        "trading_days": int(len(clean)),
        "total_return": float(nav.iloc[-1] - 1),
        "annualized_return": _annualized_from_nav(nav),
        "max_drawdown": _max_drawdown(nav),
        "positive_day_rate": float((clean > 0).mean()),
    }


def _risk_metric(port_analysis: pd.DataFrame, group: str, metric: str) -> Optional[float]:
    try:
        return _safe_float(port_analysis.loc[(group, metric), "risk"])
    except Exception:
        return None


def _portfolio_metrics(report: Optional[pd.DataFrame], start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
    if report is None:
        return {}
    sliced = report.loc[start:end].copy()
    if sliced.empty:
        return {"error": f"No report rows in requested window: {start} .. {end}"}
    if "account" not in sliced.columns:
        return {"error": "report_normal_1day.pkl has no account column"}

    nav = sliced["account"] / sliced["account"].iloc[0]
    metrics = {
        "start": str(sliced.index.min())[:10],
        "end": str(sliced.index.max())[:10],
        "trading_days": int(len(sliced)),
        "portfolio_total_return": float(nav.iloc[-1] - 1),
        "portfolio_annualized_return": _annualized_from_nav(nav),
        "portfolio_max_drawdown": _max_drawdown(nav),
    }
    if "bench" in sliced.columns:
        bench_nav = (1 + sliced["bench"].fillna(0)).cumprod()
        metrics["benchmark_total_return"] = float(bench_nav.iloc[-1] - 1)
        metrics["benchmark_annualized_return"] = _annualized_from_nav(bench_nav)
        metrics["benchmark_max_drawdown"] = _max_drawdown(bench_nav)
    return metrics


def _binary_auc(y_true: pd.Series, score: pd.Series) -> Optional[float]:
    """Compute AUC with average ranks; returns None if only one class exists."""
    y = y_true.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = score.rank(method="average")
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _ic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    by_day = df.groupby(level="datetime", sort=True)

    def corr(g: pd.DataFrame, method: str) -> float:
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            return np.nan
        return g["score"].corr(g["label"], method=method)

    daily_ic = by_day.apply(lambda g: corr(g, "pearson"))
    daily_rankic = by_day.apply(lambda g: corr(g, "spearman"))
    y_true = (df["label"] > 0).astype(int)
    score_positive = df["score"] > 0

    return {
        "rows": int(len(df)),
        "dates": int(df.index.get_level_values("datetime").nunique()),
        "instruments": int(df.index.get_level_values("instrument").nunique()),
        "prediction_start": str(df.index.get_level_values("datetime").min())[:10],
        "prediction_end": str(df.index.get_level_values("datetime").max())[:10],
        "label_positive_rate": float(y_true.mean()),
        "score_positive_rate": float(score_positive.mean()),
        "overall_ic": _safe_float(df["score"].corr(df["label"])),
        "overall_rankic": _safe_float(df["score"].corr(df["label"], method="spearman")),
        "daily_ic_mean": _safe_float(daily_ic.mean()),
        "daily_ic_std": _safe_float(daily_ic.std()),
        "daily_ic_ir": _safe_float(daily_ic.mean() / daily_ic.std()),
        "daily_ic_positive_day_rate": float((daily_ic > 0).mean()),
        "daily_rankic_mean": _safe_float(daily_rankic.mean()),
        "daily_rankic_std": _safe_float(daily_rankic.std()),
        "daily_rankic_ir": _safe_float(daily_rankic.mean() / daily_rankic.std()),
        "daily_rankic_positive_day_rate": float((daily_rankic > 0).mean()),
        "binary_auc_label_gt_0": _safe_float(_binary_auc(y_true, df["score"])),
        "accuracy_score_gt_0": float((score_positive.astype(int) == y_true).mean()),
    }


def _topk_metrics(df: pd.DataFrame, topk_values: Iterable[int]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    by_day = df.groupby(level="datetime", sort=True)

    for k in topk_values:
        rows: List[Dict[str, Any]] = []
        prev_top: Optional[set] = None
        for dt, g in by_day:
            if len(g) < k * 2:
                continue
            ranked = g.sort_values("score", ascending=False)
            top = ranked.head(k)
            bottom = ranked.tail(k)
            all_mean = float(g["label"].mean())
            top_mean = float(top["label"].mean())
            bottom_mean = float(bottom["label"].mean())
            top_set = set(top.index.get_level_values("instrument"))
            turnover = None
            if prev_top is not None:
                turnover = 1 - len(top_set & prev_top) / k
            prev_top = top_set
            rows.append(
                {
                    "datetime": dt,
                    "top_mean": top_mean,
                    "bottom_mean": bottom_mean,
                    "all_mean": all_mean,
                    "top_bottom_spread": top_mean - bottom_mean,
                    "top_all_spread": top_mean - all_mean,
                    "top_positive_rate": float((top["label"] > 0).mean()),
                    "bottom_positive_rate": float((bottom["label"] > 0).mean()),
                    "all_positive_rate": float((g["label"] > 0).mean()),
                    "top_mean_positive": top_mean > 0,
                    "top_beats_all": top_mean > all_mean,
                    "top_beats_bottom": top_mean > bottom_mean,
                    "turnover": turnover,
                }
            )
        m = pd.DataFrame(rows)
        if m.empty:
            continue
        path = _return_path_metrics(m.set_index("datetime")["top_mean"])
        result[str(k)] = {
            "days": int(len(m)),
            "top_mean_return": float(m["top_mean"].mean()),
            "bottom_mean_return": float(m["bottom_mean"].mean()),
            "all_mean_return": float(m["all_mean"].mean()),
            "top_bottom_spread": float(m["top_bottom_spread"].mean()),
            "top_all_spread": float(m["top_all_spread"].mean()),
            "top_positive_rate": float(m["top_positive_rate"].mean()),
            "bottom_positive_rate": float(m["bottom_positive_rate"].mean()),
            "all_positive_rate": float(m["all_positive_rate"].mean()),
            "top_mean_positive_day_rate": float(m["top_mean_positive"].mean()),
            "top_beats_all_day_rate": float(m["top_beats_all"].mean()),
            "top_beats_bottom_day_rate": float(m["top_beats_bottom"].mean()),
            "avg_daily_turnover": _safe_float(m["turnover"].dropna().mean()),
            "path_total_return": path["total_return"],
            "path_annualized_return": path["annualized_return"],
            "path_max_drawdown": path["max_drawdown"],
        }
    return result


def _top1_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    prev_inst: Optional[str] = None
    for dt, g in df.groupby(level="datetime", sort=True):
        if len(g) < 20:
            continue
        ranked = g.sort_values("score", ascending=False)
        top1 = ranked.iloc[0]
        top_inst = ranked.index.get_level_values("instrument")[0]
        all_mean = float(g["label"].mean())
        top20_mean = float(ranked.head(min(20, len(ranked)))["label"].mean())
        rank_pct = float(g["label"].rank(pct=True).loc[ranked.index[0]])
        score_gap_2 = float(ranked.iloc[0]["score"] - ranked.iloc[1]["score"]) if len(ranked) > 1 else None
        score_gap_20 = (
            float(ranked.iloc[0]["score"] - ranked.head(min(20, len(ranked)))["score"].mean())
            if len(ranked) >= 2
            else None
        )
        rows.append(
            {
                "datetime": dt,
                "instrument": top_inst,
                "label": float(top1["label"]),
                "all_mean": all_mean,
                "top20_mean": top20_mean,
                "positive": float(top1["label"]) > 0,
                "beats_all": float(top1["label"]) > all_mean,
                "beats_top20_mean": float(top1["label"]) > top20_mean,
                "in_actual_top_5pct": rank_pct >= 0.95,
                "in_actual_top_10pct": rank_pct >= 0.90,
                "changed_from_prev_day": prev_inst is not None and top_inst != prev_inst,
                "score_gap_2": score_gap_2,
                "score_gap_20": score_gap_20,
            }
        )
        prev_inst = top_inst
    m = pd.DataFrame(rows)
    if m.empty:
        return {}
    path = _return_path_metrics(m.set_index("datetime")["label"])
    gap_filters: Dict[str, Any] = {}
    for gap_col in ("score_gap_2", "score_gap_20"):
        valid_gap = m[gap_col].dropna()
        if valid_gap.empty:
            continue
        gap_filters[gap_col] = {}
        for quantile in (0.5, 0.7, 0.8, 0.9):
            threshold = float(valid_gap.quantile(quantile))
            filtered = m.loc[m[gap_col] >= threshold].copy()
            filtered_path = _return_path_metrics(filtered.set_index("datetime")["label"])
            gap_filters[gap_col][str(int(quantile * 100))] = {
                "threshold": threshold,
                "days": int(len(filtered)),
                "coverage": float(len(filtered) / len(m)),
                "mean_return": float(filtered["label"].mean()) if not filtered.empty else None,
                "median_return": float(filtered["label"].median()) if not filtered.empty else None,
                "positive_day_rate": float(filtered["positive"].mean()) if not filtered.empty else None,
                "beats_all_day_rate": float(filtered["beats_all"].mean()) if not filtered.empty else None,
                "actual_top_10pct_hit_rate": (
                    float(filtered["in_actual_top_10pct"].mean()) if not filtered.empty else None
                ),
                "path_total_return": filtered_path["total_return"],
                "path_annualized_return": filtered_path["annualized_return"],
                "path_max_drawdown": filtered_path["max_drawdown"],
            }
    return {
        "days": int(len(m)),
        "mean_return": float(m["label"].mean()),
        "median_return": float(m["label"].median()),
        "positive_day_rate": float(m["positive"].mean()),
        "beats_all_day_rate": float(m["beats_all"].mean()),
        "beats_top20_mean_day_rate": float(m["beats_top20_mean"].mean()),
        "actual_top_5pct_hit_rate": float(m["in_actual_top_5pct"].mean()),
        "actual_top_10pct_hit_rate": float(m["in_actual_top_10pct"].mean()),
        "avg_daily_turnover": float(m["changed_from_prev_day"].mean()),
        "avg_score_gap_2": _safe_float(m["score_gap_2"].mean()),
        "avg_score_gap_20": _safe_float(m["score_gap_20"].mean()),
        "path_total_return": path["total_return"],
        "path_annualized_return": path["annualized_return"],
        "path_max_drawdown": path["max_drawdown"],
        "score_gap_filters": gap_filters,
    }


def _quantile_metrics(df: pd.DataFrame, n_quantiles: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for dt, g in df.groupby(level="datetime", sort=True):
        if len(g) < n_quantiles * 5:
            continue
        ranked = g.copy()
        try:
            ranked["bucket"] = pd.qcut(
                ranked["score"].rank(method="first"),
                n_quantiles,
                labels=False,
            ) + 1
        except ValueError:
            continue
        for bucket, part in ranked.groupby("bucket"):
            rows.append(
                {
                    "bucket": int(bucket),
                    "label": float(part["label"].mean()),
                    "positive_rate": float((part["label"] > 0).mean()),
                }
            )
    q = pd.DataFrame(rows)
    if q.empty:
        return {}
    by_bucket = q.groupby("bucket").mean(numeric_only=True)
    return {
        str(int(idx)): {
            "mean_return": float(row["label"]),
            "positive_rate": float(row["positive_rate"]),
        }
        for idx, row in by_bucket.iterrows()
    }


def _yearly_metrics(df: pd.DataFrame, topk_values: Iterable[int]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    dates = pd.DatetimeIndex(df.index.get_level_values("datetime"))
    years = sorted(dates.year.unique())
    for year in years:
        mask = dates.year == year
        sub = df.loc[mask]
        if sub.empty:
            continue
        year_item: Dict[str, Any] = {"signal": _ic_metrics(sub), "topk": {}}
        topk = _topk_metrics(sub, topk_values)
        for k, item in topk.items():
            year_item["topk"][k] = {
                "top_mean_return": item["top_mean_return"],
                "top_bottom_spread": item["top_bottom_spread"],
                "top_beats_all_day_rate": item["top_beats_all_day_rate"],
                "top_beats_bottom_day_rate": item["top_beats_bottom_day_rate"],
                "path_annualized_return": item["path_annualized_return"],
                "path_max_drawdown": item["path_max_drawdown"],
            }
        year_item["top1"] = _top1_metrics(sub)
        result[str(year)] = year_item
    return result


def load_recorder_objects(experiment_name: str, recorder_id: str, qlib_data_dir: str):
    import qlib
    from qlib.constant import REG_CN
    from qlib.workflow import R

    qlib.init(provider_uri=os.path.expanduser(qlib_data_dir), region=REG_CN)
    recorder = R.get_exp(experiment_name=experiment_name).get_recorder(recorder_id=recorder_id)
    pred = _score_frame(recorder.load_object("pred.pkl"))
    label = _label_series(recorder.load_object("label.pkl"))

    report = None
    port_analysis = None
    try:
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    except Exception:
        pass
    try:
        port_analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    except Exception:
        pass
    return pred, label, report, port_analysis


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    pred, label, backtest_report, port_analysis = load_recorder_objects(
        args.experiment,
        args.recorder_id,
        args.qlib_data_dir,
    )
    df = pred.join(label, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if args.start or args.end:
        df = df.loc[args.start : args.end]
    if df.empty:
        raise ValueError("No joined pred/label rows in requested date range.")

    result: Dict[str, Any] = {
        "recorder_id": args.recorder_id,
        "experiment": args.experiment,
        "date_filter": {"start": args.start, "end": args.end},
        "signal": _ic_metrics(df),
        "topk": _topk_metrics(df, args.topk),
        "top1": _top1_metrics(df),
        "quantiles": _quantile_metrics(df, args.quantiles),
        "yearly": _yearly_metrics(df, args.topk),
    }
    if backtest_report is not None:
        result["portfolio"] = _portfolio_metrics(backtest_report, args.start, args.end)
    if port_analysis is not None:
        result["excess"] = {
            "with_cost_annualized_return": _risk_metric(
                port_analysis, "excess_return_with_cost", "annualized_return"
            ),
            "with_cost_max_drawdown": _risk_metric(port_analysis, "excess_return_with_cost", "max_drawdown"),
            "without_cost_annualized_return": _risk_metric(
                port_analysis, "excess_return_without_cost", "annualized_return"
            ),
            "without_cost_max_drawdown": _risk_metric(
                port_analysis, "excess_return_without_cost", "max_drawdown"
            ),
        }
    return result


def print_text_report(report: Dict[str, Any]) -> None:
    signal = report["signal"]
    print(f"Recorder: {report['experiment']}/{report['recorder_id']}")
    print(
        f"Signal coverage: {signal['prediction_start']} .. {signal['prediction_end']}, "
        f"rows={signal['rows']}, instruments={signal['instruments']}"
    )
    print("\nSignal quality:")
    print(f"  daily IC mean / IR: {signal['daily_ic_mean']:.6f} / {signal['daily_ic_ir']:.4f}")
    print(f"  daily RankIC mean / IR: {signal['daily_rankic_mean']:.6f} / {signal['daily_rankic_ir']:.4f}")
    print(f"  RankIC positive day rate: {signal['daily_rankic_positive_day_rate']:.2%}")
    print(f"  label positive rate: {signal['label_positive_rate']:.2%}")
    print(f"  binary AUC(label>0): {signal['binary_auc_label_gt_0']:.4f}")

    if "portfolio" in report:
        p = report["portfolio"]
        if "error" not in p:
            print("\nPortfolio artifact metrics:")
            print(f"  window: {p['start']} .. {p['end']} ({p['trading_days']} days)")
            print(f"  portfolio total / annual: {p['portfolio_total_return']:.2%} / {p['portfolio_annualized_return']:.2%}")
            print(f"  portfolio max drawdown: {p['portfolio_max_drawdown']:.2%}")
            if "benchmark_annualized_return" in p:
                print(f"  benchmark annual: {p['benchmark_annualized_return']:.2%}")
    if "excess" in report:
        e = report["excess"]
        print("\nExcess artifact metrics:")
        print(f"  with cost annual / max drawdown: {e['with_cost_annualized_return']:.2%} / {e['with_cost_max_drawdown']:.2%}")
        print(f"  without cost annual: {e['without_cost_annualized_return']:.2%}")

    print("\nTopK diagnostics:")
    for k, item in report["topk"].items():
        print(
            f"  Top{k}: mean={item['top_mean_return']:.4%}, "
            f"top-bottom={item['top_bottom_spread']:.4%}, "
            f"beats_all={item['top_beats_all_day_rate']:.2%}, "
            f"beats_bottom={item['top_beats_bottom_day_rate']:.2%}, "
            f"turnover={item['avg_daily_turnover']:.2%}, "
            f"path_ann={item['path_annualized_return']:.2%}, "
            f"path_mdd={item['path_max_drawdown']:.2%}"
        )

    t = report["top1"]
    if t:
        print("\nTop1 diagnostics:")
        print(f"  mean / median return: {t['mean_return']:.4%} / {t['median_return']:.4%}")
        print(f"  positive day rate: {t['positive_day_rate']:.2%}")
        print(f"  beats all / Top20 mean: {t['beats_all_day_rate']:.2%} / {t['beats_top20_mean_day_rate']:.2%}")
        print(f"  actual top 5% / 10% hit rate: {t['actual_top_5pct_hit_rate']:.2%} / {t['actual_top_10pct_hit_rate']:.2%}")
        print(f"  daily turnover: {t['avg_daily_turnover']:.2%}")
        print(f"  path annual / max drawdown: {t['path_annualized_return']:.2%} / {t['path_max_drawdown']:.2%}")
        if t.get("score_gap_filters"):
            print("  score gap filters:")
            for gap_name, buckets in t["score_gap_filters"].items():
                for quantile, item in buckets.items():
                    print(
                        f"    {gap_name} p{quantile}: coverage={item['coverage']:.2%}, "
                        f"mean={item['mean_return']:.4%}, positive={item['positive_day_rate']:.2%}, "
                        f"path_ann={item['path_annualized_return']:.2%}, "
                        f"path_mdd={item['path_max_drawdown']:.2%}"
                    )

    if report["quantiles"]:
        print("\nScore quantiles (low -> high):")
        for bucket, item in sorted(report["quantiles"].items(), key=lambda kv: int(kv[0])):
            print(f"  Q{bucket}: mean={item['mean_return']:.4%}, positive={item['positive_rate']:.2%}")

    if report["yearly"]:
        print("\nYearly diagnostics:")
        for year, item in sorted(report["yearly"].items()):
            sig = item["signal"]
            print(
                f"  {year}: RankIC={sig['daily_rankic_mean']:.6f}, "
                f"RankIC+days={sig['daily_rankic_positive_day_rate']:.2%}, "
                f"AUC={sig['binary_auc_label_gt_0']:.4f}"
            )
            for k in ("1", "5", "20"):
                if k in item["topk"]:
                    top = item["topk"][k]
                    print(
                        f"    Top{k}: mean={top['top_mean_return']:.4%}, "
                        f"spread={top['top_bottom_spread']:.4%}, "
                        f"path_ann={top['path_annualized_return']:.2%}, "
                        f"path_mdd={top['path_max_drawdown']:.2%}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a qlib recorder and report signal/backtest diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--recorder-id", required=True)
    parser.add_argument("--experiment", default="workflow")
    parser.add_argument("--qlib-data-dir", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20, 30, 50])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
