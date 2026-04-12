#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib Workflow: Train / Predict / Backtest / Analyze

Separates model training from strategy backtesting so that changing strategy
parameters or the test window never triggers an expensive re-training step.

Prediction generation uses the union of index constituents across the backtest
window (see ``resolve_prediction_instruments``), which avoids signal gaps for
stocks that leave the index mid-period.

Subcommands (CLI)::

    python workflow_by_code_v2.py train [gbdt|mlp|mlp_deep]
    python workflow_by_code_v2.py backtest  --topk 30 --n-drop 3
    python workflow_by_code_v2.py bridge    --plot-output nav.html
    python workflow_by_code_v2.py predict   --start 2024-01-01 --end 2026-04-10
    python workflow_by_code_v2.py layer     --n-groups 5
    python workflow_by_code_v2.py analyze   --top-n 10
    python workflow_by_code_v2.py list      --experiment workflow

Python API::

    from workflow_by_code_v2 import train_model, run_backtest_only

    rid = train_model(model_type="gbdt")
    run_backtest_only(recorder_id=rid, topk=30, n_drop=3)
"""
import argparse
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_MARKET, GBDT_MODEL
from qlib.contrib.report.analysis_model.analysis_model_performance import (
    model_performance_graph,
)


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS: Dict[str, dict] = {
    "gbdt": GBDT_MODEL,
    "mlp": {
        "class": "DNNModelPytorch",
        "module_path": "qlib.contrib.model.pytorch_nn",
        "kwargs": {
            "lr": 0.0001,
            "max_steps": 8000,
            "batch_size": 2048,
            "early_stop_rounds": 100,
            "eval_steps": 20,
            "optimizer": "adam",
            "loss": "mse",
            "GPU": -1,
            "seed": 42,
            "weight_decay": 0.001,
            "pt_model_kwargs": {
                "input_dim": 158,
                "layers": (256, 128),
                "act": "LeakyReLU",
            },
        },
    },
    "mlp_deep": {
        "class": "DNNModelPytorch",
        "module_path": "qlib.contrib.model.pytorch_nn",
        "kwargs": {
            "lr": 0.00005,
            "max_steps": 10000,
            "batch_size": 2048,
            "early_stop_rounds": 150,
            "eval_steps": 20,
            "optimizer": "adam",
            "loss": "mse",
            "GPU": -1,
            "seed": 42,
            "weight_decay": 0.001,
            "pt_model_kwargs": {
                "input_dim": 158,
                "layers": (512, 256, 128),
                "act": "LeakyReLU",
            },
        },
    },
}


# ============================================================
# Strategy / Backtest Defaults (modify without re-training)
# ============================================================

STRATEGY_CONFIG = {
    "topk": 20,
    "n_drop": 2,
    "hold_thresh": 1,
}


def _get_latest_calendar_date(
    calendar_path: str = "~/.qlib/qlib_data/cn_data/calendars/day.txt",
) -> str:
    calendar_file = Path(calendar_path).expanduser()
    if not calendar_file.exists():
        return "2025-08-01"
    with calendar_file.open("r", encoding="utf-8") as f:
        trading_days = [line.strip() for line in f if line.strip()]
    return trading_days[-1] if trading_days else "2025-08-01"


LATEST_BACKTEST_END = _get_latest_calendar_date()

BACKTEST_CONFIG = {
    "start_time": "2024-01-01",
    "end_time": LATEST_BACKTEST_END,
    "account": 1_000_000,
    "benchmark": CSI300_BENCH,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0001,
        "close_cost": 0.0001,
        "min_cost": 1,
    },
}


# ============================================================
# Internal Helpers
# ============================================================


def ensure_qlib_initialized() -> None:
    """Initialize qlib once; subsequent calls are no-ops."""
    try:
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        provider_uri = "~/.qlib/qlib_data/cn_data"
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)


def _sorted_recorder_pairs(experiment_name: str) -> List[tuple]:
    """Return ``[(recorder_id, start_time), ...]`` sorted newest-first."""
    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders(rtype=exp.RT_L)
    if not recorders:
        raise ValueError(
            f"实验 '{experiment_name}' 中没有 recorder，请先运行 train_model()"
        )

    if isinstance(recorders, dict):
        items = list(recorders.items())
    else:
        items = [(getattr(r, "id", r), r) for r in recorders]

    pairs = [
        (rid, rec.info.get("start_time", 0) if hasattr(rec, "info") else 0)
        for rid, rec in items
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _load_recorder(
    experiment_name: str,
    recorder_id: Optional[str] = None,
    require_pred: bool = False,
):
    """Return ``(recorder, recorder_id)``.

    When *recorder_id* is ``None``, iterate recorders newest-first.
    If *require_pred* is ``True``, skip any recorder that does not contain a
    non-empty ``pred.pkl`` — this avoids selecting a failed or in-progress
    training run.
    """
    ensure_qlib_initialized()
    exp = R.get_exp(experiment_name=experiment_name)

    if recorder_id is not None:
        return exp.get_recorder(recorder_id=recorder_id), recorder_id

    for rid, _ in _sorted_recorder_pairs(experiment_name):
        recorder = exp.get_recorder(recorder_id=rid)
        if not require_pred:
            print(f"自动选择最新 Recorder: {rid}")
            return recorder, rid
        try:
            pred = recorder.load_object("pred.pkl")
            if pred is not None and len(pred) > 0:
                print(f"自动选择最新有效 Recorder: {rid}")
                return recorder, rid
        except Exception:
            continue

    hint = " (含 pred.pkl)" if require_pred else ""
    raise ValueError(f"实验 '{experiment_name}' 中没有找到有效 recorder{hint}")


def _list_recorders(experiment_name: str) -> None:
    """Print every recorder in an experiment with pred.pkl status."""
    ensure_qlib_initialized()
    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders(rtype=exp.RT_L)
    if not recorders:
        print("  (无)")
        return

    if isinstance(recorders, dict):
        items = list(recorders.items())
    else:
        items = [(getattr(r, "id", r), r) for r in recorders]

    for i, (rid, rec) in enumerate(items):
        ts = "N/A"
        if hasattr(rec, "info"):
            try:
                ts = rec.info.get("start_time", "N/A")
            except Exception:
                pass
        has_pred = False
        try:
            recorder = exp.get_recorder(recorder_id=rid)
            pred = recorder.load_object("pred.pkl")
            has_pred = pred is not None and len(pred) > 0
        except Exception:
            pass
        pred_tag = "pred:yes" if has_pred else "pred:no"
        print(f"  [{i}] {rid}  ({ts})  [{pred_tag}]")


# ============================================================
# Dataset Configuration
# ============================================================


def get_extended_dataset_config(
    train_periods: Optional[List[tuple]] = None,
    valid: tuple = ("2022-01-01", "2023-12-31"),
    test: Optional[tuple] = None,
    instruments=CSI300_MARKET,
    end_time: Optional[str] = None,
) -> dict:
    """Build an Alpha158 DatasetH config with multi-segment training support."""
    if train_periods is None:
        train_periods = [
            ("2010-01-01", "2014-12-31"),
            ("2016-01-01", "2019-12-31"),
            ("2020-06-01", "2021-12-31"),
        ]
    if test is None:
        test = (BACKTEST_CONFIG["start_time"], BACKTEST_CONFIG["end_time"])
    if end_time is None:
        end_time = test[1]

    train_start = min(p[0] for p in train_periods)
    train_end = max(p[1] for p in train_periods)

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": train_start,
                    "end_time": end_time,
                    "fit_start_time": train_start,
                    "fit_end_time": train_end,
                    "instruments": instruments,
                    "infer_processors": [
                        {"class": "ProcessInf", "kwargs": {}},
                        {
                            "class": "ZScoreNorm",
                            "kwargs": {
                                "fit_start_time": train_start,
                                "fit_end_time": train_end,
                            },
                        },
                        {"class": "Fillna", "kwargs": {}},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                    ],
                },
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": valid,
                "test": test,
            },
        },
    }


# ============================================================
# Universe Helpers
# ============================================================


def get_union_instruments_for_period(
    market: str,
    start_time: str,
    end_time: str,
) -> List[str]:
    """
    Return the sorted union of index constituents across the given period.

    This keeps prediction coverage continuous for stocks that leave the index
    mid-period, preventing artificial signal gaps in downstream backtests.
    """
    ensure_qlib_initialized()
    instruments = D.instruments(market)
    return sorted(
        D.list_instruments(
            instruments,
            start_time=start_time,
            end_time=end_time,
            as_list=True,
            freq="day",
        )
    )


def resolve_prediction_instruments(
    instruments: Union[str, List[str]],
    test_start_time: str,
    test_end_time: str,
    use_union_constituents: bool = True,
) -> Union[str, List[str]]:
    """Expand a dynamic index universe into its period-union for prediction."""
    if not use_union_constituents or not isinstance(instruments, str):
        return instruments
    union = get_union_instruments_for_period(
        market=instruments,
        start_time=test_start_time,
        end_time=test_end_time,
    )
    print(
        f"使用 {instruments} 在 {test_start_time} ~ {test_end_time} 的成分股并集，"
        f"共 {len(union)} 只股票生成预测。"
    )
    return union


# ============================================================
# Core API
# ============================================================


def train_model(
    model_type: str = "gbdt",
    experiment_name: str = "workflow",
    use_union_constituents: bool = True,
) -> str:
    """
    Train a model and save predictions (``pred.pkl``).

    Returns the *recorder_id* for subsequent backtest / analysis.
    """
    ensure_qlib_initialized()

    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"未知的模型类型: {model_type}  (可选: {list(MODEL_CONFIGS)})"
        )
    model_config = MODEL_CONFIGS[model_type]
    print(f"使用模型: {model_type}")

    test_period = (BACKTEST_CONFIG["start_time"], BACKTEST_CONFIG["end_time"])
    prediction_instruments = resolve_prediction_instruments(
        instruments=CSI300_MARKET,
        test_start_time=test_period[0],
        test_end_time=test_period[1],
        use_union_constituents=use_union_constituents,
    )
    dataset_config = get_extended_dataset_config(
        test=test_period,
        instruments=prediction_instruments,
        end_time=test_period[1],
    )

    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)

    segments = dataset_config["kwargs"]["segments"]
    print(f"\n训练集: {segments['train'][0]} ~ {segments['train'][1]}")
    print(f"验证集: {segments['valid'][0]} ~ {segments['valid'][1]}")
    print(f"测试集: {segments['test'][0]} ~ {segments['test'][1]}")
    print(f"训练样本数: {len(dataset.prepare('train'))}")

    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))

        print("\n开始训练...")
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        recorder_id = recorder.id

        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()

        print(f"\n训练完成  Recorder ID: {recorder_id}")
        print(f"运行回测: run_backtest_only('{recorder_id}')")
        return recorder_id


def generate_predictions_for_new_period(
    recorder_id: Optional[str] = None,
    experiment_name: str = "workflow",
    test_start_time: Optional[str] = None,
    test_end_time: Optional[str] = None,
    instruments=CSI300_MARKET,
    new_experiment_name: Optional[str] = None,
    use_union_constituents: bool = True,
) -> str:
    """
    Re-use a trained model to generate predictions for a new time period.

    Returns the new *recorder_id*.
    """
    recorder, recorder_id = _load_recorder(experiment_name, recorder_id)
    model = recorder.load_object("params.pkl")
    print(f"已加载模型  (来源 recorder: {recorder_id})")

    start = test_start_time or BACKTEST_CONFIG["start_time"]
    end = test_end_time or BACKTEST_CONFIG["end_time"]

    prediction_instruments = resolve_prediction_instruments(
        instruments=instruments,
        test_start_time=start,
        test_end_time=end,
        use_union_constituents=use_union_constituents,
    )
    dataset_config = get_extended_dataset_config(
        test=(start, end),
        instruments=prediction_instruments,
        end_time=end,
    )
    dataset = init_instance_by_config(dataset_config)

    if isinstance(prediction_instruments, list):
        print(f"  股票池: {instruments} 成分股并集 ({len(prediction_instruments)} 只)")
    else:
        print(f"  股票池: {prediction_instruments}")
    print(f"  测试时间段: {start} ~ {end}")

    target_experiment = new_experiment_name or f"{experiment_name}_new_period"

    with R.start(experiment_name=target_experiment):
        R.log_params(
            **flatten_dict(
                {
                    "source_recorder_id": recorder_id,
                    "source_experiment": experiment_name,
                    "test_period": (start, end),
                }
            )
        )

        new_recorder = R.get_recorder()
        new_recorder_id = new_recorder.id
        new_recorder.save_objects(**{"params.pkl": model, "dataset": dataset})

        SignalRecord(model, dataset, new_recorder).generate()

        print(f"\n预测生成完成  新 Recorder ID: {new_recorder_id}")
        print(f"  实验: {target_experiment}")
        return new_recorder_id


def run_backtest_only(
    recorder_id: Optional[str] = None,
    experiment_name: str = "workflow",
    topk: int = STRATEGY_CONFIG["topk"],
    n_drop: int = STRATEGY_CONFIG["n_drop"],
    hold_thresh: int = STRATEGY_CONFIG["hold_thresh"],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Run a backtest using saved predictions (no re-training needed).

    Returns the ``port_analysis_1day.pkl`` DataFrame if available, or ``None``.
    """
    recorder, recorder_id = _load_recorder(
        experiment_name, recorder_id, require_pred=True,
    )

    bt_start = start_time or BACKTEST_CONFIG["start_time"]
    bt_end = end_time or BACKTEST_CONFIG["end_time"]

    pred = recorder.load_object("pred.pkl")
    dt_range = pred.index.get_level_values("datetime")

    print(f"\n回测配置:")
    print(f"  Recorder:  {recorder_id}")
    print(f"  时间段:    {bt_start} ~ {bt_end}")
    print(f"  topk={topk}  n_drop={n_drop}  hold_thresh={hold_thresh}")
    print(f"  预测数据:  {pred.shape[0]} 条  ({dt_range.min()} ~ {dt_range.max()})")

    backtest_config = {**BACKTEST_CONFIG, "start_time": bt_start, "end_time": bt_end}

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": topk,
                "n_drop": n_drop,
                "hold_thresh": hold_thresh,
            },
        },
        "backtest": backtest_config,
    }

    print("\n运行回测...")
    PortAnaRecord(recorder, port_analysis_config, "day").generate()
    print("回测完成！")

    try:
        return recorder.load_object("port_analysis_1day.pkl")
    except Exception:
        return None


def run_layer_analysis(
    recorder_id: Optional[str] = None,
    experiment_name: str = "workflow",
    output_dir: Optional[str] = None,
    n_groups: int = 5,
    graph_names: Optional[List[str]] = None,
) -> list:
    """
    Layer-group return analysis based on pred + label.

    Splits prediction scores into *n_groups* buckets and plots cumulative
    returns per group, long-short, IC, autocorrelation, etc.
    """
    recorder, recorder_id = _load_recorder(
        experiment_name, recorder_id, require_pred=True,
    )
    print(f"\n分层图分析  Recorder: {recorder_id}  实验: {experiment_name}")

    pred = recorder.load_object("pred.pkl")
    label = recorder.load_object("label.pkl")
    if pred is None or pred.empty:
        raise ValueError("无法加载 pred.pkl")
    if label is None or label.empty:
        raise ValueError("无法加载 label.pkl，分层分析需要 label 数据")

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "score"})

    label_series = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label

    pred_label = pred.copy()
    pred_label["label"] = label_series.reindex(pred.index)
    pred_label = pred_label.dropna(subset=["label", "score"])

    if pred_label.empty:
        raise ValueError("pred 与 label 合并后无有效数据，请检查时间范围")

    dt_range = pred_label.index.get_level_values("datetime")
    print(f"  有效数据: {len(pred_label)} 条  ({dt_range.min()} ~ {dt_range.max()})")

    graph_names = graph_names or ["group_return", "pred_ic", "pred_autocorr"]
    figures = model_performance_graph(
        pred_label=pred_label,
        N=n_groups,
        graph_names=graph_names,
        show_notebook=False,
    )

    out_dir = output_dir or os.path.join(os.getcwd(), "layer_analysis")
    os.makedirs(out_dir, exist_ok=True)
    for i, fig in enumerate(figures):
        path = os.path.join(out_dir, f"layer_analysis_{i + 1}.html")
        fig.write_html(path)
        print(f"  已保存: {path}")

    print("分层图分析完成！")
    return figures


# ============================================================
# GBDT Feature Importance
# ============================================================

ALPHA158_DESCRIPTIONS: Dict[str, str] = {
    "KMID": "K-bar mid-price ratio: (close-open)/open",
    "KLEN": "K-bar length: (high-low)/open",
    "KMID2": "K-bar mid-price ratio v2: (close-open)/(high-low)",
    "KUP": "K-bar upper shadow: (high-max(open,close))/(high-low)",
    "KUP2": "K-bar upper shadow v2: (high-max(open,close))/open",
    "KLOW": "K-bar lower shadow: (min(open,close)-low)/(high-low)",
    "KLOW2": "K-bar lower shadow v2: (min(open,close)-low)/open",
    "KSFT": "K-bar shift: (2*close-high-low)/open",
    "KSFT2": "K-bar shift v2: (2*close-high-low)/(high-low)",
    "OPEN": "Open/Close price ratio",
    "HIGH": "High/Close price ratio",
    "LOW": "Low/Close price ratio",
    "VWAP": "VWAP/Close price ratio",
    "ROC": "Rate of change: close/delay(close,d)-1",
    "MA": "Moving average ratio: mean(close,d)/close",
    "STD": "Rolling std: std(close,d)/close",
    "BETA": "Rolling beta against market",
    "RSQR": "Rolling R-squared",
    "RESI": "Rolling residual",
    "MAX": "Rolling max drawup: max(high,d)/close",
    "MIN": "Rolling min drawdown: min(low,d)/close",
    "QTLU": "Quantile upper: quantile(close,d,0.8)/close",
    "QTLD": "Quantile lower: quantile(close,d,0.2)/close",
    "RANK": "Rolling rank: rank(close) in past d days",
    "RSV": "Relative strength value: (close-min)/(max-min)",
    "IMAX": "Index of max: argmax(high,d)/d",
    "IMIN": "Index of min: argmin(low,d)/d",
    "IMXD": "Max-min index diff: (argmax-argmin)/d",
    "CORR": "Rolling correlation with volume",
    "CORD": "Rolling correlation delta",
    "CNTP": "Count positive returns in d days",
    "CNTN": "Count negative returns in d days",
    "CNTD": "Count diff: (positive - negative) / d",
    "SUMP": "Sum of positive returns in d days",
    "SUMN": "Sum of negative returns in d days",
    "SUMD": "Sum diff: (positive - negative) returns",
    "VMA": "Volume MA ratio: mean(volume,d)/(volume+1)",
    "VSTD": "Volume rolling std: std(volume,d)/(volume+1)",
    "WVMA": "Weighted volume MA: std(abs(return)*volume,d)/(mean(abs(return)*volume,d)+1)",
    "VSUMP": "Volume sum positive: sum of volume on up days",
    "VSUMN": "Volume sum negative: sum of volume on down days",
    "VSUMD": "Volume sum diff: (VSUMP - VSUMN) / sum(volume,d)",
}


def _describe_feature(feature_name) -> str:
    """Return a human-readable description for an Alpha158 feature column name."""
    col = str(feature_name)
    if isinstance(feature_name, tuple):
        col = feature_name[-1] if len(feature_name) > 0 else str(feature_name)

    for key, desc in ALPHA158_DESCRIPTIONS.items():
        if key in col:
            window = ""
            for part in str(col).split("_"):
                if part.isdigit():
                    window = f" (window={part}d)"
                    break
            return f"{desc}{window}"
    return col


def analyze_feature_importance(
    recorder_id: Optional[str] = None,
    experiment_name: str = "workflow",
    top_n: int = 5,
    output_path: Optional[str] = None,
) -> pd.Series:
    """
    Analyze GBDT feature importance and print the top-N most influential factors.

    Returns a ``pd.Series`` of all features sorted by importance (descending).
    """
    recorder, recorder_id = _load_recorder(experiment_name, recorder_id)
    model = recorder.load_object("params.pkl")

    print(f"\nGBDT Feature Importance  Recorder: {recorder_id}  实验: {experiment_name}")

    raw_fi = model.get_feature_importance()

    dataset_config = get_extended_dataset_config(
        test=(BACKTEST_CONFIG["start_time"], BACKTEST_CONFIG["end_time"]),
        instruments=CSI300_MARKET,
        end_time=BACKTEST_CONFIG["end_time"],
    )
    dataset = init_instance_by_config(dataset_config)
    df = dataset.prepare(
        segments=slice(None), col_set="feature", data_key=DataHandlerLP.DK_R
    )
    feature_cols = df.columns

    fi_named = {}
    for key, imp in raw_fi.to_dict().items():
        key_str = str(key)
        if "Column_" in key_str or "column_" in key_str:
            idx = int(key_str.split("_")[1])
            fi_named[feature_cols[idx] if idx < len(feature_cols) else key] = imp
        else:
            fi_named[key] = imp

    fi_series = pd.Series(fi_named).sort_values(ascending=False)

    print(f"\nTop {top_n} most important factors:\n")
    print(f"{'Rank':<6}{'Feature':<40}{'Importance':<15}{'Description'}")
    print("-" * 100)
    for rank, (feat, importance) in enumerate(fi_series.head(top_n).items(), 1):
        desc = _describe_feature(feat)
        feat_str = (
            "/".join(str(x) for x in feat) if isinstance(feat, tuple) else str(feat)
        )
        print(f"{rank:<6}{feat_str:<40}{importance:<15.1f}{desc}")

    total_importance = fi_series.sum()
    if total_importance > 0:
        top_share = fi_series.head(top_n).sum() / total_importance * 100
        print(f"\nTop-{top_n} share of total importance: {top_share:.1f}%")
    print(f"Total features: {len(fi_series)}")

    if output_path:
        try:
            import plotly.graph_objects as go

            top_fi = fi_series.head(min(20, len(fi_series)))
            labels = [
                "/".join(str(x) for x in f) if isinstance(f, tuple) else str(f)
                for f in top_fi.index
            ]

            fig = go.Figure(
                go.Bar(x=list(top_fi.values)[::-1], y=labels[::-1], orientation="h")
            )
            fig.update_layout(
                title=f"Top-{min(20, len(fi_series))} Feature Importance (GBDT)",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_white",
                height=max(400, min(20, len(fi_series)) * 28),
            )
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.write_html(output_path)
            print(f"\nSaved chart → {output_path}")
        except ImportError:
            print("\nplotly not installed; skipping chart output.")

    return fi_series


# ============================================================
# CLI
# ============================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qlib Workflow: 模型训练 / 预测 / 回测 / 分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python workflow_by_code_v2.py train gbdt
              python workflow_by_code_v2.py backtest --topk 30 --n-drop 3
              python workflow_by_code_v2.py bridge --plot-output nav.html
              python workflow_by_code_v2.py predict --start 2024-01-01 --end 2026-04-10
              python workflow_by_code_v2.py layer --experiment workflow
              python workflow_by_code_v2.py analyze --top-n 10
              python workflow_by_code_v2.py list --experiment workflow

            Python API:
              from workflow_by_code_v2 import train_model, run_backtest_only
              rid = train_model("gbdt")
              run_backtest_only(recorder_id=rid, topk=30)
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    p = sub.add_parser("train", help="训练模型并保存预测")
    p.add_argument(
        "model_type",
        nargs="?",
        default="gbdt",
        choices=list(MODEL_CONFIGS),
        help="模型类型 (default: gbdt)",
    )
    p.add_argument("--experiment", default="workflow")

    # --- backtest ---
    p = sub.add_parser("backtest", help="使用已有预测运行回测")
    p.add_argument("--recorder-id", default=None, help="Recorder ID (默认使用最新)")
    p.add_argument("--experiment", default="workflow")
    p.add_argument("--topk", type=int, default=STRATEGY_CONFIG["topk"])
    p.add_argument("--n-drop", type=int, default=STRATEGY_CONFIG["n_drop"])
    p.add_argument("--hold-thresh", type=int, default=STRATEGY_CONFIG["hold_thresh"])
    p.add_argument("--start", default=None, help="回测开始日期 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")

    # --- predict ---
    p = sub.add_parser("predict", help="用已训练模型为新时间段生成预测")
    p.add_argument("--recorder-id", default=None)
    p.add_argument("--experiment", default="workflow")
    p.add_argument("--start", required=True, help="新测试开始日期")
    p.add_argument("--end", required=True, help="新测试结束日期")
    p.add_argument("--new-experiment", default=None)

    # --- layer ---
    p = sub.add_parser("layer", help="分层图分析 (pred + label)")
    p.add_argument("--recorder-id", default=None)
    p.add_argument("--experiment", default="workflow")
    p.add_argument("--n-groups", type=int, default=5)
    p.add_argument("--output-dir", default=None)

    # --- analyze ---
    p = sub.add_parser("analyze", help="GBDT 因子重要性分析")
    p.add_argument("--recorder-id", default=None)
    p.add_argument("--experiment", default="workflow")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--output", default=None, help="HTML chart output path")

    # --- bridge ---
    p = sub.add_parser("bridge", help="使用 backtrader 引擎运行回测 (需要 backtrader)")
    p.add_argument("--recorder-id", default=None, help="Recorder ID (默认使用最新)")
    p.add_argument("--experiment", default="workflow")
    p.add_argument("--topk", type=int, default=STRATEGY_CONFIG["topk"])
    p.add_argument("--n-drop", type=int, default=STRATEGY_CONFIG["n_drop"])
    p.add_argument("--hold-thresh", type=int, default=STRATEGY_CONFIG["hold_thresh"])
    p.add_argument("--start", default=None, help="回测开始日期 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="回测结束日期 YYYY-MM-DD")
    p.add_argument(
        "--plot-output", default=None,
        help="保存 NAV 曲线图 (.html 或 .png)",
    )
    p.add_argument("--verbose", action="store_true")

    # --- list ---
    p = sub.add_parser("list", help="列出实验中所有 recorder")
    p.add_argument("--experiment", default="workflow")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n提示: 使用 -h 查看各子命令的详细帮助，例如:")
        print("  python workflow_by_code_v2.py backtest -h")
        return

    if args.command == "train":
        rid = train_model(
            model_type=args.model_type,
            experiment_name=args.experiment,
        )
        run_backtest_only(recorder_id=rid, experiment_name=args.experiment)

    elif args.command == "backtest":
        run_backtest_only(
            recorder_id=args.recorder_id,
            experiment_name=args.experiment,
            topk=args.topk,
            n_drop=args.n_drop,
            hold_thresh=args.hold_thresh,
            start_time=args.start,
            end_time=args.end,
        )

    elif args.command == "predict":
        new_rid = generate_predictions_for_new_period(
            recorder_id=args.recorder_id,
            experiment_name=args.experiment,
            test_start_time=args.start,
            test_end_time=args.end,
            new_experiment_name=args.new_experiment,
        )
        target_exp = args.new_experiment or f"{args.experiment}_new_period"
        run_backtest_only(
            recorder_id=new_rid,
            experiment_name=target_exp,
            start_time=args.start,
            end_time=args.end,
        )

    elif args.command == "layer":
        run_layer_analysis(
            recorder_id=args.recorder_id,
            experiment_name=args.experiment,
            n_groups=args.n_groups,
            output_dir=args.output_dir,
        )

    elif args.command == "analyze":
        analyze_feature_importance(
            recorder_id=args.recorder_id,
            experiment_name=args.experiment,
            top_n=args.top_n,
            output_path=args.output
            or os.path.join(os.getcwd(), "feature_importance", "feature_importance.html"),
        )

    elif args.command == "bridge":
        try:
            from workflow_backtrader_bridge import run_backtrader_bridge
        except ImportError:
            raise SystemExit(
                "无法导入 workflow_backtrader_bridge。请确保 backtrader 已安装，"
                "并从 examples/ 目录运行此命令。"
            )
        run_backtrader_bridge(
            recorder_id=args.recorder_id,
            experiment_name=args.experiment,
            topk=args.topk,
            n_drop=args.n_drop,
            hold_thresh=args.hold_thresh,
            start_time=args.start,
            end_time=args.end,
            verbose=args.verbose,
            plot=bool(args.plot_output),
            plot_output=args.plot_output,
        )

    elif args.command == "list":
        print(f"\n实验 '{args.experiment}' 中的 Recorder:\n")
        _list_recorders(args.experiment)


if __name__ == "__main__":
    main()
