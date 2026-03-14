#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib Workflow Example with Custom Strategy

将模型训练和策略回测分离：
1. train_model() - 训练模型并保存预测结果
2. run_backtest_only() - 仅运行回测（使用已保存的预测结果）
3. generate_predictions_for_new_period() - 基于已训练模型为新时间段生成预测

这样修改策略参数或回测时间段时不需要重新训练模型。

新时间段回测流程：
1. 使用 get_extended_dataset_config 创建新数据集（指定新的 test 时间段）
2. 从 recorder 加载已训练的模型
3. 使用 SignalRecord 生成新时间段的预测
4. 保存到新的 recorder 路径
5. 使用新 recorder 运行回测
"""
import os
from pprint import pprint
from pathlib import Path
from typing import List, Union

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_MARKET, GBDT_MODEL
from qlib.backtest import backtest as run_backtest
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

# Import custom strategy and utils (optional, comment out if not available)
# from strategies import ThresholdTopkDropoutStrategy
# from utils import print_stock_analysis


# ============================================================
# 模型配置
# ============================================================

GBDT_MODEL_CONFIG = GBDT_MODEL

MLP_MODEL_CONFIG = {
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
}

MLP_DEEP_MODEL_CONFIG = {
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
}


# ============================================================
# 策略配置（修改这里不会触发重新训练）
# ============================================================

STRATEGY_CONFIG = {
    "topk": 20,          # 持仓股票数量
    "n_drop": 2,         # 每天最多替换的股票数量
    "hold_thresh": 1,    # 最小持有天数
}


def get_latest_calendar_date(calendar_path: str = "~/.qlib/qlib_data/cn_data/calendars/day.txt") -> str:
    calendar_file = Path(calendar_path).expanduser()
    if not calendar_file.exists():
        return "2025-08-01"

    with calendar_file.open("r", encoding="utf-8") as f:
        trading_days = [line.strip() for line in f if line.strip()]

    return trading_days[-1] if trading_days else "2025-08-01"


LATEST_BACKTEST_END = get_latest_calendar_date()
# LATEST_BACKTEST_END = "2025-08-01"


BACKTEST_CONFIG = {
    "start_time": "2024-01-01",
    "end_time": LATEST_BACKTEST_END,
    "account": 1000000,
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
# 数据集配置
# ============================================================


def ensure_qlib_initialized():
    provider_uri = "~/.qlib/qlib_data/cn_data"
    try:
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)


def get_union_instruments_for_period(
    market: str,
    start_time: str,
    end_time: str,
) -> List[str]:
    """
    Return the union of index constituents during the given period.

    This keeps prediction coverage continuous for stocks that leave the index
    mid-period. Without this expansion, pred.pkl only contains scores for the
    current constituents on each date, which can create artificial signal gaps
    in downstream backtests.
    """
    ensure_qlib_initialized()
    instruments = D.instruments(market)
    union_instruments = D.list_instruments(
        instruments,
        start_time=start_time,
        end_time=end_time,
        as_list=True,
        freq="day",
    )
    return sorted(union_instruments)


def resolve_prediction_instruments(
    instruments: Union[str, List[str]],
    test_start_time: str,
    test_end_time: str,
    use_union_constituents: bool = True,
) -> Union[str, List[str]]:
    """
    Expand a dynamic index universe into its period union for prediction.

    The strategy/backtest can still choose how to trade, but pred generation is
    made stable across constituent changes so that removed stocks do not lose
    scores purely because the index membership rolled.
    """
    if not use_union_constituents or not isinstance(instruments, str):
        return instruments

    union_instruments = get_union_instruments_for_period(
        market=instruments,
        start_time=test_start_time,
        end_time=test_end_time,
    )
    print(
        f"使用 {instruments} 在 {test_start_time} ~ {test_end_time} 的成分股并集，"
        f"共 {len(union_instruments)} 只股票生成预测。"
    )
    return union_instruments


def get_extended_dataset_config(
    train_periods=None,
    valid=("2022-01-01", "2023-12-31"),
    test=("2024-01-01", "2025-08-01"),
    instruments=CSI300_MARKET,
    end_time="2025-08-01",
):
    """创建扩展的数据集配置"""
    if train_periods is None:
        train_periods = [
            ("2010-01-01", "2014-12-31"),
            ("2016-01-01", "2019-12-31"),
            ("2020-06-01", "2021-12-31"),
        ]
    
    start_time = min(period[0] for period in train_periods)
    train_start = min(period[0] for period in train_periods)
    train_end = max(period[1] for period in train_periods)
    
    infer_processors = [
        {"class": "ProcessInf", "kwargs": {}},
        {
            "class": "ZScoreNorm",
            "kwargs": {
                "fit_start_time": train_start,
                "fit_end_time": train_end,
            },
        },
        {"class": "Fillna", "kwargs": {}},
    ]
    
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]
    
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "fit_start_time": train_start,
                    "fit_end_time": train_end,
                    "instruments": instruments,
                    "infer_processors": infer_processors,
                    "learn_processors": learn_processors,
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
# 训练函数
# ============================================================

def train_model(model_type="gbdt", experiment_name="workflow", use_union_constituents=True):
    """
    训练模型并保存预测结果
    
    Parameters
    ----------
    model_type : str
        模型类型: "gbdt", "mlp", "mlp_deep"
    experiment_name : str
        实验名称
        
    Returns
    -------
    str
        recorder_id，用于后续回测
    """
    # 初始化 Qlib
    ensure_qlib_initialized()
    
    # 选择模型
    if model_type == "gbdt":
        model_config = GBDT_MODEL_CONFIG
        print("使用模型: LightGBM (GBDT)")
    elif model_type == "mlp":
        model_config = MLP_MODEL_CONFIG
        print("使用模型: MLP 神经网络")
    elif model_type == "mlp_deep":
        model_config = MLP_DEEP_MODEL_CONFIG
        print("使用模型: 深层 MLP 神经网络")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 数据集配置
    # By default, generate predictions on the union of constituents in the
    # backtest period instead of the point-in-time index members. This avoids
    # score discontinuities for stocks that leave the index during the test
    # window while still preserving a bounded universe.
    test_period = (
        BACKTEST_CONFIG["start_time"],
        BACKTEST_CONFIG["end_time"],
    )
    prediction_instruments = resolve_prediction_instruments(
        instruments=CSI300_MARKET,
        test_start_time=test_period[0],
        test_end_time=test_period[1],
        use_union_constituents=use_union_constituents,
    )
    extended_dataset_config = get_extended_dataset_config(
        test=test_period,
        instruments=prediction_instruments,
        end_time=test_period[1],
    )
    
    custom_task = {
        "model": model_config,
        "dataset": extended_dataset_config,
    }
    
    model = init_instance_by_config(custom_task["model"])
    dataset = init_instance_by_config(custom_task["dataset"])
    
    # 打印数据集信息
    print("\n" + "="*80)
    print("数据集配置信息")
    print("="*80)
    segments = custom_task["dataset"]["kwargs"]["segments"]
    print(f"训练集: {segments['train'][0]} ~ {segments['train'][1]}")
    print(f"验证集: {segments['valid'][0]} ~ {segments['valid'][1]}")
    print(f"测试集: {segments['test'][0]} ~ {segments['test'][1]}")
    
    example_df = dataset.prepare("train")
    print(f"\n训练数据集大小: {len(example_df)} 条记录")
    
    # 开始训练
    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(custom_task))
        
        print("\n" + "="*80)
        print("开始训练模型...")
        print("="*80)
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})
        
        recorder = R.get_recorder()
        recorder_id = recorder.id
        
        # 生成预测信号
        print("\n生成预测信号...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 信号分析
        print("\n信号分析...")
        sar = SigAnaRecord(recorder)
        sar.generate()
        
        print("\n" + "="*80)
        print(f"训练完成！Recorder ID: {recorder_id}")
        print(f"预测结果已保存，可以使用 run_backtest_only('{recorder_id}') 运行回测")
        print("="*80)
        
        return recorder_id


# ============================================================
# 为新时间段生成预测（不需要重新训练）
# ============================================================

def generate_predictions_for_new_period(
    recorder_id=None,
    experiment_name="workflow",
    test_start_time=None,
    test_end_time=None,
    instruments=CSI300_MARKET,
    new_experiment_name=None,
    use_union_constituents=True,
):
    """
    基于已训练的模型，为新时间段生成预测
    
    使用 get_extended_dataset_config 创建新数据集，从 recorder 加载模型，
    使用 SignalRecord 生成预测并保存到新的 recorder 路径。
    
    Parameters
    ----------
    recorder_id : str, optional
        Recorder ID，如果为 None 则使用最新的 recorder
    experiment_name : str
        原始实验名称
    test_start_time : str, optional
        新测试时间段的开始时间，格式：'YYYY-MM-DD'
        如果为 None，使用 BACKTEST_CONFIG 中的 start_time
    test_end_time : str, optional
        新测试时间段的结束时间，格式：'YYYY-MM-DD'
        如果为 None，使用 BACKTEST_CONFIG 中的 end_time
    instruments : str or list, optional
        股票池，默认使用 CSI300_MARKET
    new_experiment_name : str, optional
        新实验名称，如果为 None 则使用 experiment_name + "_new_period"
        
    Returns
    -------
    str
        新 recorder_id
    """
    # 初始化 Qlib
    ensure_qlib_initialized()
    
    # 获取 recorder
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id is None:
        recorders = exp.list_recorders(rtype=exp.RT_L)
        if not recorders:
            raise ValueError(f"实验 '{experiment_name}' 中没有找到 recorder，请先运行 train_model()")
        
        if isinstance(recorders, dict):
            recorder_list = [(rid, rec.info.get("start_time", 0) if hasattr(rec, 'info') else 0) 
                           for rid, rec in recorders.items()]
            recorder_list.sort(key=lambda x: x[1], reverse=True)
            recorder_id = recorder_list[0][0]
        else:
            try:
                sorted_recorders = sorted(
                    recorders,
                    key=lambda r: r.info.get("start_time", 0) if hasattr(r, 'info') else 0,
                    reverse=True
                )
                recorder_id = sorted_recorders[0].id if hasattr(sorted_recorders[0], 'id') else sorted_recorders[0]
            except:
                recorder_id = recorders[-1].id if hasattr(recorders[-1], 'id') else recorders[-1]
    
    recorder = exp.get_recorder(recorder_id=recorder_id)
    
    # 加载模型
    print("\n" + "="*80)
    print("加载已训练的模型...")
    print("="*80)
    model = recorder.load_object("params.pkl")
    
    # 确定新的测试时间段
    _test_start = test_start_time if test_start_time is not None else BACKTEST_CONFIG["start_time"]
    _test_end = test_end_time if test_end_time is not None else BACKTEST_CONFIG["end_time"]
    prediction_instruments = resolve_prediction_instruments(
        instruments=instruments,
        test_start_time=_test_start,
        test_end_time=_test_end,
        use_union_constituents=use_union_constituents,
    )
    
    # 使用 get_extended_dataset_config 创建新数据集配置
    # 保持训练时间段不变，只更新测试时间段和 end_time。默认将动态指数
    # 股票池展开为区间成分股并集，以避免 pred 在成分股调整时出现断层。
    new_dataset_config = get_extended_dataset_config(
        test=(_test_start, _test_end),
        instruments=prediction_instruments,
        end_time=_test_end,  # 确保 end_time 覆盖测试时间段
    )
    
    print(f"\n新数据集配置:")
    print(f"  测试时间段: {_test_start} ~ {_test_end}")
    if isinstance(prediction_instruments, list):
        print(f"  股票池: {instruments} 成分股并集 ({len(prediction_instruments)} 只)")
    else:
        print(f"  股票池: {prediction_instruments}")
    
    # 创建新数据集
    dataset = init_instance_by_config(new_dataset_config)
    
    # 创建新的实验和 recorder
    _new_experiment_name = new_experiment_name if new_experiment_name else f"{experiment_name}_new_period"
    
    print("\n" + "="*80)
    print("生成预测并保存到新 recorder...")
    print("="*80)
    
    # 使用 SignalRecord 生成预测（参考训练流程）
    with R.start(experiment_name=_new_experiment_name):
        R.log_params(**flatten_dict({
            "source_recorder_id": recorder_id,
            "source_experiment": experiment_name,
            "test_period": (_test_start, _test_end),
        }))
        
        new_recorder = R.get_recorder()
        new_recorder_id = new_recorder.id
        
        # 保存模型和数据集配置
        new_recorder.save_objects(**{"params.pkl": model, "dataset": dataset})
        
        # 生成预测信号（参考训练流程）
        sr = SignalRecord(model, dataset, new_recorder)
        sr.generate()
        
        print("\n" + "="*80)
        print(f"预测生成完成！")
        print(f"  新 Recorder ID: {new_recorder_id}")
        print(f"  新实验名称: {_new_experiment_name}")
        print("="*80)
        
        return new_recorder_id


# ============================================================
# 回测函数（不需要重新训练）
# ============================================================

def run_backtest_only(
    recorder_id=None,
    experiment_name="workflow",
    topk=20,
    n_drop=2,
    hold_thresh=10,
    start_time=None,
    end_time=None,
):
    """
    仅运行回测（使用已保存的预测结果）
    
    Parameters
    ----------
    recorder_id : str, optional
        Recorder ID，如果为 None 则使用最新的 recorder
    experiment_name : str
        实验名称
    topk : int, optional
        持仓股票数量，如果为 None 使用 STRATEGY_CONFIG 中的值
    n_drop : int, optional
        每天最多替换的股票数量
    hold_thresh : int, optional
        最小持有天数
    start_time : str, optional
        回测开始时间，格式：'YYYY-MM-DD'
        如果为 None，使用 BACKTEST_CONFIG 中的 start_time
    end_time : str, optional
        回测结束时间，格式：'YYYY-MM-DD'
        如果为 None，使用 BACKTEST_CONFIG 中的 end_time
    """
    # 初始化 Qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    try:
        # 检查是否已初始化（通过尝试获取数据来判断）
        from qlib.data import D
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        # 未初始化，执行初始化
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    # 确定回测时间段
    _start_time = start_time if start_time is not None else BACKTEST_CONFIG["start_time"]
    _end_time = end_time if end_time is not None else BACKTEST_CONFIG["end_time"]
    
    # 使用传入的参数或默认配置
    _topk = topk if topk is not None else STRATEGY_CONFIG["topk"]
    _n_drop = n_drop if n_drop is not None else STRATEGY_CONFIG["n_drop"]
    _hold_thresh = hold_thresh if hold_thresh is not None else STRATEGY_CONFIG["hold_thresh"]
    
    print("\n" + "="*80)
    print("回测配置")
    print("="*80)
    print(f"  回测时间段: {_start_time} ~ {_end_time}")
    print(f"  topk: {_topk}")
    print(f"  n_drop: {_n_drop}")
    print(f"  hold_thresh: {_hold_thresh}")
    
    # 获取 recorder
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id is None:
        # 使用最新的 recorder（按创建时间排序）
        recorders = exp.list_recorders(rtype=exp.RT_L)
        if not recorders:
            raise ValueError(f"实验 '{experiment_name}' 中没有找到 recorder，请先运行 train_model()")
        
        # 获取所有 recorder 并按开始时间排序，取最新的
        if isinstance(recorders, dict):
            # 获取每个 recorder 的信息并排序
            recorder_list = []
            for rid, rec in recorders.items():
                try:
                    rec_start_time = rec.info.get("start_time", 0) if hasattr(rec, 'info') else 0
                    recorder_list.append((rid, rec_start_time))
                except:
                    recorder_list.append((rid, 0))
            # 按开始时间排序，取最新的
            recorder_list.sort(key=lambda x: x[1], reverse=True)
            recorder_id = recorder_list[0][0]
        else:
            # 是列表，尝试按时间排序
            try:
                sorted_recorders = sorted(
                    recorders,
                    key=lambda r: r.info.get("start_time", 0) if hasattr(r, 'info') else 0,
                    reverse=True
                )
                recorder_id = sorted_recorders[0].id if hasattr(sorted_recorders[0], 'id') else sorted_recorders[0]
            except:
                # 如果排序失败，取最后一个
                recorder_id = recorders[-1].id if hasattr(recorders[-1], 'id') else recorders[-1]
        
        print(f"\n使用最新的 Recorder: {recorder_id}")
        
        # 列出所有可用的 recorder
        print("\n可用的 Recorder 列表:")
        if isinstance(recorders, dict):
            for i, (rid, rec) in enumerate(recorders.items()):
                try:
                    rec_start_time = rec.info.get("start_time", "N/A") if hasattr(rec, 'info') else "N/A"
                    print(f"  [{i}] {rid} (创建时间: {rec_start_time})")
                except:
                    print(f"  [{i}] {rid}")
        else:
            for i, rec in enumerate(recorders):
                rid = rec.id if hasattr(rec, 'id') else rec
                try:
                    rec_start_time = rec.info.get("start_time", "N/A") if hasattr(rec, 'info') else "N/A"
                    print(f"  [{i}] {rid} (创建时间: {rec_start_time})")
                except:
                    print(f"  [{i}] {rid}")
    
    recorder = exp.get_recorder(recorder_id=recorder_id)
    
    # 检查预测结果
    print("\n" + "="*80)
    print("检查预测结果...")
    print("="*80)
    
    try:
        pred = recorder.load_object("pred.pkl")
        print(f"已加载预测数据，形状: {pred.shape}")
        if len(pred) > 0:
            dt_min = pred.index.get_level_values('datetime').min()
            dt_max = pred.index.get_level_values('datetime').max()
            print(f"预测数据时间范围: {dt_min} ~ {dt_max}")
            print(f"回测时间范围: {_start_time} ~ {_end_time}")
            
            # 显示预测数据样本
            print(f"\n预测数据前5行:")
            print(pred.head())
            print(f"\n预测数据后5行:")
            print(pred.tail())
        else:
            raise ValueError("预测数据为空，请先运行 train_model() 或 generate_predictions_for_new_period()")
    except Exception as e:
        raise ValueError(f"无法加载预测结果: {e}，请先运行 train_model() 或 generate_predictions_for_new_period()")
    
    # 构建回测配置
    backtest_config = BACKTEST_CONFIG.copy()
    backtest_config["start_time"] = _start_time
    backtest_config["end_time"] = _end_time
    
    # 注意：使用 "<PRED>" 占位符，PortAnaRecord 会自动从 recorder 加载 pred.pkl 并替换
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
                "signal": "<PRED>",  # 使用占位符，PortAnaRecord 会自动替换为 pred.pkl
                "topk": _topk,
                "n_drop": _n_drop,
                "hold_thresh": _hold_thresh,
            },
        },
        "backtest": backtest_config,
    }
    
    # 运行回测
    print("\n" + "="*80)
    print("运行回测...")
    print("="*80)
    
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
    
    print("\n" + "="*80)
    print("回测完成！")
    print("="*80)


# ============================================================
# 分层图分析（基于 pred + label 的分组收益分析）
# ============================================================

def run_layer_analysis(
    recorder_id=None,
    experiment_name="workflow",
    output_dir=None,
    n_groups=5,
    graph_names=None,
):
    """
    基于分层图进行回测结果分析
    
    将预测分数按大小分为 N 组（默认 5 组），分析各组收益表现，
    生成 Group1~Group5 累计收益曲线、long-short、long-average 等图表。
    
    Parameters
    ----------
    recorder_id : str, optional
        Recorder ID，如果为 None 则使用最新的 recorder
    experiment_name : str
        实验名称
    output_dir : str, optional
        图表输出目录，如果为 None 则使用当前目录下的 layer_analysis/
    n_groups : int
        分组数量，默认 5
    graph_names : list, optional
        要生成的图表类型，默认 ["group_return", "pred_ic", "pred_autocorr"]
        - group_return: 分层累计收益图（Group1~Group5、long-short、long-average）
        - pred_ic: 信息系数 IC 图
        - pred_autocorr: 预测自相关图
        - pred_turnover: 换手率图
        
    Returns
    -------
    list
        生成的 plotly Figure 列表
    """
    # 初始化 Qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    try:
        from qlib.data import D
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    # 获取 recorder
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id is None:
        recorders = exp.list_recorders(rtype=exp.RT_L)
        if not recorders:
            raise ValueError(f"实验 '{experiment_name}' 中没有找到 recorder")
        if isinstance(recorders, dict):
            recorder_list = [(rid, rec.info.get("start_time", 0) if hasattr(rec, 'info') else 0)
                           for rid, rec in recorders.items()]
            recorder_list.sort(key=lambda x: x[1], reverse=True)
            recorder_id = recorder_list[0][0]
        else:
            try:
                sorted_recorders = sorted(
                    recorders,
                    key=lambda r: r.info.get("start_time", 0) if hasattr(r, 'info') else 0,
                    reverse=True
                )
                recorder_id = sorted_recorders[0].id if hasattr(sorted_recorders[0], 'id') else sorted_recorders[0]
            except Exception:
                recorder_id = recorders[-1].id if hasattr(recorders[-1], 'id') else recorders[-1]
    
    recorder = exp.get_recorder(recorder_id=recorder_id)
    
    # 加载 pred 和 label
    print("\n" + "="*80)
    print("分层图分析")
    print("="*80)
    print(f"  Recorder ID: {recorder_id}")
    print(f"  实验名称: {experiment_name}")
    
    pred = recorder.load_object("pred.pkl")
    label = recorder.load_object("label.pkl")
    
    if pred is None or pred.empty:
        raise ValueError("无法加载 pred.pkl，请确保已运行 train_model 或 generate_predictions_for_new_period")
    
    if label is None or label.empty:
        raise ValueError("无法加载 label.pkl，分层分析需要 label 数据。请使用 train_model 或 generate_predictions_for_new_period 生成的 recorder")
    
    # 构建 pred_label：需要 score 和 label 列，index 为 (instrument, datetime)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: "score"})
    
    if isinstance(label, pd.DataFrame):
        label_series = label.iloc[:, 0]
    else:
        label_series = label
    
    pred_label = pred.copy()
    pred_label["label"] = label_series.reindex(pred.index)
    pred_label = pred_label.dropna(subset=["label", "score"])
    
    if pred_label.empty:
        raise ValueError("pred 与 label 合并后无有效数据，请检查时间范围是否一致")
    
    print(f"  有效数据量: {len(pred_label)} 条")
    print(f"  时间范围: {pred_label.index.get_level_values('datetime').min()} ~ {pred_label.index.get_level_values('datetime').max()}")
    
    # 生成分层图
    if graph_names is None:
        graph_names = ["group_return", "pred_ic", "pred_autocorr"]
    
    figure_list = model_performance_graph(
        pred_label=pred_label,
        N=n_groups,
        graph_names=graph_names,
        show_notebook=False,
    )
    
    # 保存图表到文件
    _output_dir = output_dir or os.path.join(os.getcwd(), "layer_analysis")
    os.makedirs(_output_dir, exist_ok=True)
    
    for i, fig in enumerate(figure_list):
        out_path = os.path.join(_output_dir, f"layer_analysis_{i+1}.html")
        fig.write_html(out_path)
        print(f"  已保存: {out_path}")
    
    print("\n" + "="*80)
    print("分层图分析完成！")
    print("="*80)
    
    return figure_list


# ============================================================
# 辅助函数
# ============================================================

def _save_and_analyze_results(recorder, portfolio_metric_dict, indicator_dict):
    """保存回测结果并进行风险分析"""
    for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"report_normal_{_freq}.pkl": report_normal}
        )
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"positions_normal_{_freq}.pkl": positions_normal}
        )
    
    for _freq, indicators_normal in indicator_dict.items():
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"indicators_normal_{_freq}.pkl": indicators_normal[0]}
        )
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]}
        )
    
    for _freq, (report_normal, _) in portfolio_metric_dict.items():
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=_freq
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_freq
        )
        
        analysis_df = pd.concat(analysis)
        
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"port_analysis_{_freq}.pkl": analysis_df}
        )
        
        analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
        recorder.log_metrics(**{f"threshold.{_freq}.{k}": v for k, v in analysis_dict.items()})
        
        pprint(f"The following are analysis results of benchmark return({_freq}).")
        pprint(risk_analysis(report_normal["bench"], freq=_freq))
        pprint(f"The following are analysis results of the excess return without cost({_freq}).")
        pprint(analysis["excess_return_without_cost"])
        pprint(f"The following are analysis results of the excess return with cost({_freq}).")
        pprint(analysis["excess_return_with_cost"])
    
    for _freq, indicators_normal in indicator_dict.items():
        indicators_df = indicators_normal[0]
        analysis_df = indicator_analysis(indicators_df)
        
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"indicator_analysis_{_freq}.pkl": analysis_df}
        )
        
        analysis_dict = analysis_df["value"].to_dict()
        recorder.log_metrics(**{f"threshold.{_freq}.{k}": v for k, v in analysis_dict.items()})
        
        pprint(f"The following are analysis results of indicators({_freq}).")
        pprint(analysis_df)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主函数 - 演示训练和回测分离，支持新时间段回测
    
    使用方式：
    1. 首次运行或需要重新训练时：
       python workflow_by_code_v2.py train
       
    2. 仅修改策略参数运行回测：
       python workflow_by_code_v2.py backtest
       
    3. 或者在 Python 中：
       from workflow_by_code_v2 import train_model, run_backtest_only, generate_predictions_for_new_period
       
       # 训练（只需运行一次）
       recorder_id = train_model(model_type="gbdt")
       
       # 回测（可以多次运行，修改参数）
       run_backtest_only(topk=20, n_drop=2, hold_thresh=1)
       run_backtest_only(topk=50, n_drop=5, hold_thresh=3)
       
       # 为新时间段生成预测
       new_recorder_id = generate_predictions_for_new_period(
           test_start_time="2025-01-01",
           test_end_time="2025-12-31"
       )
       
       # 使用新 recorder 回测
       run_backtest_only(recorder_id=new_recorder_id, experiment_name="workflow_new_period")
       
       # 分层图分析（基于 pred + label 的分组收益分析）
       run_layer_analysis(
           recorder_id="9bc75beec25c442c9830551cc401c094",
           experiment_name="workflow_new_period",
       )
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            # 训练模型
            model_type = sys.argv[2] if len(sys.argv) > 2 else "gbdt"
            train_model(model_type=model_type)
            
        elif command == "backtest":
            # 仅回测
            run_backtest_only(
                recorder_id="9bc75beec25c442c9830551cc401c094",
                experiment_name="workflow_new_period",
                start_time="2024-01-01",
                end_time="2025-08-01",
            )
        
        elif command == "layer":
            # 分层图分析（基于指定 recorder 的 pred + label）
            run_layer_analysis(
                recorder_id="9bc75beec25c442c9830551cc401c094",
                experiment_name="workflow_new_period",
                output_dir=os.path.join(os.getcwd(), "layer_analysis"),
                n_groups=5,
                graph_names=["group_return", "pred_ic", "pred_autocorr"],
            )
        
        elif command == "backtest_new_period":
            # 回测新时间段
            new_recorder_id = generate_predictions_for_new_period(
                test_start_time="2024-01-01", test_end_time="2025-12-24",
            )
            run_backtest_only(
                recorder_id=new_recorder_id,
                experiment_name="workflow_new_period",
                start_time="2024-01-01",
                end_time="2025-08-01",
            )
            
        else:
            print(f"未知命令: {command}")
            print("使用方式:")
            print("  python workflow_by_code_v2.py train [gbdt|mlp|mlp_deep]")
            print("  python workflow_by_code_v2.py backtest")
            print("  python workflow_by_code_v2.py layer   # 分层图分析")
    else:
        # 默认：训练 + 回测
        print("="*80)
        print("默认模式：训练模型 + 运行回测")
        print("="*80)
        print("\n提示：")
        print("  - 仅训练: python workflow_by_code_v2.py train")
        print("  - 仅回测: python workflow_by_code_v2.py backtest")
        print("  - 分层图分析: python workflow_by_code_v2.py layer")
        print("="*80)
        
        recorder_id = train_model(model_type="gbdt")
        run_backtest_only(recorder_id=recorder_id)


if __name__ == "__main__":
    main()
