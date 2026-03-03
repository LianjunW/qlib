#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib Workflow Example with Custom Strategy

将模型训练和策略回测分离：
1. train_model() - 训练模型并保存预测结果（支持买入和卖出两个模型）
2. run_backtest_only() - 仅运行回测（使用已保存的预测结果）
3. generate_predictions_for_new_period() - 基于已训练模型为新时间段生成预测

这样修改策略参数或回测时间段时不需要重新训练模型。

双模型设计：
- 买入模型：预测第三天对第二天的涨跌幅 (Ref($close, -2)/Ref($close, -1) - 1)
  用于买入决策，选择未来收益潜力大的股票
- 卖出模型：预测第二天对当前的涨跌幅 (Ref($close, -1)/$close - 1)
  用于卖出决策，判断已持有股票的未来表现

流程检测：若双模型效果较差，可用 sell_label_same_as_buy=True 让卖出模型使用与买入模型相同的 label，
以判断是 label 设计问题还是流程问题。若收益率改善，则可能是卖出模型 label 设计问题。

新时间段回测流程：
1. 使用 get_extended_dataset_config 创建新数据集（指定新的 test 时间段）
2. 从 recorder 加载已训练的模型
3. 使用 SignalRecord 生成新时间段的预测
4. 保存到新的 recorder 路径
5. 使用新 recorder 运行回测
"""
import argparse
import os
from pprint import pprint

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_MARKET, GBDT_MODEL
from qlib.backtest import backtest as run_backtest
from qlib.contrib.evaluate import risk_analysis, indicator_analysis

# Import custom strategy and utils (optional, comment out if not available)
# from strategies import ThresholdTopkDropoutStrategy
# from utils import print_stock_analysis

# Import for custom strategy
import copy
import numpy as np
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.backtest.signal import SignalWCache
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy


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

# 回测时间段需与 dataset test 一致，否则预测数据不覆盖
TEST_START_TIME = "2024-01-01"
TEST_END_TIME = "2025-08-01"

BACKTEST_CONFIG = {
    "start_time": TEST_START_TIME,
    "end_time": TEST_END_TIME,
    "account": 100000000,
    "benchmark": CSI300_BENCH,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
        "trade_unit": 100,  # 中国 A 股最小交易单位：1 手 = 100 股
    },
}


# ============================================================
# 数据集配置
# ============================================================

def get_extended_dataset_config(
    train_periods=None,
    valid=("2022-01-01", "2023-12-31"),
    test=(TEST_START_TIME, TEST_END_TIME),
    instruments=CSI300_MARKET,
    end_time=TEST_END_TIME,
    label_type="buy",
):
    """
    创建扩展的数据集配置
    
    Parameters
    ----------
    label_type : str
        "buy": 买入模型label，第三天对第二天的涨跌幅 (Ref($close, -2)/Ref($close, -1) - 1)
        "sell": 卖出模型label，第二天对当前的涨跌幅 (Ref($close, -1)/$close - 1)
    """
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
    
    # 根据label_type设置不同的label
    # label_config 格式应该是 (表达式列表, 名称列表)，与 get_label_config() 返回格式一致
    if label_type == "buy":
        # 买入模型：第三天对第二天的涨跌幅（原有逻辑）
        label_config = (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
    elif label_type == "sell":
        # 卖出模型：第二天对当前的涨跌幅（用于已持有股票的卖出决策）
        label_config = (["Ref($close, -1)/$close - 1"], ["LABEL0"])
    else:
        raise ValueError(f"未知的label_type: {label_type}，应为 'buy' 或 'sell'")
    
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
                    "label": label_config,  # 自定义label
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

def train_model(model_type="mlp", experiment_name="workflow", train_both_models=True, sell_label_same_as_buy=False):
    """
    训练模型并保存预测结果
    
    现在支持训练两个模型：
    1. 买入模型：预测第三天对第二天的涨跌幅（用于买入决策）
    2. 卖出模型：预测第二天对当前的涨跌幅（用于卖出决策）
    
    Parameters
    ----------
    model_type : str
        模型类型: "gbdt", "mlp", "mlp_deep"
    experiment_name : str
        实验名称
    train_both_models : bool
        是否训练两个模型（买入和卖出），如果为False则只训练买入模型（兼容旧版本）
    sell_label_same_as_buy : bool
        是否让卖出模型使用与买入模型相同的 label（用于检测流程问题）
        若为 True，卖出模型将使用 Ref($close,-2)/Ref($close,-1)-1，与买入模型一致
        用于验证：若收益率改善，说明可能是卖出模型 label 设计问题；否则可能是流程问题
        
    Returns
    -------
    dict
        包含 "buy_recorder_id" 和 "sell_recorder_id" 的字典
        如果 train_both_models=False，则只返回 "buy_recorder_id"
    """
    # 初始化 Qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN, default_disk_cache=0)
    
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
    
    result = {}
    
    # 训练买入模型
    print("\n" + "="*80)
    print("训练买入模型（第三天对第二天的涨跌幅）")
    print("="*80)
    
    buy_dataset_config = get_extended_dataset_config(label_type="buy")
    buy_task = {
        "model": model_config,
        "dataset": buy_dataset_config,
    }
    
    buy_model = init_instance_by_config(buy_task["model"])
    buy_dataset = init_instance_by_config(buy_task["dataset"])
    
    # 打印数据集信息
    print("\n买入模型数据集配置信息")
    print("="*80)
    segments = buy_task["dataset"]["kwargs"]["segments"]
    print(f"训练集: {segments['train'][0]} ~ {segments['train'][1]}")
    print(f"验证集: {segments['valid'][0]} ~ {segments['valid'][1]}")
    print(f"测试集: {segments['test'][0]} ~ {segments['test'][1]}")
    print(f"Label: Ref($close, -2)/Ref($close, -1) - 1 (第三天对第二天的涨跌幅)")
    
    example_df = buy_dataset.prepare("train")
    print(f"\n训练数据集大小: {len(example_df)} 条记录")
    
    # 开始训练买入模型
    with R.start(experiment_name=f"{experiment_name}_buy"):
        R.log_params(**flatten_dict({**buy_task, "model_type": "buy"}))
        
        print("\n开始训练买入模型...")
        fit_kwargs = {"verbose_eval": 0} if model_type == "gbdt" else {"verbose": False}
        buy_model.fit(buy_dataset, **fit_kwargs)
        R.save_objects(**{"params.pkl": buy_model})
        
        buy_recorder = R.get_recorder()
        buy_recorder_id = buy_recorder.id
        
        # 生成预测信号
        print("\n生成买入模型预测信号...")
        sr = SignalRecord(buy_model, buy_dataset, buy_recorder)
        sr.generate()
        
        # 信号分析
        print("\n买入模型信号分析...")
        sar = SigAnaRecord(buy_recorder)
        sar.generate()
        
        print(f"\n买入模型训练完成！Recorder ID: {buy_recorder_id}")
        result["buy_recorder_id"] = buy_recorder_id
    
    # 训练卖出模型
    if train_both_models:
        print("\n" + "="*80)
        if sell_label_same_as_buy:
            print("训练卖出模型（使用与买入模型相同的 label，用于检测流程问题）")
        else:
            print("训练卖出模型（第二天对当前的涨跌幅）")
        print("="*80)
        
        # sell_label_same_as_buy=True 时使用买入模型的 label，用于验证流程
        sell_label_type = "buy" if sell_label_same_as_buy else "sell"
        sell_dataset_config = get_extended_dataset_config(label_type=sell_label_type)
        sell_task = {
            "model": model_config,
            "dataset": sell_dataset_config,
        }
        
        sell_model = init_instance_by_config(sell_task["model"])
        sell_dataset = init_instance_by_config(sell_task["dataset"])
        
        # 打印数据集信息
        print("\n卖出模型数据集配置信息")
        print("="*80)
        segments = sell_task["dataset"]["kwargs"]["segments"]
        print(f"训练集: {segments['train'][0]} ~ {segments['train'][1]}")
        print(f"验证集: {segments['valid'][0]} ~ {segments['valid'][1]}")
        print(f"测试集: {segments['test'][0]} ~ {segments['test'][1]}")
        if sell_label_same_as_buy:
            print(f"Label: Ref($close, -2)/Ref($close, -1) - 1 (与买入模型相同，第三天对第二天的涨跌幅)")
        else:
            print(f"Label: Ref($close, -1)/$close - 1 (第二天对当前的涨跌幅)")
        
        example_df = sell_dataset.prepare("train")
        print(f"\n训练数据集大小: {len(example_df)} 条记录")
        
        # 开始训练卖出模型
        with R.start(experiment_name=f"{experiment_name}_sell"):
            R.log_params(**flatten_dict({**sell_task, "model_type": "sell"}))
            
            print("\n开始训练卖出模型...")
            # 创建 evals_result 字典来保存训练过程中的loss
            sell_evals_result = {}
            sell_fit_kwargs = {"evals_result": sell_evals_result, "verbose_eval": 0} if model_type == "gbdt" else {"evals_result": sell_evals_result, "verbose": False}
            sell_model.fit(sell_dataset, **sell_fit_kwargs)
            R.save_objects(**{"params.pkl": sell_model})
            
            sell_recorder = R.get_recorder()
            sell_recorder_id = sell_recorder.id
            
            # 保存loss变化数据（静默保存，不打印）
            if sell_evals_result:
                # 将loss数据转换为DataFrame格式便于查看和分析
                # 兼容两种格式：GBDT 为 {"train": {"l2": [...]}}，PyTorch 为 {"train": [...]}
                loss_data = {}
                for dataset_name, metrics in sell_evals_result.items():
                    if isinstance(metrics, dict):
                        for metric_name, values in metrics.items():
                            key = f"{dataset_name}_{metric_name}"
                            loss_data[key] = values
                    elif isinstance(metrics, (list, np.ndarray)):
                        key = f"{dataset_name}_loss"
                        loss_data[key] = list(metrics)
                
                # 创建DataFrame
                if loss_data:
                    max_len = max(len(v) for v in loss_data.values())
                    # 确保所有列表长度一致（用NaN填充）
                    loss_df = pd.DataFrame({
                        k: v + [np.nan] * (max_len - len(v)) if len(v) < max_len else v
                        for k, v in loss_data.items()
                    })
                    # 添加step列
                    loss_df.insert(0, 'step', range(len(loss_df)))
                    
                    # 保存到recorder
                    sell_recorder.save_objects(**{"loss_history.pkl": loss_df})
                    sell_recorder.save_objects(**{"loss_history.csv": loss_df.to_csv(index=False)})
            
            # 生成预测信号
            print("\n生成卖出模型预测信号...")
            sr = SignalRecord(sell_model, sell_dataset, sell_recorder)
            sr.generate()
            
            # 信号分析
            print("\n卖出模型信号分析...")
            sar = SigAnaRecord(sell_recorder)
            sar.generate()
            
            print(f"\n卖出模型训练完成！Recorder ID: {sell_recorder_id}")
            result["sell_recorder_id"] = sell_recorder_id
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    if train_both_models:
        print(f"买入模型 Recorder ID: {result['buy_recorder_id']}")
        print(f"卖出模型 Recorder ID: {result['sell_recorder_id']}")
        print(f"可以使用 run_backtest_only() 运行回测（将自动使用两个模型）")
    else:
        print(f"买入模型 Recorder ID: {result['buy_recorder_id']}")
        print(f"预测结果已保存，可以使用 run_backtest_only('{result['buy_recorder_id']}') 运行回测")
    
    return result


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
    provider_uri = "~/.qlib/qlib_data/cn_data"
    try:
        from qlib.data import D
        D.calendar(start_time="2020-01-01", end_time="2020-01-02")
    except Exception:
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN, default_disk_cache=0)
    
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
    
    # 使用 get_extended_dataset_config 创建新数据集配置
    # 保持训练时间段不变，只更新测试时间段和 end_time
    new_dataset_config = get_extended_dataset_config(
        test=(_test_start, _test_end),
        instruments=instruments,
        end_time=_test_end,  # 确保 end_time 覆盖测试时间段
    )
    
    print(f"\n新数据集配置:")
    print(f"  测试时间段: {_test_start} ~ {_test_end}")
    print(f"  股票池: {instruments}")
    
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
    use_dual_model=True,
    buy_experiment_name=None,
    sell_experiment_name=None,
    buy_recorder_id=None,
    sell_recorder_id=None,
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
    
    # 规范化 recorder_id：如果是字典，提取 buy_recorder_id
    if recorder_id is not None:
        if isinstance(recorder_id, dict):
            recorder_id = recorder_id.get("buy_recorder_id", None)
        # 确保是字符串类型
        if recorder_id is not None and not isinstance(recorder_id, str):
            recorder_id = str(recorder_id)
    
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
    _use_dual_model = use_dual_model
    _buy_experiment_name = buy_experiment_name if buy_experiment_name else f"{experiment_name}_buy"
    _sell_experiment_name = sell_experiment_name if sell_experiment_name else f"{experiment_name}_sell"

    
    if recorder_id is None:
        # 双模型模式从 workflow_buy 获取 recorder_id（workflow 可能为空）
        exp_for_recorder = R.get_exp(experiment_name=_buy_experiment_name) if _use_dual_model else exp
        recorders = exp_for_recorder.list_recorders(rtype=exp_for_recorder.RT_L)
        if not recorders:
            raise ValueError(
                f"实验 '{_buy_experiment_name}' 中没有找到 recorder，请先运行 train_model()"
            )
        
        # 获取所有 recorder 并按开始时间排序，取最新的
        if isinstance(recorders, dict):
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
    
    # 加载预测结果
    print("\n" + "="*80)
    print("检查预测结果...")
    print("="*80)
    
    if _use_dual_model:
        # 使用双模型模式
        print("使用双模型模式（买入模型 + 卖出模型）")
        print(f"  买入实验: {_buy_experiment_name}, 卖出实验: {_sell_experiment_name}")
        
        # 加载买入模型预测
        if buy_recorder_id:
            buy_exp = R.get_exp(experiment_name=_buy_experiment_name)
            buy_recorder = buy_exp.get_recorder(recorder_id=buy_recorder_id)
        else:
            # 尝试从当前实验加载
            buy_exp = R.get_exp(experiment_name=_buy_experiment_name)
            buy_recorders = buy_exp.list_recorders(rtype=buy_exp.RT_L)
            if not buy_recorders:
                print(f"警告: 未找到买入模型recorder，尝试使用单模型模式")
                _use_dual_model = False
                # recorder_id 已经在函数开头规范化
                buy_recorder = exp.get_recorder(recorder_id=recorder_id) if recorder_id else None
                if buy_recorder is None:
                    raise ValueError("无法找到可用的recorder")
            else:
                if isinstance(buy_recorders, dict):
                    buy_recorder_id = list(buy_recorders.keys())[-1]
                else:
                    buy_recorder_id = buy_recorders[-1].id if hasattr(buy_recorders[-1], 'id') else buy_recorders[-1]
                buy_recorder = buy_exp.get_recorder(recorder_id=buy_recorder_id)
        
        print(f"买入模型 Recorder ID: {buy_recorder.id}")
        buy_pred = buy_recorder.load_object("pred.pkl")
        print(f"买入模型预测数据形状: {buy_pred.shape}")
        
        # 诊断买入模型预测数据
        if buy_pred.empty:
            raise ValueError("买入模型预测数据为空！请检查模型训练和预测生成过程。")
        
        # 检查数据时间范围
        if isinstance(buy_pred.index, pd.MultiIndex):
            dates = buy_pred.index.get_level_values('datetime')
            print(f"买入模型预测时间范围: {dates.min()} ~ {dates.max()}")
            print(f"买入模型预测日期数量: {len(dates.unique())}")
            
            # 检查预测值统计
            pred_values = buy_pred.iloc[:, 0] if isinstance(buy_pred, pd.DataFrame) else buy_pred
            print(f"买入模型预测值统计:")
            print(f"  - 均值: {pred_values.mean():.6f}")
            print(f"  - 标准差: {pred_values.std():.6f}")
            print(f"  - 最小值: {pred_values.min():.6f}")
            print(f"  - 最大值: {pred_values.max():.6f}")
            print(f"  - NaN数量: {pred_values.isna().sum()}")
            print(f"  - 非空数量: {pred_values.notna().sum()}")
        
        # 加载卖出模型预测
        if sell_recorder_id:
            sell_exp = R.get_exp(experiment_name=_sell_experiment_name)
            sell_recorder = sell_exp.get_recorder(recorder_id=sell_recorder_id)
        else:
            sell_exp = R.get_exp(experiment_name=_sell_experiment_name)
            sell_recorders = sell_exp.list_recorders(rtype=sell_exp.RT_L)
            if not sell_recorders:
                print(f"警告: 未找到卖出模型recorder，尝试使用单模型模式")
                _use_dual_model = False
                sell_recorder = buy_recorder
            else:
                if isinstance(sell_recorders, dict):
                    sell_recorder_id = list(sell_recorders.keys())[-1]
                else:
                    sell_recorder_id = sell_recorders[-1].id if hasattr(sell_recorders[-1], 'id') else sell_recorders[-1]
                sell_recorder = sell_exp.get_recorder(recorder_id=sell_recorder_id)
        
        if _use_dual_model:
            print(f"卖出模型 Recorder ID: {sell_recorder.id}")
            sell_pred = sell_recorder.load_object("pred.pkl")
            print(f"卖出模型预测数据形状: {sell_pred.shape}")
            
            # 诊断卖出模型预测数据
            if sell_pred.empty:
                raise ValueError("卖出模型预测数据为空！请检查模型训练和预测生成过程。")
            
            # 检查数据时间范围
            if isinstance(sell_pred.index, pd.MultiIndex):
                dates = sell_pred.index.get_level_values('datetime')
                print(f"卖出模型预测时间范围: {dates.min()} ~ {dates.max()}")
                print(f"卖出模型预测日期数量: {len(dates.unique())}")
                
                # 检查预测值统计
                pred_values = sell_pred.iloc[:, 0] if isinstance(sell_pred, pd.DataFrame) else sell_pred
                print(f"卖出模型预测值统计:")
                print(f"  - 均值: {pred_values.mean():.6f}")
                print(f"  - 标准差: {pred_values.std():.6f}")
                print(f"  - 最小值: {pred_values.min():.6f}")
                print(f"  - 最大值: {pred_values.max():.6f}")
                print(f"  - NaN数量: {pred_values.isna().sum()}")
                print(f"  - 非空数量: {pred_values.notna().sum()}")
            
            # 创建Signal对象
            buy_signal = SignalWCache(signal=buy_pred.iloc[:, 0] if isinstance(buy_pred, pd.DataFrame) else buy_pred)
            sell_signal = SignalWCache(signal=sell_pred.iloc[:, 0] if isinstance(sell_pred, pd.DataFrame) else sell_pred)
            
            # 使用 buy_recorder 保存结果（用于 PortAnaRecord）
            recorder = buy_recorder
            recorder.save_objects(**{"pred.pkl": buy_pred})  # 保存买入预测作为主要预测
            
            # 构建回测配置 - 使用自定义双模型策略
            backtest_config = BACKTEST_CONFIG.copy()
            backtest_config["start_time"] = _start_time
            backtest_config["end_time"] = _end_time
            
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
                    "class": DualModelTopkDropoutStrategy,  # 直接传递类对象，避免模块导入问题
                    "kwargs": {
                        "signal": buy_signal,  # 买入信号
                        "sell_signal": sell_signal,  # 卖出信号
                        "topk": _topk,
                        "n_drop": _n_drop,
                        "hold_thresh": _hold_thresh,
                    },
                },
                "backtest": backtest_config,
            }
        else:
            # 回退到单模型模式
            pred = buy_pred
            recorder = buy_recorder
    
    if not _use_dual_model:
        # 单模型模式（兼容旧版本）
        print("使用单模型模式（兼容旧版本）")
        recorder = exp.get_recorder(recorder_id=recorder_id)
        
        try:
            pred = recorder.load_object("pred.pkl")
            print(f"已加载预测数据，形状: {pred.shape}")
        except Exception as e:
            raise ValueError(f"无法加载预测结果: {e}，请先运行 train_model() 或 generate_predictions_for_new_period()")
        
        # 诊断预测数据
        if pred.empty:
            raise ValueError("预测数据为空！请检查模型训练和预测生成过程。")
        
        if len(pred) > 0:
            dt_min = pred.index.get_level_values('datetime').min()
            dt_max = pred.index.get_level_values('datetime').max()
            print(f"预测数据时间范围: {dt_min} ~ {dt_max}")
            print(f"回测时间范围: {_start_time} ~ {_end_time}")
            
            # 检查预测值统计
            pred_values = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
            print(f"预测值统计:")
            print(f"  - 均值: {pred_values.mean():.6f}")
            print(f"  - 标准差: {pred_values.std():.6f}")
            print(f"  - 最小值: {pred_values.min():.6f}")
            print(f"  - 最大值: {pred_values.max():.6f}")
            print(f"  - NaN数量: {pred_values.isna().sum()}")
            print(f"  - 非空数量: {pred_values.notna().sum()}")
        
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
# 自定义策略：使用买入和卖出两个模型
# ============================================================

class DualModelTopkDropoutStrategy(BaseSignalStrategy):
    """
    使用两个模型的TopkDropout策略：
    - 买入决策：使用买入模型的预测（第三天对第二天的涨跌幅）
    - 卖出决策：使用卖出模型的预测（第二天对当前的涨跌幅）
    """
    
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        sell_signal=None,  # 卖出模型的信号
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            持仓股票数量
        n_drop : int
            每天最多替换的股票数量
        method_sell : str
            卖出方法: "bottom" 或 "random"
        method_buy : str
            买入方法: "top" 或 "random"
        hold_thresh : int
            最小持有天数
        only_tradable : bool
            是否只考虑可交易股票
        forbid_all_trade_at_limit : bool
            是否禁止在涨跌停时交易
        sell_signal : Signal
            卖出模型的信号对象，如果为None则使用买入信号（兼容模式）
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.sell_signal = sell_signal if sell_signal is not None else self.signal  # 默认使用买入信号
    
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # 买入信号：使用买入模型（self.signal）
        buy_pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(buy_pred_score, pd.DataFrame):
            buy_pred_score = buy_pred_score.iloc[:, 0]
        
        # 卖出信号：使用卖出模型（self.sell_signal）
        sell_pred_score = self.sell_signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(sell_pred_score, pd.DataFrame):
            sell_pred_score = sell_pred_score.iloc[:, 0]
        
        if buy_pred_score is None or sell_pred_score is None:
            return TradeDecisionWO([], self)
        
        # 检查信号是否为空
        if len(buy_pred_score) == 0:
            print(f"警告: 买入信号为空 (时间: {pred_start_time} ~ {pred_end_time})")
            return TradeDecisionWO([], self)
        if len(sell_pred_score) == 0:
            print(f"警告: 卖出信号为空 (时间: {pred_start_time} ~ {pred_end_time})")
            return TradeDecisionWO([], self)
        
        def get_first_n(li, n):
            return list(li)[:n]

        def get_last_n(li, n):
            return list(li)[-n:]

        def filter_stock(li):
            return li
        
        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        
        
        last = sell_pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        
        # 使用买入信号选择要买入的股票
        if self.method_buy == "top":
            today = get_first_n(
                buy_pred_score[~buy_pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(buy_pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        
        # 使用卖出信号对组合（当前持仓+候选买入）进行排序
        comb = sell_pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        
        # 使用卖出信号确定要卖出的股票
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = pd.Index(candi)
        else:
            raise NotImplementedError(f"This type of input is not supported")
        
        # 确定要买入的股票
        buy = today[: len(sell) + self.topk - len(last)]
        
        # 检查最小持有天数
        time_per_step = self.trade_calendar.get_freq()
        for code in current_stock_list:
            if code in sell and current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                sell = sell.drop(code)
        
        # 生成卖出订单
        for code in sell:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            
            sell_amount = current_temp.get_stock_amount(code=code)
            if sell_amount > 0:
                factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
                sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                # 使用 deal_order 处理卖出订单，自动更新 position 和 cash
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # 更新 cash（卖出后现金增加）
                    cash += trade_val - trade_cost
        
        # 生成买入订单
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
        
        return TradeDecisionWO(sell_order_list + buy_order_list, self)


# ============================================================
# 辅助函数
# ============================================================

def plot_loss_curve(
    recorder_id=None,
    experiment_name="workflow_sell",
    save_path=None,
    show_plot=True,
):
    """
    绘制训练loss曲线
    
    Parameters
    ----------
    recorder_id : str, optional
        Recorder ID，如果为 None 则使用最新的 recorder
    experiment_name : str
        实验名称，默认为 "workflow_sell"（卖出模型）
    save_path : str, optional
        保存图片的路径，如果为 None 则不保存
    show_plot : bool
        是否显示图片（在支持的环境中）
        
    Returns
    -------
    pd.DataFrame
        loss数据DataFrame
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib未安装，无法绘制图表。请运行: pip install matplotlib")
        return None
    
    # 获取recorder
    exp = R.get_exp(experiment_name=experiment_name)
    if recorder_id is None:
        recorders = exp.list_recorders(rtype=exp.RT_L)
        if not recorders:
            raise ValueError(f"实验 '{experiment_name}' 中没有找到 recorder")
        
        if isinstance(recorders, dict):
            recorder_id = list(recorders.keys())[-1]
        else:
            recorder_id = recorders[-1].id if hasattr(recorders[-1], 'id') else recorders[-1]
    
    recorder = exp.get_recorder(recorder_id=recorder_id)
    
    # 加载loss数据
    try:
        loss_df = recorder.load_object("loss_history.pkl")
    except Exception as e:
        print(f"无法加载loss数据: {e}")
        print("请确保已经训练过模型并保存了loss数据")
        return None
    
    if loss_df is None or loss_df.empty:
        print("Loss数据为空")
        return None
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 找出所有loss列（排除step列）
    loss_columns = [col for col in loss_df.columns if col != 'step']
    
    if not loss_columns:
        print("未找到loss数据列")
        return loss_df
    
    # 绘制训练和验证loss
    for col in loss_columns:
        # 移除NaN值
        valid_data = loss_df[['step', col]].dropna()
        if len(valid_data) > 0:
            if 'train' in col.lower():
                plt.plot(valid_data['step'], valid_data[col], 
                        label=col, linewidth=2, alpha=0.8, linestyle='-')
            elif 'valid' in col.lower():
                plt.plot(valid_data['step'], valid_data[col], 
                        label=col, linewidth=2, alpha=0.8, linestyle='--')
            else:
                plt.plot(valid_data['step'], valid_data[col], 
                        label=col, linewidth=1.5, alpha=0.7)
    
    # 设置图表格式
    plt.title(f'Training Loss Curve (Recorder: {recorder_id[:8]}...)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Step/Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # 添加统计信息
    if 'valid' in str(loss_columns).lower():
        valid_cols = [col for col in loss_columns if 'valid' in col.lower()]
        if valid_cols:
            valid_col = valid_cols[0]
            valid_data = loss_df[['step', valid_col]].dropna()
            if len(valid_data) > 0:
                best_idx = valid_data[valid_col].idxmin()
                best_step = valid_data.loc[best_idx, 'step']
                best_loss = valid_data.loc[best_idx, valid_col]
                plt.axvline(x=best_step, color='red', linestyle=':', alpha=0.7,
                           label=f'Best: {best_loss:.6f} @ step {int(best_step)}')
                plt.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss曲线已保存到: {save_path}")
    
    # 显示图片（如果支持）
    if show_plot:
        try:
            plt.show()
        except:
            print("无法显示图片（可能在不支持GUI的环境中）")
    
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*80)
    print("Loss统计信息")
    print("="*80)
    for col in loss_columns:
        valid_data = loss_df[col].dropna()
        if len(valid_data) > 0:
            print(f"{col}:")
            print(f"  - 初始值: {valid_data.iloc[0]:.6f}")
            print(f"  - 最终值: {valid_data.iloc[-1]:.6f}")
            print(f"  - 最小值: {valid_data.min():.6f}")
            print(f"  - 最大值: {valid_data.max():.6f}")
            print(f"  - 平均值: {valid_data.mean():.6f}")
    
    return loss_df


def view_loss_curve(
    recorder_id=None,
    experiment_name="workflow_sell",
    model_type="sell",
):
    """
    快速查看loss曲线的便捷函数
    
    Parameters
    ----------
    recorder_id : str, optional
        Recorder ID
    experiment_name : str
        实验名称，如果为None则根据model_type自动确定
    model_type : str
        "buy" 或 "sell"，用于确定实验名称
    """
    if experiment_name is None or experiment_name == "workflow_sell":
        if model_type == "buy":
            experiment_name = "workflow_buy"
        else:
            experiment_name = "workflow_sell"
    
    # 生成保存路径
    save_path = f"loss_curve_{model_type}_{recorder_id[:8] if recorder_id else 'latest'}.png"
    
    return plot_loss_curve(
        recorder_id=recorder_id,
        experiment_name=experiment_name,
        save_path=save_path,
        show_plot=False,  # 服务器环境通常不支持显示
    )

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
    主函数 - 演示训练和回测分离，支持新时间段回测和双模型策略
    
    使用方式：
    1. 首次运行或需要重新训练时：
       python workflow_by_code_v2.py train
       python workflow_by_code_v2.py train mlp workflow       # 指定 experiment_name=workflow
       python workflow_by_code_v2.py train mlp workflow_mlp   # 使用 workflow_mlp_buy/sell
       python workflow_by_code_v2.py train gbdt workflow same_label   # 卖出模型用买入 label
       
       默认会训练两个模型：
       - 买入模型：预测第三天对第二天的涨跌幅（用于买入决策）
       - 卖出模型：预测第二天对当前的涨跌幅（用于卖出决策）
       
    2. 仅修改策略参数运行回测：
       python workflow_by_code_v2.py backtest
       
       默认使用双模型策略（买入用买入模型，卖出用卖出模型）
       
    3. 或者在 Python 中：
       from workflow_by_code_v2 import train_model, run_backtest_only, generate_predictions_for_new_period
       
       # 训练两个模型（只需运行一次）
       result = train_model(model_type="gbdt", train_both_models=True)
       # 检测流程：卖出模型用买入 label
       result = train_model(model_type="gbdt", train_both_models=True, sell_label_same_as_buy=True)
       buy_recorder_id = result["buy_recorder_id"]
       sell_recorder_id = result["sell_recorder_id"]
       
       # 回测（可以多次运行，修改参数）
       # 使用双模型策略
       run_backtest_only(
           topk=20, n_drop=2, hold_thresh=1,
           use_dual_model=True,
           buy_experiment_name="workflow_buy",
           sell_experiment_name="workflow_sell"
       )
       
       # 或者使用单模型策略（兼容旧版本）
       run_backtest_only(
           topk=20, n_drop=2, hold_thresh=1,
           use_dual_model=False,
           recorder_id=buy_recorder_id
       )
       
       # 为新时间段生成预测
       new_recorder_id = generate_predictions_for_new_period(
           test_start_time="2025-01-01",
           test_end_time="2025-12-31"
       )
       
       # 使用新 recorder 回测
       run_backtest_only(recorder_id=new_recorder_id, experiment_name="workflow_new_period")
    """
    parser = argparse.ArgumentParser(
        description="Qlib 工作流：训练模型与回测分离，支持双模型策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s train --model-type mlp --experiment workflow
  %(prog)s train --model-type gbdt --experiment workflow_mlp --sell-same-label
  %(prog)s backtest --experiment workflow
  %(prog)s backtest --experiment workflow --start-time 2024-01-01 --end-time 2025-08-01
  %(prog)s plot-loss --model-type sell --experiment workflow_sell
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # train 子命令
    train_parser = subparsers.add_parser("train", help="训练买入和卖出模型")
    train_parser.add_argument(
        "--model-type", "-m",
        choices=["gbdt", "mlp", "mlp_deep"],
        default="gbdt",
        help="模型类型 (默认: gbdt)",
    )
    train_parser.add_argument(
        "--experiment", "-e",
        default="workflow",
        help="实验名称，对应 {name}_buy 和 {name}_sell (默认: workflow)",
    )
    train_parser.add_argument(
        "--sell-same-label",
        action="store_true",
        help="卖出模型使用与买入模型相同的 label（用于检测流程问题）",
    )

    # backtest 子命令
    backtest_parser = subparsers.add_parser("backtest", help="运行回测（使用已保存的预测）")
    backtest_parser.add_argument(
        "--experiment", "-e",
        default="workflow",
        help="实验名称 (默认: workflow)",
    )
    backtest_parser.add_argument(
        "--start-time",
        default=None,
        help=f"回测开始时间 (默认: {TEST_START_TIME})",
    )
    backtest_parser.add_argument(
        "--end-time",
        default=None,
        help=f"回测结束时间 (默认: {TEST_END_TIME})",
    )
    backtest_parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="持仓股票数量",
    )
    backtest_parser.add_argument(
        "--n-drop",
        type=int,
        default=None,
        help="每天最多替换的股票数量",
    )
    backtest_parser.add_argument(
        "--hold-thresh",
        type=int,
        default=None,
        help="最小持有天数",
    )
    backtest_parser.add_argument(
        "--single-model",
        action="store_true",
        help="使用 v2 单实验模式：买入和卖出共用同一实验（适用于 workflow_by_code_v2 的训练结果）",
    )
    backtest_parser.add_argument(
        "--recorder-id",
        default=None,
        help="Recorder ID，不指定则使用最新的",
    )

    # backtest_new_period 子命令
    backtest_new_parser = subparsers.add_parser("backtest_new_period", help="为新时间段生成预测并回测")
    backtest_new_parser.add_argument(
        "--start-time",
        default=TEST_START_TIME,
        help="测试开始时间",
    )
    backtest_new_parser.add_argument(
        "--end-time",
        default=TEST_END_TIME,
        help="测试结束时间",
    )
    backtest_new_parser.add_argument(
        "--source-experiment",
        default="workflow_buy",
        help="源实验名称，用于加载模型生成预测 (默认: workflow_buy)",
    )
    backtest_new_parser.add_argument(
        "--experiment",
        default="workflow_new_period",
        help="新实验名称 (默认: workflow_new_period)",
    )

    # plot_loss 子命令
    plot_parser = subparsers.add_parser("plot-loss", aliases=["plot_loss", "view_loss"], help="查看训练 loss 曲线")
    plot_parser.add_argument(
        "--model-type", "-m",
        choices=["buy", "sell"],
        default="sell",
        help="模型类型 (默认: sell)",
    )
    plot_parser.add_argument(
        "--recorder-id",
        default=None,
        help="Recorder ID，不指定则使用最新的",
    )
    plot_parser.add_argument(
        "--experiment", "-e",
        default=None,
        help="实验名称，不指定则根据 model-type 使用 workflow_buy 或 workflow_sell",
    )
    plot_parser.add_argument(
        "--save-path",
        default=None,
        help="保存图片路径",
    )

    args = parser.parse_args()

    if args.command == "train":
        if args.sell_same_label:
            print("【检测模式】卖出模型将使用与买入模型相同的 label")
        print(f"实验名称: {args.experiment} (买入: {args.experiment}_buy, 卖出: {args.experiment}_sell)")
        train_model(
            model_type=args.model_type,
            experiment_name=args.experiment,
            sell_label_same_as_buy=args.sell_same_label,
        )

    elif args.command == "backtest":
        if getattr(args, "single_model", False):
            print(f"回测使用实验（单模型/v2 模式）: {args.experiment}")
            run_backtest_only(
                recorder_id=getattr(args, "recorder_id", None),
                experiment_name=args.experiment,
                buy_experiment_name=args.experiment,
                sell_experiment_name=args.experiment,
                use_dual_model=False,
                start_time=args.start_time,
                end_time=args.end_time,
                topk=args.topk,
                n_drop=args.n_drop,
                hold_thresh=args.hold_thresh,
            )
        else:
            print(f"回测使用实验: {args.experiment} (买入: {args.experiment}_buy, 卖出: {args.experiment}_sell)")
            run_backtest_only(
                recorder_id=getattr(args, "recorder_id", None),
                experiment_name=args.experiment,
                buy_experiment_name=f"{args.experiment}_buy",
                sell_experiment_name=f"{args.experiment}_sell",
                use_dual_model=True,
                start_time=args.start_time,
                end_time=args.end_time,
                topk=args.topk,
                n_drop=args.n_drop,
                hold_thresh=args.hold_thresh,
            )

    elif args.command == "backtest_new_period":
        new_recorder_id = generate_predictions_for_new_period(
            experiment_name=args.source_experiment,
            test_start_time=args.start_time,
            test_end_time=args.end_time,
        )
        run_backtest_only(
            recorder_id=new_recorder_id,
            experiment_name=args.experiment,
            start_time=args.start_time,
            end_time=args.end_time,
        )

    elif args.command in ("plot-loss", "plot_loss", "view_loss"):
        exp_name = args.experiment or f"workflow_{args.model_type}"
        save_path = args.save_path or f"loss_curve_{args.model_type}_{(args.recorder_id[:8] if args.recorder_id else 'latest')}.png"
        print(f"\n查看{args.model_type}模型的 loss 曲线...")
        loss_df = plot_loss_curve(
            recorder_id=args.recorder_id,
            experiment_name=exp_name,
            save_path=save_path,
            show_plot=False,
        )
        if loss_df is not None:
            print(f"\nLoss 数据已加载，共 {len(loss_df)} 个训练步骤")
            print(f"Loss 曲线图片已保存到: {save_path}")

    else:
        # 无子命令时：默认训练 + 回测
        print("=" * 80)
        print("默认模式：训练模型 + 运行回测")
        print("=" * 80)
        print("\n提示：")
        print("  - 仅训练: python workflow_by_code_v2.py train")
        print("  - 仅回测: python workflow_by_code_v2.py backtest")
        print("  - 查看帮助: python workflow_by_code_v2.py -h")
        print("=" * 80)
        recorder_id = train_model(model_type="gbdt")
        run_backtest_only(recorder_id=recorder_id)


if __name__ == "__main__":
    main()
