#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib Workflow Example with Custom Strategy

将模型训练和策略回测分离：
1. train_model() - 训练模型并保存预测结果
2. run_backtest_only() - 仅运行回测（使用已保存的预测结果）

这样修改策略参数时不需要重新训练模型。
"""
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

# Import custom strategy and utils
from strategies import ThresholdTopkDropoutStrategy
from utils import print_stock_analysis


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

BACKTEST_CONFIG = {
    "start_time": "2024-01-01",
    "end_time": "2025-08-01",
    "account": 100000000,
    "benchmark": CSI300_BENCH,
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
}


# ============================================================
# 数据集配置
# ============================================================

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

def train_model(model_type="gbdt", experiment_name="workflow"):
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
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
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
    extended_dataset_config = get_extended_dataset_config()
    
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
# 回测函数（不需要重新训练）
# ============================================================

def run_backtest_only(
    recorder_id=None,
    experiment_name="workflow",
    topk=20,
    n_drop=2,
    hold_thresh=10,
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
    
    # 使用传入的参数或默认配置
    _topk = topk if topk is not None else STRATEGY_CONFIG["topk"]
    _n_drop = n_drop if n_drop is not None else STRATEGY_CONFIG["n_drop"]
    _hold_thresh = hold_thresh if hold_thresh is not None else STRATEGY_CONFIG["hold_thresh"]
    
    print("\n" + "="*80)
    print("回测策略配置")
    print("="*80)
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
                    start_time = rec.info.get("start_time", 0) if hasattr(rec, 'info') else 0
                    recorder_list.append((rid, start_time))
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
                    start_time = rec.info.get("start_time", "N/A") if hasattr(rec, 'info') else "N/A"
                    print(f"  [{i}] {rid} (创建时间: {start_time})")
                except:
                    print(f"  [{i}] {rid}")
        else:
            for i, rec in enumerate(recorders):
                rid = rec.id if hasattr(rec, 'id') else rec
                try:
                    start_time = rec.info.get("start_time", "N/A") if hasattr(rec, 'info') else "N/A"
                    print(f"  [{i}] {rid} (创建时间: {start_time})")
                except:
                    print(f"  [{i}] {rid}")
    
    recorder = exp.get_recorder(recorder_id=recorder_id)
    
    # 验证预测结果存在
    print("\n验证预测结果...")
    try:
        pred = recorder.load_object("pred.pkl")
        print(f"预测数据形状: {pred.shape}")
        dt_min = pred.index.get_level_values('datetime').min()
        dt_max = pred.index.get_level_values('datetime').max()
        print(f"预测数据时间范围: {dt_min} ~ {dt_max}")
        print(f"回测时间范围: {BACKTEST_CONFIG['start_time']} ~ {BACKTEST_CONFIG['end_time']}")
        
        # 检查时间范围是否匹配
        import pandas as pd
        bt_start = pd.Timestamp(BACKTEST_CONFIG['start_time'])
        bt_end = pd.Timestamp(BACKTEST_CONFIG['end_time'])
        if dt_max < bt_start:
            print(f"\n" + "!"*80)
            print(f"错误：预测数据时间范围 ({dt_max}) 早于回测开始时间 ({bt_start})！")
            print("这会导致回测无法执行任何交易。")
            print(f"\n请重新训练模型：python workflow_by_code_v2.py train gbdt")
            print("!"*80)
            raise ValueError(f"预测数据时间范围不匹配，请重新训练模型")
        
        # 显示预测数据样本
        print(f"\n预测数据前5行:")
        print(pred.head())
        print(f"\n预测数据后5行:")
        print(pred.tail())
    except Exception as e:
        raise ValueError(f"无法加载预测结果: {e}，请先运行 train_model()")
    
    # 构建回测配置
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
        "backtest": BACKTEST_CONFIG,
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
    主函数 - 演示训练和回测分离
    
    使用方式：
    1. 首次运行或需要重新训练时：
       python workflow_by_code_v2.py train
       
    2. 仅修改策略参数运行回测：
       python workflow_by_code_v2.py backtest
       
    3. 或者在 Python 中：
       from workflow_by_code_v2 import train_model, run_backtest_only
       
       # 训练（只需运行一次）
       recorder_id = train_model(model_type="gbdt")
       
       # 回测（可以多次运行，修改参数）
       run_backtest_only(topk=20, n_drop=2, hold_thresh=1)
       run_backtest_only(topk=50, n_drop=5, hold_thresh=3)
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
            run_backtest_only()
            
        else:
            print(f"未知命令: {command}")
            print("使用方式:")
            print("  python workflow_by_code_v2.py train [gbdt|mlp|mlp_deep]")
            print("  python workflow_by_code_v2.py backtest")
    else:
        # 默认：训练 + 回测
        print("="*80)
        print("默认模式：训练模型 + 运行回测")
        print("="*80)
        print("\n提示：")
        print("  - 仅训练: python workflow_by_code_v2.py train")
        print("  - 仅回测: python workflow_by_code_v2.py backtest")
        print("="*80)
        
        recorder_id = train_model(model_type="gbdt")
        run_backtest_only(recorder_id=recorder_id)


if __name__ == "__main__":
    main()
