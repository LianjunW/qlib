#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib Workflow Example with Custom Strategy

This script demonstrates two approaches for running backtests:
1. Using the native TopkDropoutStrategy with PortAnaRecord (recommended for compatibility)
2. Using a custom ThresholdTopkDropoutStrategy with direct backtest API

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this,
which nearly does the same thing as `qrun XXX.yaml`
"""
from pprint import pprint

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_MARKET, get_gbdt_task, GBDT_MODEL
from qlib.backtest import backtest as run_backtest
from qlib.contrib.evaluate import risk_analysis, indicator_analysis

# Import custom strategy and utils
from strategies import ThresholdTopkDropoutStrategy
from utils import print_stock_analysis


# ============================================================
# 模型配置
# ============================================================

# GBDT 模型配置（LightGBM）
GBDT_MODEL_CONFIG = GBDT_MODEL

# MLP 模型配置（深度神经网络）
# Alpha158 特征数为 158
# 注意：如果出现 NaN，可能需要进一步降低学习率或检查数据
MLP_MODEL_CONFIG = {
    "class": "DNNModelPytorch",
    "module_path": "qlib.contrib.model.pytorch_nn",
    "kwargs": {
        "lr": 0.0001,           # 降低学习率，避免梯度爆炸
        "max_steps": 8000,
        "batch_size": 2048,     # 减小 batch_size，提高稳定性
        "early_stop_rounds": 100,
        "eval_steps": 20,
        "optimizer": "adam",
        "loss": "mse",
        "GPU": -1,              # 使用 CPU（-1）或 GPU（0）
        "seed": 42,
        "weight_decay": 0.001,  # 增加正则化
        "pt_model_kwargs": {
            "input_dim": 158,   # Alpha158 特征数
            "layers": (256, 128),  # 简化网络结构，减少层数
            "act": "LeakyReLU",
        },
    },
}

# 更深的 MLP 模型配置（稳定版）
MLP_DEEP_MODEL_CONFIG = {
    "class": "DNNModelPytorch",
    "module_path": "qlib.contrib.model.pytorch_nn",
    "kwargs": {
        "lr": 0.00005,          # 更低的学习率
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
            "layers": (512, 256, 128),  # 三层
            "act": "LeakyReLU",
        },
    },
}


def run_native_strategy_backtest(recorder, benchmark):
    """
    方式1: 使用原生 TopkDropoutStrategy + PortAnaRecord
    
    推荐方式，完全兼容 Qlib 官方的回测和分析流程。
    
    Parameters
    ----------
    recorder : Recorder
        Qlib recorder 实例
    benchmark : str
        基准指数代码
    """
    print("\n" + "="*80)
    print("方式1: 原生 TopkDropoutStrategy + PortAnaRecord")
    print("="*80)
    
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
                "signal": "<PRED>",  # 使用占位符，会被替换为预测结果
                # "topk": 5,
                # "n_drop": 1,         # 减少每天替换数量（原来是5）→ 降低交易频次
                # "hold_thresh": 1,    # 最少持有3天（原来是1天）→ 降低交易频次
                "topk": 20,
                "n_drop": 2,         # 减少每天替换数量（原来是5）→ 降低交易频次
                "hold_thresh": 1,    # 最少持有3天（原来是1天）→ 降低交易频次
            },
        },
        "backtest": {
            "start_time": "2024-01-01",
            "end_time": "2025-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # backtest using PortAnaRecord
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()


def run_custom_strategy_backtest(recorder, model, dataset, benchmark):
    """
    方式2: 使用自定义 ThresholdTopkDropoutStrategy + 直接回测
    
    支持基于评分阈值的交易过滤，输出格式对齐官方 PortAnaRecord。
    
    Parameters
    ----------
    recorder : Recorder
        Qlib recorder 实例
    model : Model
        训练好的模型
    dataset : Dataset
        数据集
    benchmark : str
        基准指数代码
    """
    print("\n" + "="*80)
    print("方式2: 自定义 ThresholdTopkDropoutStrategy 回测")
    print("="*80)
    
    # 创建基于阈值的策略实例
    threshold_strategy = ThresholdTopkDropoutStrategy(
        signal=(model, dataset),
        topk=50,
        n_drop=3,              # 减少每天替换数量
        buy_thresh=0.01,       # 只买入评分>0.01的股票
        sell_thresh=-0.01,     # 只卖出评分<-0.01的股票
        score_diff_thresh=0.005,  # 新股票评分需比持仓最低评分高0.005才替换
        hold_thresh=3,         # 最少持有3天
    )
    
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    
    # 直接运行回测
    portfolio_metric_dict, indicator_dict = run_backtest(
        start_time="2024-01-01",
        end_time="2025-08-01",
        strategy=threshold_strategy,
        executor=executor_config,
        benchmark=benchmark,
        account=100000000,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )
    
    # 保存和分析结果
    _save_and_analyze_results(recorder, portfolio_metric_dict, indicator_dict)
    
    print("\n" + "="*80)
    print("自定义策略回测完成，结果已保存到 threshold_portfolio_analysis/ 目录")
    print("="*80)


def _save_and_analyze_results(recorder, portfolio_metric_dict, indicator_dict):
    """
    保存回测结果并进行风险分析（对齐 PortAnaRecord 的格式）
    
    Parameters
    ----------
    recorder : Recorder
        Qlib recorder 实例
    portfolio_metric_dict : dict
        回测结果字典
    indicator_dict : dict
        指标字典
    """
    # 保存回测结果
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
    
    # 风险分析
    for _freq, (report_normal, _) in portfolio_metric_dict.items():
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq=_freq
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_freq
        )
        
        analysis_df = pd.concat(analysis)
        
        # 保存风险分析结果
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"port_analysis_{_freq}.pkl": analysis_df}
        )
        
        # 记录指标到 recorder
        analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
        recorder.log_metrics(**{f"threshold.{_freq}.{k}": v for k, v in analysis_dict.items()})
        
        # 打印结果
        pprint(f"The following are analysis results of benchmark return({_freq}).")
        pprint(risk_analysis(report_normal["bench"], freq=_freq))
        pprint(f"The following are analysis results of the excess return without cost({_freq}).")
        pprint(analysis["excess_return_without_cost"])
        pprint(f"The following are analysis results of the excess return with cost({_freq}).")
        pprint(analysis["excess_return_with_cost"])
    
    # 指标分析
    for _freq, indicators_normal in indicator_dict.items():
        indicators_df = indicators_normal[0]
        analysis_df = indicator_analysis(indicators_df)
        
        # 保存指标分析结果
        recorder.save_objects(
            artifact_path="threshold_portfolio_analysis",
            **{f"indicator_analysis_{_freq}.pkl": analysis_df}
        )
        
        # 记录指标到 recorder
        analysis_dict = analysis_df["value"].to_dict()
        recorder.log_metrics(**{f"threshold.{_freq}.{k}": v for k, v in analysis_dict.items()})
        
        # 打印结果
        pprint(f"The following are analysis results of indicators({_freq}).")
        pprint(analysis_df)


def get_extended_dataset_config(
    train_periods=None,
    valid=("2022-01-01", "2023-12-31"),
    test=("2024-01-01", "2025-08-01"),
    instruments=CSI300_MARKET,
    end_time="2025-08-01",
):
    """
    创建扩展的数据集配置，支持多时间段训练数据
    
    Parameters
    ----------
    train_periods : list of tuple, optional
        训练数据的时间段列表，例如:
        [("2010-01-01", "2014-12-31"), ("2016-01-01", "2019-12-31"), ("2020-06-01", "2021-12-31")]
        如果为 None，使用默认的扩展时间段
    valid : tuple
        验证集时间段
    test : tuple
        测试集时间段
    instruments : str
        股票池
    end_time : str
        数据结束时间
        
    Returns
    -------
    dict
        数据集配置
    """
    if train_periods is None:
        # 默认使用多个时间段的训练数据，覆盖不同市场环境
        # 包含牛市、熊市、震荡市等不同市场状态
        train_periods = [
            ("2010-01-01", "2014-12-31"),  # 2010-2014: 震荡+牛市
            ("2016-01-01", "2019-12-31"),  # 2016-2019: 熊市+震荡
            ("2020-06-01", "2021-12-31"),  # 2020-2021: 结构性牛市
        ]
    
    # 计算最早的开始时间
    start_time = min(period[0] for period in train_periods)
    
    # 合并训练时间段为单一范围（Qlib 标准方式）
    # 注意：Qlib 的 DatasetH 只支持单一连续时间段
    # 如果需要多时间段，需要使用 train 的最早开始到最晚结束
    train_start = min(period[0] for period in train_periods)
    train_end = max(period[1] for period in train_periods)
    
    # 数据预处理器配置（关键：处理 NaN 和归一化）
    infer_processors = [
        {"class": "ProcessInf", "kwargs": {}},  # 处理 Inf 值
        {
            "class": "ZScoreNorm",
            "kwargs": {
                "fit_start_time": train_start,
                "fit_end_time": train_end,
            },
        },  # Z-Score 归一化
        {"class": "Fillna", "kwargs": {}},  # 填充 NaN
    ]
    
    learn_processors = [
        {"class": "DropnaLabel"},  # 删除标签为 NaN 的样本
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},  # 标签归一化
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
                    "infer_processors": infer_processors,  # 添加推理时的预处理
                    "learn_processors": learn_processors,  # 添加训练时的预处理
                },
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": valid,
                "test": test,
            },
        },
    }


def main():
    """主函数"""
    # ============================================================
    # 初始化 Qlib
    # ============================================================
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ============================================================
    # 配置任务 - 使用扩展的多时间段训练数据
    # ============================================================
    # 训练数据配置说明：
    # - 使用更长的历史数据（2010-2021）来训练模型
    # - 包含多种市场环境：牛市、熊市、震荡市
    # - 验证集使用 2022-2023 年数据
    # - 测试集使用 2024-2025 年数据进行回测
    # ============================================================
    
    # 方案1：使用扩展的连续时间段（推荐）
    extended_dataset_config = get_extended_dataset_config(
        train_periods=[
            ("2010-01-01", "2014-12-31"),  # 震荡+牛市期
            ("2016-01-01", "2019-12-31"),  # 熊市+震荡期
            ("2020-06-01", "2021-12-31"),  # 结构性牛市
        ],
        valid=("2022-01-01", "2023-12-31"),
        test=("2024-01-01", "2025-08-01"),
        instruments=CSI300_MARKET,
        end_time="2025-08-01",
    )
    
    # ============================================================
    # 选择模型类型
    # ============================================================
    # 可选模型：
    #   - "gbdt": LightGBM 梯度提升树（快速，适合小数据）
    #   - "mlp": MLP 神经网络（需要更多数据和训练时间）
    #   - "mlp_deep": 更深的 MLP 神经网络
    # ============================================================
    MODEL_TYPE = "mlp"  # 修改这里切换模型
    
    if MODEL_TYPE == "gbdt":
        model_config = GBDT_MODEL_CONFIG
        print("使用模型: LightGBM (GBDT)")
    elif MODEL_TYPE == "mlp":
        model_config = MLP_MODEL_CONFIG
        print("使用模型: MLP 神经网络 (3层: 512-256-128)")
    elif MODEL_TYPE == "mlp_deep":
        model_config = MLP_DEEP_MODEL_CONFIG
        print("使用模型: 深层 MLP 神经网络 (4层: 1024-512-256-128)")
    else:
        raise ValueError(f"未知的模型类型: {MODEL_TYPE}")
    
    custom_task = {
        "model": model_config,
        "dataset": extended_dataset_config,
    }
    
    model = init_instance_by_config(custom_task["model"])
    dataset = init_instance_by_config(custom_task["dataset"])

    # ============================================================
    # 数据集信息
    # ============================================================
    print("\n" + "="*80)
    print("数据集配置信息")
    print("="*80)
    segments = custom_task["dataset"]["kwargs"]["segments"]
    print(f"训练集: {segments['train'][0]} ~ {segments['train'][1]}")
    print(f"验证集: {segments['valid'][0]} ~ {segments['valid'][1]}")
    print(f"测试集: {segments['test'][0]} ~ {segments['test'][1]}")
    
    # 预览数据集
    example_df = dataset.prepare("train")
    print(f"\n训练数据集大小: {len(example_df)} 条记录")
    print("训练数据集预览:")
    print(example_df.head())
    
    # ============================================================
    # 交易频次控制参数说明
    # ============================================================
    # 方式1: 使用原始 TopkDropoutStrategy，通过调整参数降低频次
    #   - n_drop: 每天最多替换的股票数量（减小此值降低频次）
    #   - hold_thresh: 最小持有天数（增大此值降低频次）
    #
    # 方式2: 使用自定义 ThresholdTopkDropoutStrategy，基于阈值控制
    #   - buy_thresh: 买入阈值，评分需要高于此值才买入（如0.02）
    #   - sell_thresh: 卖出阈值，评分需要低于此值才卖出（如-0.02）
    #   - score_diff_thresh: 评分差异阈值，新股票评分需高出此值才替换
    # ============================================================

    # ============================================================
    # 开始实验
    # ============================================================
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(custom_task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # 获取 recorder
        recorder = R.get_recorder()
        
        # 生成预测信号
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 方式1: 原生策略回测
        run_native_strategy_backtest(recorder, CSI300_BENCH)

        # 方式2: 自定义策略回测
        run_custom_strategy_backtest(recorder, model, dataset, CSI300_BENCH)

        # ============================================================
        # 单股票分析示例
        # ============================================================
        # target_stock = "SH600000"
        # print_stock_analysis(recorder, target_stock, buy_threshold=0.02, sell_threshold=-0.02)

        # # ============================================================
        # # 使用说明
        # # ============================================================
        # print("\n" + "="*80)
        # print("使用说明")
        # print("="*80)
        # print("1. 使用 get_stock_score_curve(recorder, 'STOCK_CODE') 获取单个股票评分")
        # print("2. 预测数据保存在 recorder 的 'pred.pkl' 文件中")
        # print("3. 回测结果保存在 recorder 的 'portfolio_analysis/' 目录中")
        # print("4. 自定义策略结果保存在 'threshold_portfolio_analysis/' 目录中")
        # print("5. 要进行更详细的分析，可以导入 utils 模块中的函数")


if __name__ == "__main__":
    main()
