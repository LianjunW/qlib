#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
单股票评分分析工具

提供从 recorder 中获取和分析单个股票评分曲线的功能。
"""
import pandas as pd
from typing import Optional, Dict, Any


def get_stock_score_curve(recorder, stock_code: str) -> Optional[pd.Series]:
    """
    从 recorder 中获取指定股票的评分曲线
    
    Parameters
    ----------
    recorder : qlib.workflow.recorder.Recorder
        实验记录器
    stock_code : str
        股票代码，例如 'SH600000'
        
    Returns
    -------
    pd.Series or None
        指定股票的时间序列评分数据，如果股票不存在则返回 None
        
    Examples
    --------
    >>> from qlib.workflow import R
    >>> recorder = R.get_recorder()
    >>> scores = get_stock_score_curve(recorder, "SH600000")
    >>> if scores is not None:
    ...     print(f"平均评分: {scores.mean():.4f}")
    """
    # 从 SignalRecord 中加载预测结果
    pred_df = recorder.load_object("pred.pkl")
    
    # 筛选指定股票的数据
    if stock_code in pred_df.index.get_level_values('instrument'):
        stock_scores = pred_df.loc[pred_df.index.get_level_values('instrument') == stock_code]
        return stock_scores['score'] if 'score' in stock_scores.columns else stock_scores.iloc[:, 0]
    else:
        print(f"股票代码 {stock_code} 不在预测结果中")
        available_stocks = pred_df.index.get_level_values('instrument').unique()[:10]
        print(f"可用的股票代码示例: {list(available_stocks)}")
        return None


def analyze_stock_score(scores: pd.Series) -> Dict[str, Any]:
    """
    分析股票评分的统计信息
    
    Parameters
    ----------
    scores : pd.Series
        股票评分时间序列
        
    Returns
    -------
    dict
        包含统计信息的字典
        
    Examples
    --------
    >>> stats = analyze_stock_score(scores)
    >>> print(f"平均评分: {stats['mean']:.4f}")
    >>> print(f"最大评分: {stats['max']:.4f}")
    """
    return {
        'count': len(scores),
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'median': scores.median(),
        'q25': scores.quantile(0.25),
        'q75': scores.quantile(0.75),
    }


def simple_threshold_strategy(
    recorder,
    stock_code: str,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02
) -> Optional[Dict[str, Any]]:
    """
    基于评分阈值的简单策略测试
    
    分析单个股票在给定阈值下的买入/卖出/持有信号分布。
    
    Parameters
    ----------
    recorder : qlib.workflow.recorder.Recorder
        实验记录器
    stock_code : str
        股票代码
    buy_threshold : float
        买入阈值，评分高于此值时生成买入信号
    sell_threshold : float
        卖出阈值，评分低于此值时生成卖出信号
        
    Returns
    -------
    dict or None
        包含评分和信号的字典，如果股票不存在则返回 None
        
    Examples
    --------
    >>> result = simple_threshold_strategy(recorder, "SH600000", 0.02, -0.02)
    >>> if result:
    ...     print(f"买入信号天数: {result['buy_signals'].sum()}")
    """
    # 获取评分数据
    pred_df = recorder.load_object("pred.pkl")
    if stock_code not in pred_df.index.get_level_values('instrument'):
        print(f"Stock {stock_code} not found in predictions")
        return None
        
    stock_scores = pred_df.loc[pred_df.index.get_level_values('instrument') == stock_code]
    scores = stock_scores['score'] if 'score' in stock_scores.columns else stock_scores.iloc[:, 0]
    
    # 生成交易信号
    buy_signals = scores > buy_threshold
    sell_signals = scores < sell_threshold
    hold_signals = (scores >= sell_threshold) & (scores <= buy_threshold)
    
    return {
        'scores': scores,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
    }


def print_stock_analysis(
    recorder,
    stock_code: str,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02
) -> None:
    """
    打印股票评分分析报告
    
    Parameters
    ----------
    recorder : qlib.workflow.recorder.Recorder
        实验记录器
    stock_code : str
        股票代码
    buy_threshold : float
        买入阈值
    sell_threshold : float
        卖出阈值
        
    Examples
    --------
    >>> print_stock_analysis(recorder, "SH600000", 0.02, -0.02)
    """
    print(f"\n{'='*60}")
    print(f"股票 {stock_code} 评分分析报告")
    print(f"{'='*60}")
    
    # 获取评分数据
    scores = get_stock_score_curve(recorder, stock_code)
    if scores is None:
        return
    
    print(f"成功获取 {stock_code} 的评分数据，共 {len(scores)} 个数据点")
    
    # 打印统计信息
    stats = analyze_stock_score(scores)
    print(f"\n【评分统计信息】")
    print(f"  评分数量: {stats['count']}")
    print(f"  平均评分: {stats['mean']:.4f}")
    print(f"  标准差:   {stats['std']:.4f}")
    print(f"  最高评分: {stats['max']:.4f}")
    print(f"  最低评分: {stats['min']:.4f}")
    print(f"  中位数:   {stats['median']:.4f}")
    print(f"  25%分位:  {stats['q25']:.4f}")
    print(f"  75%分位:  {stats['q75']:.4f}")
    
    # 显示评分时间序列示例
    print(f"\n【评分时间序列示例】")
    print("前5个数据点:")
    print(scores.head())
    print("\n后5个数据点:")
    print(scores.tail())
    
    # 阈值策略分析
    result = simple_threshold_strategy(recorder, stock_code, buy_threshold, sell_threshold)
    if result is None:
        return
    
    print(f"\n【阈值策略信号统计】")
    print(f"  买入阈值: {buy_threshold}")
    print(f"  卖出阈值: {sell_threshold}")
    print(f"  买入信号 (评分 > {buy_threshold}): {result['buy_signals'].sum()} 天")
    print(f"  卖出信号 (评分 < {sell_threshold}): {result['sell_signals'].sum()} 天")
    print(f"  持有信号: {result['hold_signals'].sum()} 天")
    print(f"  信号覆盖率: 买入 {result['buy_signals'].mean():.1%}, "
          f"卖出 {result['sell_signals'].mean():.1%}, "
          f"持有 {result['hold_signals'].mean():.1%}")
    
    # 显示信号示例
    if result['buy_signals'].any():
        print(f"\n【最近的买入信号示例】")
        recent_buys = result['scores'][result['buy_signals']].tail(3)
        for (instrument, date), score in recent_buys.items():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
            print(f"  {date_str}: 评分 {score:.4f}")
    
    if result['sell_signals'].any():
        print(f"\n【最近的卖出信号示例】")
        recent_sells = result['scores'][result['sell_signals']].tail(3)
        for (instrument, date), score in recent_sells.items():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
            print(f"  {date_str}: 评分 {score:.4f}")
    
    print(f"\n{'='*60}")

