#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
基于阈值的 TopkDropout 策略

通过设置买入/卖出阈值和评分差异阈值来降低交易频次，避免频繁调仓。
"""
import copy
import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position


class ThresholdTopkDropoutStrategy(TopkDropoutStrategy):
    """
    基于阈值的TopkDropout策略，降低交易频次
    
    只有当股票评分变化超过阈值时才进行交易，避免频繁调仓。
    
    Parameters
    ----------
    topk : int
        持仓股票数量
    n_drop : int
        每个交易日最多替换的股票数量
    buy_thresh : float
        买入阈值，评分需要高于此值才考虑买入（默认0.0）
    sell_thresh : float
        卖出阈值，评分需要低于此值才考虑卖出（默认为-buy_thresh）
    score_diff_thresh : float
        评分变化阈值，只有新股票评分比持仓股票高出此值才替换（默认0.0）
    hold_thresh : int
        最小持有天数
        
    Examples
    --------
    >>> strategy = ThresholdTopkDropoutStrategy(
    ...     signal=(model, dataset),
    ...     topk=50,
    ...     n_drop=3,
    ...     buy_thresh=0.01,       # 只买入评分>0.01的股票
    ...     sell_thresh=-0.01,     # 只卖出评分<-0.01的股票
    ...     score_diff_thresh=0.005,  # 新股票评分需比持仓最低评分高0.005才替换
    ...     hold_thresh=3,         # 最少持有3天
    ... )
    """
    
    def __init__(
        self,
        *,
        topk,
        n_drop,
        buy_thresh=0.0,
        sell_thresh=None,
        score_diff_thresh=0.0,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        super().__init__(
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
            hold_thresh=hold_thresh,
            only_tradable=only_tradable,
            forbid_all_trade_at_limit=forbid_all_trade_at_limit,
            **kwargs,
        )
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh if sell_thresh is not None else -buy_thresh
        self.score_diff_thresh = score_diff_thresh
        self._last_pred_score = None

    def generate_trade_decision(self, execute_result=None):
        # 获取当前交易步骤
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        
        # 应用买入阈值过滤：只考虑评分高于阈值的股票
        if self.buy_thresh != 0:
            pred_score_filtered = pred_score[pred_score > self.buy_thresh]
        else:
            pred_score_filtered = pred_score
        
        if self.only_tradable:
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]
        else:
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
        
        # 使用过滤后的评分
        last = pred_score_filtered.reindex(current_stock_list).dropna().sort_values(ascending=False).index
        
        # 应用卖出阈值：只有评分低于卖出阈值的股票才考虑卖出
        stocks_below_sell_thresh = []
        if self.sell_thresh is not None:
            for stock in current_stock_list:
                if stock in pred_score.index and pred_score[stock] < self.sell_thresh:
                    stocks_below_sell_thresh.append(stock)
        
        # 获取候选买入股票（评分高于阈值且不在当前持仓中）
        if self.method_buy == "top":
            candidates = pred_score_filtered[~pred_score_filtered.index.isin(last)].sort_values(ascending=False)
            
            # 应用评分差异阈值：只有评分显著高于持仓最低评分的股票才买入
            if self.score_diff_thresh > 0 and len(last) > 0:
                min_holding_score = pred_score_filtered.reindex(last).min()
                candidates = candidates[candidates > min_holding_score + self.score_diff_thresh]
            
            today = get_first_n(candidates.index, self.n_drop + self.topk - len(last))
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score_filtered.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        comb = pred_score_filtered.reindex(last.union(pd.Index(today))).dropna().sort_values(ascending=False).index

        # 确定要卖出的股票
        if self.method_sell == "bottom":
            # 优先卖出评分低于阈值的股票
            sell_candidates = last[last.isin(get_last_n(comb, self.n_drop))]
            # 如果有评分低于卖出阈值的股票，优先卖出
            if stocks_below_sell_thresh:
                sell = last[last.isin(stocks_below_sell_thresh + list(sell_candidates))][:self.n_drop]
            else:
                sell = sell_candidates
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        buy = today[: len(sell) + self.topk - len(last)]
        
        # 执行卖出
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost

        # 执行买入
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
        
        self._last_pred_score = pred_score.copy()
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

