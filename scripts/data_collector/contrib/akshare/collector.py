# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import random
import datetime
import importlib
import yaml
from abc import ABC
from pathlib import Path
from typing import Iterable, List, Optional
from functools import wraps

import fire
import numpy as np
import pandas as pd
from loguru import logger
import requests
from fake_useragent import UserAgent

import qlib
import multiprocessing
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
# 添加 scripts 目录到路径，以便导入 data_collector 模块
_scripts_dir = CUR_DIR.parent.parent.parent
sys.path.insert(0, str(_scripts_dir))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
)

# 导入 dump_bin（需要从 scripts 目录导入）
_scripts_path = Path(_scripts_dir)
if (_scripts_path / "dump_bin.py").exists():
    sys.path.insert(0, str(_scripts_path))
    try:
        from dump_bin import DumpDataUpdate
    except ImportError:
        DumpDataUpdate = None
        logger.warning("Failed to import DumpDataUpdate from dump_bin")
else:
    # 如果找不到，尝试从项目根目录的 scripts 查找
    _project_root = CUR_DIR.parent.parent.parent.parent
    _scripts_in_root = _project_root / "scripts" / "dump_bin.py"
    if _scripts_in_root.exists():
        sys.path.insert(0, str(_project_root / "scripts"))
        try:
            from dump_bin import DumpDataUpdate
        except ImportError:
            DumpDataUpdate = None
            logger.warning("Failed to import DumpDataUpdate from dump_bin")
    else:
        DumpDataUpdate = None
        logger.warning("dump_bin.py not found, update_data_to_bin will not work")

try:
    import akshare as ak
except ImportError:
    logger.error("akshare is not installed. Please install it: pip install akshare")
    raise


def exponential_backoff_retry(max_retries=5, base_delay=1.0, max_delay=60.0):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 403:
                        logger.warning(f"403 Forbidden detected, triggering long rest mode")
                        time.sleep(random.uniform(30, 60))
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class ClashConfigReader:
    """Clash 配置读取器（只读，不修改配置文件）"""
    
    @staticmethod
    def read_clash_config(config_path: str) -> dict:
        """读取 Clash 配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            logger.warning(f"Failed to read Clash config: {e}")
            return {}
    
    @staticmethod
    def get_clash_local_proxy(config_path: Optional[str] = None) -> Optional[str]:
        """获取 Clash 本地代理地址"""
        # 默认路径
        default_paths = [
            Path.home() / ".config" / "mihomo" / "config.yaml",
            Path.home() / ".config" / "clash" / "config.yaml",
            Path("/root/.config/mihomo/config.yaml"),
            Path("/root/.config/clash/config.yaml"),
        ]
        
        config_paths = [config_path] if config_path else []
        config_paths.extend(default_paths)
        
        for path in config_paths:
            if path and Path(path).exists():
                config = ClashConfigReader.read_clash_config(str(path))
                # 优先使用 mixed-port，其次 port，最后 socks-port
                port = config.get('mixed-port') or config.get('port') or config.get('socks-port')
                if port:
                    return f"http://127.0.0.1:{port}"
        
        return None
    
    @staticmethod
    def get_proxy_count(config_path: Optional[str] = None) -> int:
        """获取 Clash 配置中的代理节点数量"""
        default_paths = [
            Path.home() / ".config" / "mihomo" / "config.yaml",
            Path.home() / ".config" / "clash" / "config.yaml",
            Path("/root/.config/mihomo/config.yaml"),
            Path("/root/.config/clash/config.yaml"),
        ]
        
        config_paths = [config_path] if config_path else []
        config_paths.extend(default_paths)
        
        for path in config_paths:
            if path and Path(path).exists():
                config = ClashConfigReader.read_clash_config(str(path))
                proxies = config.get('proxies', [])
                return len(proxies) if isinstance(proxies, list) else 0
        
        return 0


class ProxyManager:
    """增强的代理管理器，支持 Clash 和自定义代理"""
    def __init__(
        self, 
        proxy_list: Optional[List[str]] = None,
        clash_config_path: Optional[str] = None,
        use_clash: bool = True,
        proxy_rotation_strategy: str = "round_robin"  # round_robin, random, weighted
    ):
        """
        Parameters
        ----------
        proxy_list: List[str]
            自定义代理列表
        clash_config_path: str
            Clash 配置文件路径
        use_clash: bool
            是否使用 Clash 本地代理
        proxy_rotation_strategy: str
            代理轮换策略: round_robin, random, weighted
        """
        self.proxy_list = proxy_list or []
        self.current_proxy_idx = 0
        self.failed_proxies = set()
        self.proxy_stats = {}  # 代理统计信息
        self.rotation_strategy = proxy_rotation_strategy
        self.use_clash = use_clash
        
        # 尝试从 Clash 配置获取本地代理
        self.clash_proxy = None
        if use_clash:
            self.clash_proxy = ClashConfigReader.get_clash_local_proxy(clash_config_path)
            if self.clash_proxy:
                logger.info(f"Using Clash local proxy: {self.clash_proxy}")
                proxy_count = ClashConfigReader.get_proxy_count(clash_config_path)
                if proxy_count > 0:
                    logger.info(f"Clash config contains {proxy_count} proxy nodes")
            else:
                logger.warning("Clash proxy not found, will use direct connection or custom proxies")
        
        # 如果使用 Clash，优先使用 Clash 代理
        if self.clash_proxy and not self.proxy_list:
            self.proxy_list = [self.clash_proxy]
        elif self.clash_proxy:
            # Clash 代理放在最前面
            self.proxy_list.insert(0, self.clash_proxy)
    
    def get_proxy(self) -> Optional[dict]:
        """获取下一个可用代理"""
        if not self.proxy_list:
            return None
        
        available_proxies = [p for p in self.proxy_list if p not in self.failed_proxies]
        if not available_proxies:
            # 重置失败列表
            logger.warning("All proxies failed, resetting failed list")
            self.failed_proxies.clear()
            available_proxies = self.proxy_list
        
        if not available_proxies:
            return None
        
        # 根据策略选择代理
        if self.rotation_strategy == "random":
            proxy = random.choice(available_proxies)
        elif self.rotation_strategy == "weighted":
            # 根据成功率加权选择
            proxy = self._get_weighted_proxy(available_proxies)
        else:  # round_robin
            proxy = available_proxies[self.current_proxy_idx % len(available_proxies)]
            self.current_proxy_idx += 1
        
        return {"http": proxy, "https": proxy}
    
    def _get_weighted_proxy(self, available_proxies: List[str]) -> str:
        """根据权重选择代理（成功率高的优先）"""
        if not self.proxy_stats:
            return random.choice(available_proxies)
        
        weights = []
        for proxy in available_proxies:
            stats = self.proxy_stats.get(proxy, {"success": 0, "total": 0})
            if stats["total"] == 0:
                weight = 1.0
            else:
                weight = stats["success"] / stats["total"]
            weights.append(weight)
        
        # 加权随机选择
        return random.choices(available_proxies, weights=weights, k=1)[0]
    
    def mark_success(self, proxy: str):
        """标记代理成功"""
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {"success": 0, "total": 0}
        self.proxy_stats[proxy]["success"] += 1
        self.proxy_stats[proxy]["total"] += 1
    
    def mark_failed(self, proxy: str):
        """标记代理失败"""
        self.failed_proxies.add(proxy)
        if proxy not in self.proxy_stats:
            self.proxy_stats[proxy] = {"success": 0, "total": 0}
        self.proxy_stats[proxy]["total"] += 1
    
    def get_stats(self) -> dict:
        """获取代理统计信息"""
        return self.proxy_stats.copy()


class AkshareCollector(BaseCollector):
    """Akshare 数据采集器，包含防封禁策略"""
    retry = 5
    
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,  # 默认1，避免高并发被封
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        config_path: Optional[str] = None,
        batch_size: int = 50,  # 每批处理的股票数量
        batch_rest_time: int = 60,  # 每批完成后的休息时间（秒）
        clash_config_path: Optional[str] = None,  # Clash 配置文件路径
        use_clash: bool = True,  # 是否使用 Clash 代理
        proxy_rotation_strategy: str = "round_robin",  # 代理轮换策略
    ):
        """
        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 1 (recommended for anti-ban)
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        config_path: str
            path to akshare_config.yaml
        batch_size: int
            batch size for anti-ban, default 50
        batch_rest_time: int
            rest time between batches (seconds), default 60
        clash_config_path: str
            path to Clash config file (e.g., ~/.config/mihomo/config.yaml)
        use_clash: bool
            whether to use Clash local proxy, default True
        proxy_rotation_strategy: str
            proxy rotation strategy: round_robin, random, weighted, default round_robin
        """
        super(AkshareCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        
        self.batch_size = batch_size
        self.batch_rest_time = batch_rest_time
        self.ua = UserAgent()
        self.proxy_manager = None
        self.clash_config_path = clash_config_path
        self.use_clash = use_clash
        self.proxy_rotation_strategy = proxy_rotation_strategy
        
        # 初始化代理管理器
        self._init_proxy_manager(config_path)
    
    def _init_proxy_manager(self, config_path: Optional[str] = None):
        """初始化代理管理器"""
        proxy_list = []
        
        # 从 akshare_config.yaml 加载自定义代理
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'proxies' in config:
                        proxy_list = config['proxies']
                        logger.info(f"Loaded {len(proxy_list)} custom proxies from config")
            except Exception as e:
                logger.warning(f"Failed to load akshare config: {e}")
        else:
            default_config = CUR_DIR / "akshare_config.yaml"
            if default_config.exists():
                try:
                    with open(default_config, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if config and 'proxies' in config:
                            proxy_list = config['proxies']
                            logger.info(f"Loaded {len(proxy_list)} custom proxies from default config")
                except Exception as e:
                    logger.warning(f"Failed to load default config: {e}")
        
        # 创建代理管理器（会自动尝试加载 Clash）
        self.proxy_manager = ProxyManager(
            proxy_list=proxy_list if proxy_list else None,
            clash_config_path=self.clash_config_path,
            use_clash=self.use_clash,
            proxy_rotation_strategy=self.proxy_rotation_strategy
        )
    
    def get_instrument_list(self):
        """获取股票列表"""
        logger.info("get HS stock symbols from akshare......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols
    
    def normalize_symbol(self, symbol: str):
        """标准化股票代码"""
        # akshare 使用格式: 600000 (无后缀)
        if "." in symbol:
            symbol = symbol.split(".")[0]
        return code_to_fname(symbol).upper()
    
    def _get_last_date(self, symbol: str) -> Optional[pd.Timestamp]:
        """获取本地已有数据的最后日期（增量更新）"""
        symbol_normalized = self.normalize_symbol(symbol)
        symbol_fname = code_to_fname(symbol_normalized)
        file_path = self.save_dir / f"{symbol_fname}.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if not df.empty and 'date' in df.columns:
                    last_date = pd.to_datetime(df['date']).max()
                    return pd.Timestamp(last_date)
            except Exception as e:
                logger.warning(f"Failed to read last date for {symbol}: {e}")
        return None
    
    @exponential_backoff_retry(max_retries=5, base_delay=2.0)
    def get_data_from_akshare(
        self, 
        symbol: str, 
        start: str, 
        end: str,
        adjust: str = "qfq"  # qfq: 前复权, bfq: 不复权, hfq: 后复权
    ) -> pd.DataFrame:
        """从 akshare 获取数据（支持代理）"""
        # 动态随机延迟
        time.sleep(random.uniform(0.5, 2.0))
        
        # 转换股票代码格式: 600000.ss -> 600000
        if "." in symbol:
            symbol_code = symbol.split(".")[0]
        else:
            symbol_code = symbol
        
        # 获取代理设置
        proxy_dict = None
        current_proxy = None
        if self.proxy_manager:
            proxy_dict = self.proxy_manager.get_proxy()
            if proxy_dict:
                current_proxy = proxy_dict.get("http") or proxy_dict.get("https")
                # 设置环境变量供 akshare 使用
                import os
                if current_proxy:
                    os.environ['HTTP_PROXY'] = current_proxy
                    os.environ['HTTPS_PROXY'] = current_proxy
                    logger.debug(f"Using proxy: {current_proxy} for {symbol_code}")
        
        try:
            # 使用 akshare 获取数据
            # 注意：akshare 内部使用 requests，会自动读取环境变量中的代理设置
            df = ak.stock_zh_a_hist(
                symbol=symbol_code,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust
            )
            
            # 标记代理成功
            if current_proxy and self.proxy_manager:
                self.proxy_manager.mark_success(current_proxy)
            
            # 检查返回数据是否为空
            if df is None or df.empty:
                # 检查是否是无效股票代码（akshare 可能返回错误信息）
                logger.debug(f"{symbol_code}: akshare returned empty data for {start} to {end}")
                return pd.DataFrame()
            
            # 重命名列以匹配 Qlib 格式
            column_mapping = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
            
            df = df.rename(columns=column_mapping)
            
            # 选择需要的列
            required_cols = ["date", "open", "close", "high", "low", "volume"]
            df = df[[col for col in required_cols if col in df.columns]]
            
            # 添加 symbol 列
            df["symbol"] = self.normalize_symbol(symbol)
            
            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"])
            
            return df
            
        except requests.exceptions.HTTPError as e:
            # 标记代理失败
            if current_proxy and self.proxy_manager:
                self.proxy_manager.mark_failed(current_proxy)
            
            if e.response.status_code == 403:
                logger.error(f"403 Forbidden for {symbol}, triggering long rest")
                # 403 错误时切换代理并长休息
                if current_proxy and self.proxy_manager:
                    logger.warning(f"Switching proxy due to 403 error")
                time.sleep(random.uniform(30, 60))
            raise
        except Exception as e:
            # 标记代理失败
            if current_proxy and self.proxy_manager:
                self.proxy_manager.mark_failed(current_proxy)
            
            logger.warning(f"Failed to get data for {symbol}: {e}")
            raise
        finally:
            # 清理环境变量
            import os
            if current_proxy:
                os.environ.pop('HTTP_PROXY', None)
                os.environ.pop('HTTPS_PROXY', None)
    
    def get_data(
        self, 
        symbol: str, 
        interval: str, 
        start_datetime: pd.Timestamp, 
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """获取数据（支持增量更新）"""
        if interval != self.INTERVAL_1d:
            raise ValueError(f"akshare collector only supports 1d interval, got {interval}")
        
        # 增量更新：检查本地已有数据
        last_date = self._get_last_date(symbol)
        if last_date and last_date >= start_datetime:
            # 从最后日期的下一天开始
            start_datetime = last_date + pd.Timedelta(days=1)
            if start_datetime >= end_datetime:
                logger.info(f"{symbol} is already up to date")
                return pd.DataFrame()
        
        start_str = start_datetime.strftime("%Y-%m-%d")
        end_str = end_datetime.strftime("%Y-%m-%d")
        
        # 检查日期范围是否合理
        days_diff = (end_datetime - start_datetime).days
        if days_diff <= 0:
            logger.warning(f"{symbol}: Invalid date range: {start_str} to {end_str}")
            return pd.DataFrame()
        
        # 对于很小的日期范围（<=2天），可能是非交易日，只重试1次
        max_retries = 1 if days_diff <= 2 else self.retry
        
        @deco_retry(retry=max_retries, retry_sleep=random.uniform(1, 3))
        def _get_simple():
            self.sleep()
            resp = self.get_data_from_akshare(symbol, start_str, end_str)
            if resp is None or resp.empty:
                # 对于空数据，提供更详细的错误信息
                if days_diff <= 2:
                    logger.debug(f"{symbol}: No data for date range {start_str} to {end_str} (may be non-trading days)")
                else:
                    logger.warning(f"{symbol}: No data returned for date range {start_str} to {end_str}")
                raise ValueError(f"get data error: {symbol}--{start_str}--{end_str} (empty result)")
            return resp
        
        try:
            return _get_simple()
        except ValueError as e:
            # 对于空数据错误，不记录为严重错误，直接返回空 DataFrame
            error_msg = str(e)
            if "empty result" in error_msg:
                logger.debug(f"{symbol}: Skipping empty data result (may be delisted or non-trading days)")
            else:
                logger.warning(f"{symbol}: {error_msg}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _collector(self, instrument_list):
        """分批采集数据"""
        error_symbol = []
        total_batches = (len(instrument_list) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(instrument_list))
            batch = instrument_list[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} symbols)")
            
            # 处理当前批次
            for symbol in batch:
                try:
                    result = self._simple_collector(symbol) # --> get_data(),对于已更新，不再重新请求下载； 返回NORMAL_FLAG
                    if result != self.NORMAL_FLAG:
                        error_symbol.append(symbol)
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    error_symbol.append(symbol)
            
            # 批次间强制休息（最后一批除外）
            if batch_idx < total_batches - 1:
                rest_time = self.batch_rest_time + random.uniform(-10, 10)
                logger.info(f"Batch {batch_idx + 1} completed. Resting for {rest_time:.1f}s...")
                time.sleep(rest_time)
        
        logger.info(f"error symbol nums: {len(error_symbol)}")
        logger.info(f"current get symbol nums: {len(instrument_list)}")
        error_symbol.extend(self.mini_symbol_map.keys())
        return sorted(set(error_symbol))


class AkshareNormalize(BaseNormalize):
    """Akshare 数据标准化"""
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"
    
    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        """计算涨跌幅"""
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series
    
    @staticmethod
    def normalize_akshare(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        """标准化 akshare 数据"""
        if df.empty:
            return df
        
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(AkshareNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        
        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan
        
        change_series = AkshareNormalize.calc_change(df, last_close)
        df["change"] = change_series
        
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
        
        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据"""
        df = self.normalize_akshare(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        df = self.adjusted_price(df)
        return df
    
    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """复权处理"""
        raise NotImplementedError("rewrite adjusted_price")
    
    @abc.abstractmethod
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """获取交易日历"""
        raise NotImplementedError("rewrite _get_calendar_list")


class AkshareNormalize1d(AkshareNormalize, ABC):
    """Akshare 日线数据标准化"""
    DAILY_FORMAT = "%Y-%m-%d"
    
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """复权处理（akshare 已提供复权数据，这里主要做标准化）"""
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        
        # akshare 返回的是前复权数据，factor 设为 1
        df["factor"] = 1.0
        
        df.index.names = [self._date_field_name]
        return df.reset_index()
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据"""
        df = super(AkshareNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df
    
    def _get_first_close(self, df: pd.DataFrame) -> float:
        """获取第一个有效收盘价"""
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close
    
    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """手动调整数据：所有字段（除 change 外）按首日收盘价标准化"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)
        
        for _col in df.columns:
            if _col in [self._symbol_field_name, "factor", "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
        
        return df.reset_index()


class AkshareNormalizeCN1d(AkshareNormalize1d):
    """中国 A 股日线数据标准化"""
    
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """获取中国 A 股交易日历"""
        return get_calendar_list("ALL")


class AkshareNormalizeCN1dExtend(AkshareNormalizeCN1d):
    """Akshare 日线数据扩展标准化（用于增量更新）"""
    
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """
        Parameters
        ----------
        old_qlib_data_dir: str, Path
            the qlib data to be updated, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(AkshareNormalizeCN1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)
    
    def _get_old_data(self, qlib_data_dir: [str, Path]):
        """获取旧的 qlib 数据"""
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据（扩展模式，用于增量更新）"""
        df = super(AkshareNormalizeCN1dExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        
        if df.empty:
            return df.reset_index()
        
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        
        for col in self.column_list[:-1]:  # 排除 change
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        
        return df.drop(df.index[0]).reset_index()


class Run(BaseRun):
    """Akshare 数据采集运行类"""
    
    def __init__(
        self, 
        source_dir=None, 
        normalize_dir=None, 
        max_workers=1, 
        interval="1d", 
        region=REGION_CN,
        config_path: Optional[str] = None,
        clash_config_path: Optional[str] = None,
        use_clash: bool = True,
        proxy_rotation_strategy: str = "round_robin",
    ):
        """
        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved
        normalize_dir: str
            Directory for normalize data
        max_workers: int
            Concurrent number, default is 1 (recommended for anti-ban)
        interval: str
            freq, value from [1d], default 1d
        region: str
            region, default "CN"
        config_path: str
            path to akshare_config.yaml
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region
        self.config_path = config_path
        self.clash_config_path = clash_config_path
        self.use_clash = use_clash
        self.proxy_rotation_strategy = proxy_rotation_strategy
    
    @property
    def collector_class_name(self):
        return "AkshareCollector"
    
    @property
    def normalize_class_name(self):
        return f"AkshareNormalize{self.region.upper()}1d"
    
    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR
    
    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
        batch_size=50,
        batch_rest_time=60,
        clash_config_path: Optional[str] = None,
        use_clash: bool = True,
        proxy_rotation_strategy: str = "round_robin",
    ):
        """下载数据
        
        Examples
        ---------
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.5 --interval 1d
        """
        if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            raise ValueError(f"end_date: {end} is greater than the current date.")
        
        _class = getattr(self._cur_module, self.collector_class_name)
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            config_path=self.config_path,
            batch_size=batch_size,
            batch_rest_time=batch_rest_time,
            clash_config_path=clash_config_path or self.clash_config_path,
            use_clash=use_clash if clash_config_path is not None else self.use_clash,
            proxy_rotation_strategy=proxy_rotation_strategy if clash_config_path is not None else self.proxy_rotation_strategy,
        ).collector_data()
    
    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        """标准化数据
        
        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date
        )
    
    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """标准化数据扩展；用于扩展已有的 qlib 数据
        
        Notes
        -----
            Steps to extend akshare qlib data:
            
                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>
                
                2. collector source data: save to <dir2>
                
                3. normalize new source data(from step 2): 
                   python collector.py normalize_data_1d_extend --old_qlib_data_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d
                
                4. dump data: python scripts/dump_bin.py dump_update --csv_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date
                
                5. update instrument(eg. csi300): python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments
        
        Parameters
        ----------
        old_qlib_data_dir: str
            the qlib data to be updated, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        
        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_data_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()
    
    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
        batch_size: int = 50,
        batch_rest_time: int = 60,
    ):
        """更新 akshare 数据到 bin 格式（对齐 yahoo collector 用法）
        
        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        exists_skip: bool
            exists skip, by default False
        batch_size: int
            batch size for anti-ban, default 50
        batch_rest_time: int
            rest time between batches (seconds), default 60
        
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day
        
        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
        """
        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")
        
        if DumpDataUpdate is None:
            raise ImportError("dump_bin module not found. Please ensure scripts/dump_bin.py exists.")
        
        # download qlib 1d data if not exists
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )
        
        # start/end date
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        # download data from akshare
        # NOTE: when downloading data from akshare, max_workers is recommended to be 1
        self.download_data(
            delay=delay, 
            start=trading_date, 
            end=end_date, 
            check_data_length=check_data_length,
            batch_size=batch_size,
            batch_rest_time=batch_rest_time,
        )
        
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)
        
        # dump bin
        _dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()
        
        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)

