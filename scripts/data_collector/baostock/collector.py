# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import copy
import datetime
import multiprocessing
import shutil
from pathlib import Path
from typing import Iterable, List

import fire
import baostock as bs
import numpy as np
import pandas as pd
from loguru import logger

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, exists_qlib_data
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import get_calendar_list, get_hs_stock_symbols, get_instruments, symbol_suffix_to_prefix

# Earliest CSI300 coverage for qlib instrument segment (flat membership file only).
CSI300_INSTRUMENT_START = "2005-04-08"


def baostock_hs_code_to_qlib(bs_code: str) -> str:
    """baostock ``sh.600519`` / ``sz.000001`` -> qlib ``SH600519`` / ``SZ000001``."""
    s = str(bs_code).strip().lower()
    if "." not in s:
        return ""
    ex, num = s.split(".", 1)
    num = num.strip()
    if not num.isdigit():
        return ""
    if ex == "sh":
        return "SH" + num.zfill(6)
    if ex == "sz":
        return "SZ" + num.zfill(6)
    if ex == "bj":
        return "BJ" + num.upper()
    return ""


def query_hs300_symbols_baostock(trade_date: str) -> List[str]:
    """Return sorted unique qlib symbols for CSI300 (沪深300) on *trade_date* (YYYY-MM-DD)."""
    rs = bs.query_hs300_stocks(date=trade_date)
    if rs.error_code != "0":
        logger.warning(f"query_hs300_stocks({trade_date}): {rs.error_msg}")
        return []
    out: List[str] = []
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        if len(row) > 1:
            q = baostock_hs_code_to_qlib(row[1])
            if q:
                out.append(q)
    return sorted(set(out))


class BaostockCollectorCN1d(BaseCollector):
    """A-share daily collector. Index codes are optional (see include_index_symbols)."""

    INDEX_SYMBOLS = ["000300.ss", "000903.ss", "000905.ss"]

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        adjustflag: str = "2",
        qlib_data_1d_dir: str = None,
        instrument_scope: str = "all",
        include_index_symbols: bool = None,
        source_skip_existing: bool = False,
        normalize_dir: [str, Path] = None,
    ):
        """
        Parameters
        ----------
        save_dir: str
            stock save dir
        adjustflag: str
            baostock adjust flag. "2" means qfq, "1" means hfq, "3" means no adjustment.
        normalize_dir: str or Path, optional
            If set and ``source_skip_existing`` is True, also skip remote fetch when the matching
            normalized csv under this directory already covers the requested range (same rule as source).
        """
        if interval != self.INTERVAL_1d:
            raise ValueError(f"Baostock daily collector only supports interval={self.INTERVAL_1d}")
        bs.login()
        self.adjustflag = str(adjustflag)
        self.qlib_data_1d_dir = (
            str(Path(qlib_data_1d_dir).expanduser().resolve()) if qlib_data_1d_dir is not None else None
        )
        self.normalize_dir = (
            Path(normalize_dir).expanduser().resolve() if normalize_dir is not None else None
        )
        self.instrument_scope = (instrument_scope or "all").strip().lower()
        if include_index_symbols is None:
            include_index_symbols = self.instrument_scope == "all"
        self.include_index_symbols = bool(include_index_symbols)
        self.source_skip_existing = bool(source_skip_existing)
        super(BaostockCollectorCN1d, self).__init__(
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

    @staticmethod
    def process_interval(interval: str):
        if interval != BaseCollector.INTERVAL_1d:
            raise ValueError(f"unsupported interval: {interval}")
        return {
            "raw_fields": "date,code,open,high,low,close,volume,amount,adjustflag",
            "adj_fields": "date,code,close",
            "interval": "d",
        }

    @staticmethod
    def symbol_to_bs(symbol: str) -> str:
        code, exchange = symbol.lower().split(".")
        exchange = "sh" if exchange in {"sh", "ss"} else "sz"
        return f"{exchange}.{code}"

    @staticmethod
    def qlib_instr_line_to_suffix(symbol: str) -> str:
        """Map qlib instrument id (e.g. SH600000 / sz000001) to Yahoo-style suffix for baostock."""
        s = str(symbol).strip()
        if not s:
            return ""
        su = s.upper()
        if su.startswith("SH") and len(su) > 2 and su[2:].isdigit():
            return f"{su[2:]}.ss"
        if su.startswith("SZ") and len(su) > 2 and su[2:].isdigit():
            return f"{su[2:]}.sz"
        if su.startswith("BJ"):
            return ""
        sl = s.lower()
        if sl.startswith("sh") and len(sl) > 2 and sl[2:].isdigit():
            return f"{sl[2:]}.ss"
        if sl.startswith("sz") and len(sl) > 2 and sl[2:].isdigit():
            return f"{sl[2:]}.sz"
        if sl.startswith("bj"):
            return ""
        return ""

    def _instruments_file_path(self) -> Path:
        if self.qlib_data_1d_dir is None:
            return Path()
        base = Path(self.qlib_data_1d_dir).joinpath("instruments")
        if self.instrument_scope == "all":
            return base.joinpath("all.txt")
        return base.joinpath(f"{self.instrument_scope}.txt")

    def get_instrument_list_from_qlib(self):
        if self.qlib_data_1d_dir is None:
            return []
        instrument_path = self._instruments_file_path()
        if not instrument_path.exists():
            logger.warning(f"qlib instrument file not found: {instrument_path}")
            return []

        df = pd.read_csv(instrument_path, sep="\t", header=None, usecols=[0], names=["symbol"])
        core = {
            converted
            for converted in df["symbol"].map(self.qlib_instr_line_to_suffix)
            if converted and converted.split(".")[0].isdigit()
        }
        extra = set(self.INDEX_SYMBOLS) if self.include_index_symbols else set()
        symbols = sorted(core | extra)
        return symbols

    def get_instrument_list(self):
        symbols = self.get_instrument_list_from_qlib()
        if symbols:
            logger.info(f"load {len(symbols)} symbols from {self._instruments_file_path().name} (scope={self.instrument_scope})")
            return symbols

        logger.info("get HS stock symbols from remote source......")
        extra = set(self.INDEX_SYMBOLS) if self.include_index_symbols else set()
        symbols = sorted(set(get_hs_stock_symbols()) | extra)
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return symbol_suffix_to_prefix(symbol, capital=False)

    @classmethod
    def get_data_from_remote(
        cls,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
        adjustflag: str,
        fields: str,
    ) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(
            symbol,
            fields,
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency=cls.process_interval(interval=interval)["interval"],
            adjustflag=str(adjustflag),
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            return pd.DataFrame(rs.data, columns=rs.fields)
        return pd.DataFrame()

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        bs_symbol = self.symbol_to_bs(symbol)
        interval_conf = self.process_interval(interval=interval)
        raw_df = self.get_data_from_remote(
            symbol=bs_symbol,
            interval=interval,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            adjustflag="3",
            fields=interval_conf["raw_fields"],
        )
        if raw_df.empty:
            return raw_df

        adj_df = self.get_data_from_remote(
            symbol=bs_symbol,
            interval=interval,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            adjustflag=self.adjustflag,
            fields=interval_conf["adj_fields"],
        )

        raw_df = raw_df.rename(columns={"code": "symbol"})
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["symbol"] = self.normalize_symbol(symbol)

        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

        if adj_df.empty:
            raw_df["adjclose"] = raw_df["close"]
        else:
            adj_df = adj_df.rename(columns={"code": "symbol", "close": "adjclose"})
            adj_df["date"] = pd.to_datetime(adj_df["date"])
            adj_df["adjclose"] = pd.to_numeric(adj_df["adjclose"], errors="coerce")
            raw_df = raw_df.merge(adj_df.loc[:, ["date", "adjclose"]], on="date", how="left")
            raw_df["adjclose"] = raw_df["adjclose"].fillna(raw_df["close"])

        return raw_df

    @staticmethod
    def _csv_file_covers_through(csv_path: Path, need_until: pd.Timestamp) -> bool:
        """True if csv ``date`` column max (normalized) >= need_until."""
        if not csv_path.exists():
            return False
        try:
            old = pd.read_csv(csv_path, usecols=["date"], low_memory=False)
            if old.empty:
                return False
            max_d = pd.to_datetime(old["date"], errors="coerce").max()
            if pd.isna(max_d):
                return False
            return max_d.normalize() >= need_until.normalize()
        except Exception:
            return False

    def _local_csv_covers_request(self, symbol: str) -> tuple:
        """
        Returns
        -------
        (skip_remote: bool, reason: str)
            If skip_remote, existing local data already has rows through the last calendar day before end_datetime.
        """
        if not self.source_skip_existing:
            return False, ""
        fname = f"{code_to_fname(self.normalize_symbol(symbol))}.csv"
        # BaseCollector end is exclusive (open interval); last needed bar is end - 1 calendar day
        need_until = (pd.Timestamp(self.end_datetime).normalize() - pd.Timedelta(days=1)).normalize()
        src = self.save_dir.joinpath(fname)
        if self._csv_file_covers_through(src, need_until):
            return True, "source"
        if self.normalize_dir is not None:
            norm = self.normalize_dir.joinpath(fname)
            if self._csv_file_covers_through(norm, need_until):
                return True, "normalize"
        return False, ""

    def _simple_collector(self, symbol: str):
        self.sleep()
        skip, why = self._local_csv_covers_request(symbol)
        if skip:
            logger.info(f"skip download ({why} up to date): {symbol}")
            return self.NORMAL_FLAG
        df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
        _result = self.NORMAL_FLAG
        if self.check_data_length > 0:
            _result = self.cache_small_data(symbol, df)
        if _result == self.NORMAL_FLAG:
            self.save_instrument(symbol, df)
        return _result


class BaostockNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        tmp_series = df["close"].ffill()
        tmp_shift_series = tmp_series.shift(1)
        if last_close is not None:
            tmp_shift_series.iloc[0] = float(last_close)
        return tmp_series / tmp_shift_series - 1

    @staticmethod
    def normalize_baostock(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(BaostockNormalize.COLUMNS)
        df = df.copy()
        # Normalize to midnight so index labels match qlib/calendar slices (avoids "%Y-%m-%d" vs "… 00:00:00" mix).
        df[date_field_name] = pd.to_datetime(df[date_field_name], errors="coerce").dt.normalize()
        df = df.dropna(subset=[date_field_name])
        if df.empty:
            return df
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index).normalize()
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
        df["change"] = BaostockNormalize.calc_change(df, last_close)
        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan
        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_baostock(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        df = self.adjusted_price(df)
        return df

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("rewrite adjusted_price")


class BaostockNormalizeCN1d(BaostockNormalize):
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df.columns:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].replace([np.inf, -np.inf], np.nan).ffill()
        else:
            df["factor"] = 1.0
        for col in self.COLUMNS:
            if col not in df.columns:
                continue
            if col == "volume":
                df[col] = df[col] / df["factor"]
            else:
                df[col] = df[col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def _get_first_close(self, df: pd.DataFrame) -> float:
        df = df.loc[df["close"].first_valid_index() :]
        return df["close"].iloc[0]

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        first_close = self._get_first_close(df)
        for col in df.columns:
            if col in [self._symbol_field_name, "adjclose", "change", "factor", "amount", "adjustflag"]:
                continue
            if col == "volume":
                df[col] = df[col] * first_close
            else:
                df[col] = df[col] / first_close
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(BaostockNormalizeCN1d, self).normalize(df)
        return self._manual_adj_data(df)


class BaostockNormalizeCN1dExtend(BaostockNormalizeCN1d):
    def __init__(
        self,
        old_qlib_data_dir: [str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        qlib_instrument_market: str = "all",
        **kwargs,
    ):
        super(BaostockNormalizeCN1dExtend, self).__init__(date_field_name, symbol_field_name, **kwargs)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.qlib_instrument_market = (qlib_instrument_market or "all").strip().lower()
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        inst = D.instruments(self.qlib_instrument_market)
        df = D.features(inst, ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(BaostockNormalizeCN1dExtend, self).normalize(df)
        if df.empty:
            return df
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index).normalize()
        symbol_name = str(df[self._symbol_field_name].iloc[0]).upper()
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if symbol_name not in old_symbol_list:
            return df.reset_index()

        old_df = self.old_qlib_data.loc[symbol_name]
        latest_date = pd.Timestamp(old_df.index[-1]).normalize()
        df = df.loc[latest_date:]
        if df.empty:
            return df.reset_index()

        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col not in df.columns:
                continue
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class Run(BaseRun):
    def __init__(
        self,
        source_dir=None,
        normalize_dir=None,
        max_workers=1,
        interval="1d",
        region=REGION_CN,
        adjustflag: str = "2",
        instrument_scope: str = "all",
    ):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region
        self.adjustflag = str(adjustflag)
        self.instrument_scope = (instrument_scope or "all").strip().lower()

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

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
        qlib_data_1d_dir: str = None,
        instrument_scope: str = None,
        include_index_symbols: bool = None,
        source_skip_existing: bool = False,
        normalize_dir: str = None,
    ):
        if self.interval.lower() != "1d":
            raise ValueError("baostock daily updater only supports interval=1d")
        if start is None:
            start = "2000-01-01"
        if end is None:
            end = (pd.Timestamp(datetime.datetime.now().date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError(f"start date {start} must be earlier than end date {end}")
        scope = instrument_scope if instrument_scope is not None else self.instrument_scope
        ndir = normalize_dir if normalize_dir is not None else str(self.normalize_dir)
        super(Run, self).download_data(
            max_collector_count,
            delay,
            start,
            end,
            check_data_length,
            limit_nums,
            adjustflag=self.adjustflag,
            qlib_data_1d_dir=qlib_data_1d_dir,
            instrument_scope=scope,
            include_index_symbols=include_index_symbols,
            source_skip_existing=source_skip_existing,
            normalize_dir=ndir,
        )

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        if self.interval.lower() != "1d":
            raise ValueError("baostock daily updater only supports interval=1d")
        super(Run, self).normalize_data(date_field_name, symbol_field_name, end_date=end_date)

    def normalize_data_1d_extend(
        self,
        old_qlib_data_dir,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        qlib_instrument_market: str = None,
        normalize_skip_existing: bool = True,
        request_end_exclusive: str = None,
    ):
        if self.interval.lower() != "1d":
            raise ValueError("baostock daily updater only supports interval=1d")
        norm_class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        market = qlib_instrument_market if qlib_instrument_market is not None else self.instrument_scope
        skip_kw = {}
        if normalize_skip_existing and request_end_exclusive is not None:
            skip_kw["skip_target_if_covers_until"] = pd.Timestamp(request_end_exclusive)
        norm = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=norm_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
            qlib_instrument_market=market,
            **skip_kw,
        )
        norm.normalize()

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 0.5,
        exists_skip: bool = False,
        instrument_scope: str = None,
        include_index_symbols: bool = None,
        source_skip_existing: bool = True,
        normalize_skip_existing: bool = True,
        update_index_instruments: bool = False,
        refresh_csi300_instruments_baostock: bool = False,
    ):
        """
        Incrementally update CN daily qlib data using baostock.

        Parameters
        ----------
        instrument_scope:
            Which qlib `instruments/<name>.txt` to use for the symbol list, e.g. `csi300`.
            Default uses `Run` init value (default `all`).
        source_skip_existing:
            If True, skip baostock download when local **source** or **normalize** csv (same basename)
            already has rows through end-1 calendar day (open ``end_date`` interval), to avoid slow remote calls.
        normalize_skip_existing:
            If True, skip rewriting a normalize output csv when it already covers through end-1 day
            (avoids reprocessing when download was skipped and stale source would overwrite fresh normalize).
        update_index_instruments:
            If True, refresh CSI index membership via cn_index (needs network). Default False to avoid flaky APIs.
        refresh_csi300_instruments_baostock:
            If True and scope is ``csi300``, rewrite ``instruments/csi300.txt`` from baostock ``query_hs300_stocks``
            (no Eastmoney / csindex). See ``update_csi300_instruments_from_baostock``.
        """
        if self.interval.lower() != "1d":
            raise ValueError("currently only supports 1d data updates: --interval 1d")

        scope = (instrument_scope if instrument_scope is not None else self.instrument_scope).strip().lower()

        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        calendar_path = Path(qlib_data_1d_dir).joinpath("calendars/day.txt")
        calendar_df = pd.read_csv(calendar_path, header=None)
        latest_date = pd.Timestamp(calendar_df.iloc[-1, 0]).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(datetime.datetime.now().date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(end_date) <= pd.Timestamp(latest_date):
            logger.warning(f"end_date={end_date} is not later than latest qlib date={latest_date}, nothing to update")
            return

        self.download_data(
            delay=delay,
            start=latest_date,
            end=end_date,
            check_data_length=check_data_length,
            qlib_data_1d_dir=qlib_data_1d_dir,
            instrument_scope=scope,
            include_index_symbols=include_index_symbols,
            source_skip_existing=source_skip_existing,
            normalize_dir=str(self.normalize_dir),
        )

        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        self.normalize_data_1d_extend(
            qlib_data_1d_dir,
            qlib_instrument_market=scope,
            normalize_skip_existing=normalize_skip_existing,
            request_end_exclusive=end_date,
        )

        dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        dump.dump()

        if refresh_csi300_instruments_baostock and scope == "csi300":
            self.update_csi300_instruments_from_baostock(qlib_data_1d_dir=qlib_data_1d_dir)

        if update_index_instruments:
            if scope == "csi300":
                index_names = ["CSI300"]
            elif scope == "csi100":
                index_names = ["CSI100"]
            elif scope == "csi500":
                index_names = ["CSI500"]
            else:
                index_names = ["CSI100", "CSI300", "CSI500"]
            for index_name in index_names:
                try:
                    get_instruments(qlib_data_1d_dir, index_name, market_index="cn_index")
                except Exception as e:
                    logger.warning(f"skip index refresh {index_name}: {e}")

    def update_csi300_instruments_from_baostock(
        self,
        qlib_data_1d_dir: str,
        as_of_date: str = None,
        lookback_calendar_days: int = 30,
    ):
        """
        Rewrite ``instruments/csi300.txt`` using baostock ``query_hs300_stocks`` only.

        Notes
        -----
        - Does **not** use cn_index / Eastmoney / csindex (avoids RemoteDisconnected there).
        - Writes a **single membership segment** per symbol: ``start`` = CSI300_INSTRUMENT_START,
          ``end`` = *as_of_date* (or last line of ``calendars/day.txt``). This is enough for
          ``D.list_instruments`` on recent dates but **not** a full historical constituent history.

        Examples
        --------
        $ python collector.py update_csi300_instruments_from_baostock --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
        $ python collector.py update_csi300_instruments_from_baostock --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --as_of_date 2026-03-20
        """
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        inst_dir = Path(qlib_data_1d_dir).joinpath("instruments")
        inst_dir.mkdir(parents=True, exist_ok=True)
        target = inst_dir.joinpath("csi300.txt")

        cal_path = Path(qlib_data_1d_dir).joinpath("calendars", "day.txt")
        if not cal_path.exists():
            raise FileNotFoundError(f"Missing calendar: {cal_path}")
        cal_lines = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not cal_lines:
            raise ValueError("calendars/day.txt is empty")

        if as_of_date is None:
            as_of_date = cal_lines[-1]

        bs.login()
        try:
            symbols = query_hs300_symbols_baostock(as_of_date)
            if not symbols:
                for d in reversed(cal_lines[-int(lookback_calendar_days) :]):
                    symbols = query_hs300_symbols_baostock(d)
                    if symbols:
                        logger.warning(f"HS300 list empty on {as_of_date}; using baostock date {d}")
                        as_of_date = d
                        break
        finally:
            bs.logout()

        if not symbols:
            raise ValueError(
                f"No HS300 constituents from baostock for {as_of_date} (tried last {lookback_calendar_days} calendar days)."
            )

        if target.exists():
            bak = inst_dir.joinpath("csi300.txt.bak")
            shutil.copy2(target, bak)
            logger.info(f"Backed up previous csi300.txt to {bak}")

        end = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
        body = "".join(f"{sym}\t{CSI300_INSTRUMENT_START}\t{end}\n" for sym in symbols)
        target.write_text(body, encoding="utf-8")
        logger.info(f"Wrote {len(symbols)} CSI300 lines to {target} (segment {CSI300_INSTRUMENT_START} ~ {end})")


if __name__ == "__main__":
    fire.Fire(Run)
