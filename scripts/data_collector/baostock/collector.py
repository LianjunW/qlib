# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import copy
import datetime
import errno
import json
import multiprocessing
import random
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional

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

# -----------------------------------------------------------------------------
# Baostock remote calls often fail transiently (timeouts, resets). Retry with
# exponential backoff instead of failing the whole daily update on first error.
# -----------------------------------------------------------------------------
BAOSTOCK_RETRY_ATTEMPTS = 6
BAOSTOCK_RETRY_BASE_SEC = 1.0
BAOSTOCK_RETRY_MAX_SEC = 90.0

_NET_ERRNOS = frozenset(
    e
    for e in (
        getattr(errno, name, None)
        for name in (
            "ECONNRESET",
            "ECONNREFUSED",
            "ETIMEDOUT",
            "EPIPE",
            "EHOSTUNREACH",
            "ENETUNREACH",
            "ENOTCONN",
            "EAI_AGAIN",
        )
    )
    if e is not None
)


def _baostock_retry_delay(attempt_index: int) -> None:
    """Sleep after attempt ``attempt_index`` failed (0-based)."""
    sec = min(BAOSTOCK_RETRY_MAX_SEC, BAOSTOCK_RETRY_BASE_SEC * (2**attempt_index))
    sec += random.uniform(0, min(4.0, 0.2 * sec))
    time.sleep(sec)


def _baostock_transient_exc(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionError, TimeoutError, InterruptedError)):
        return True
    if isinstance(exc, OSError):
        en = getattr(exc, "errno", None)
        if en is not None and (en in _NET_ERRNOS or en == errno.EAGAIN):
            return True
    msg_raw = str(exc)
    msg = msg_raw.lower()
    if any(x in msg_raw for x in ("网络", "超时", "连接失败", "断开")):
        return True
    return any(
        x in msg
        for x in (
            "timeout",
            "timed out",
            "connection reset",
            "broken pipe",
            "network is unreachable",
            "connection aborted",
            "remotedisconnected",
            "temporarily unavailable",
            "eof occurred",
        )
    )


def bs_login_with_retry():
    """Login with retries; returns baostock login result object (check ``error_code``)."""
    last_lg = None
    for attempt in range(BAOSTOCK_RETRY_ATTEMPTS):
        try:
            last_lg = bs.login()
        except Exception as e:
            last_lg = None
            if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                raise
            if not _baostock_transient_exc(e):
                raise
            logger.warning(f"baostock.login raised ({attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}): {e}")
            _baostock_retry_delay(attempt)
            continue
        if getattr(last_lg, "error_code", None) == "0":
            return last_lg
        msg = getattr(last_lg, "error_msg", "") or ""
        logger.warning(f"baostock.login failed ({attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}): {msg}")
        try:
            bs.logout()
        except Exception:
            pass
        if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
            return last_lg
        _baostock_retry_delay(attempt)
    return last_lg


# Earliest CSI300 coverage for qlib instrument segment (flat membership file only).
CSI300_INSTRUMENT_START = "2005-04-08"
CSI300_BAOSTOCK_SNAPSHOT_SCOPE = "csi300_baostock_snapshot"


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
    for attempt in range(BAOSTOCK_RETRY_ATTEMPTS):
        try:
            rs = bs.query_hs300_stocks(date=trade_date)
            if rs.error_code != "0":
                msg = rs.error_msg
                logger.warning(
                    f"query_hs300_stocks({trade_date}) attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}: {msg}"
                )
                if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                    return []
                _baostock_retry_delay(attempt)
                continue
            out: List[str] = []
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if len(row) > 1:
                    q = baostock_hs_code_to_qlib(row[1])
                    if q:
                        out.append(q)
            return sorted(set(out))
        except Exception as e:
            logger.warning(
                f"query_hs300_stocks({trade_date}) attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS} raised: {e}"
            )
            if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                return []
            _baostock_retry_delay(attempt)
    return []


def infer_latest_trade_date_from_source_dir(source_dir: [str, Path], instrument_scope_file: Optional[Path] = None) -> Optional[str]:
    """
    Infer the latest raw trade date from downloaded baostock csv files.

    This is used when local source csv files are newer than qlib calendars/day.txt.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        return None

    allowed = None
    if instrument_scope_file is not None and instrument_scope_file.exists():
        try:
            inst_df = pd.read_csv(instrument_scope_file, sep="\t", header=None, usecols=[0], names=["symbol"])
            allowed = {str(x).strip().lower() for x in inst_df["symbol"] if str(x).strip()}
        except Exception:
            allowed = None

    latest = None
    for csv_path in source_dir.glob("*.csv"):
        stem = csv_path.stem.strip().lower()
        if allowed is not None and stem.upper() not in {x.upper() for x in allowed}:
            continue
        try:
            dates = pd.read_csv(csv_path, usecols=["date"])
        except Exception:
            continue
        if dates.empty:
            continue
        tail = pd.to_datetime(dates["date"], errors="coerce").dropna()
        if tail.empty:
            continue
        cur = pd.Timestamp(tail.max()).normalize()
        if latest is None or cur > latest:
            latest = cur
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def query_latest_trading_day_cn_baostock(
    start_back_days: int = 420,
    end_forward_days: int = 14,
) -> Optional[str]:
    """
    Latest CN stock exchange trading day for daily (1d) bars, from baostock ``query_trade_dates``.

    The window is ``[today - start_back_days, today + end_forward_days]`` (calendar days on the
    baostock side). Use this as the authoritative "data calendar through" hint before/after
    incremental updates when you want the exchange calendar instead of local clock ``+1 day``.
    """
    today = pd.Timestamp.now().normalize()
    start = (today - pd.Timedelta(days=int(start_back_days))).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=int(end_forward_days))).strftime("%Y-%m-%d")
    try:
        lg = bs_login_with_retry()
    except Exception as e:
        logger.error(f"baostock.login raised: {e}")
        return None
    if lg.error_code != "0":
        logger.error(f"baostock.login failed: {lg.error_msg}")
        return None
    try:
        rs = None
        for attempt in range(BAOSTOCK_RETRY_ATTEMPTS):
            try:
                rs = bs.query_trade_dates(start_date=start, end_date=end)
                if rs.error_code != "0":
                    msg = rs.error_msg
                    logger.warning(f"query_trade_dates attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}: {msg}")
                    if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                        logger.error(f"query_trade_dates failed: {msg}")
                        return None
                    _baostock_retry_delay(attempt)
                    continue
                break
            except Exception as e:
                logger.warning(f"query_trade_dates attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS} raised: {e}")
                if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                    logger.error(f"query_trade_dates raised: {e}")
                    return None
                _baostock_retry_delay(attempt)
        if rs is None or rs.error_code != "0":
            return None
        rows: List = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            logger.error("query_trade_dates returned no rows")
            return None
        cal = pd.DataFrame(rows, columns=rs.fields)
        if "calendar_date" not in cal.columns or "is_trading_day" not in cal.columns:
            logger.error(f"query_trade_dates unexpected fields: {list(cal.columns)}")
            return None
        flag = cal["is_trading_day"]
        if flag.dtype == object or getattr(flag.dtype, "name", "") == "string":
            mask = flag.astype(str).str.strip().isin(("1", "1.0"))
        else:
            mask = pd.to_numeric(flag, errors="coerce").fillna(0).astype(int) == 1
        dates = pd.to_datetime(cal.loc[mask, "calendar_date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def resolve_local_csi300_scope_fallback(qlib_data_1d_dir: str) -> Optional[str]:
    """
    Prefer an existing baostock snapshot scope, otherwise fall back to historical ``csi300``.

    This is used when baostock is temporarily unavailable but local instrument files already
    exist and can keep the pipeline moving in a best-effort mode.
    """
    inst_dir = Path(qlib_data_1d_dir).expanduser().resolve().joinpath("instruments")
    snapshot = inst_dir.joinpath(f"{CSI300_BAOSTOCK_SNAPSHOT_SCOPE}.txt")
    if snapshot.exists() and snapshot.stat().st_size > 0:
        return CSI300_BAOSTOCK_SNAPSHOT_SCOPE
    csi300 = inst_dir.joinpath("csi300.txt")
    if csi300.exists() and csi300.stat().st_size > 0:
        return "csi300"
    return None


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
        try:
            lg = bs_login_with_retry()
        except Exception as e:
            raise RuntimeError(f"baostock.login failed during collector init: {e}") from e
        if getattr(lg, "error_code", None) != "0":
            raise RuntimeError(
                f"baostock.login failed during collector init: {getattr(lg, 'error_msg', 'unknown error')}"
            )
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
        freq = cls.process_interval(interval=interval)["interval"]
        start_s = str(start_datetime.strftime("%Y-%m-%d"))
        end_s = str(end_datetime.strftime("%Y-%m-%d"))
        adj = str(adjustflag)
        for attempt in range(BAOSTOCK_RETRY_ATTEMPTS):
            try:
                rs = bs.query_history_k_data_plus(
                    symbol,
                    fields,
                    start_date=start_s,
                    end_date=end_s,
                    frequency=freq,
                    adjustflag=adj,
                )
            except Exception as e:
                logger.warning(
                    f"query_history_k_data_plus({symbol}) attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}: {e}"
                )
                if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                    return pd.DataFrame()
                _baostock_retry_delay(attempt)
                continue
            if rs.error_code == "0":
                if len(rs.data) > 0:
                    return pd.DataFrame(rs.data, columns=rs.fields)
                return pd.DataFrame()
            msg = rs.error_msg
            logger.warning(f"query_history_k_data_plus({symbol}) attempt {attempt + 1}/{BAOSTOCK_RETRY_ATTEMPTS}: {msg}")
            if attempt == BAOSTOCK_RETRY_ATTEMPTS - 1:
                return pd.DataFrame()
            _baostock_retry_delay(attempt)
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

    def save_instrument(self, symbol, df: pd.DataFrame):
        """Persist daily source csv in date order and keep only the latest row per trading day."""
        if df is None or df.empty:
            logger.warning(f"{symbol} is empty")
            return

        norm_symbol = self.normalize_symbol(symbol)
        instrument_path = self.save_dir.joinpath(f"{code_to_fname(norm_symbol)}.csv")
        df = df.copy()
        df["symbol"] = code_to_fname(norm_symbol)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])

        if instrument_path.exists():
            old_df = pd.read_csv(instrument_path, low_memory=False)
            if "date" in old_df.columns:
                old_df["date"] = pd.to_datetime(old_df["date"], errors="coerce").dt.normalize()
            df = pd.concat([old_df, df], sort=False, ignore_index=True)

        df = df.dropna(subset=["date"]).sort_values(["date"]).drop_duplicates(["date"], keep="last")
        df.to_csv(instrument_path, index=False, date_format="%Y-%m-%d")


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
            # Keep newly downloaded trade dates even when qlib's existing day.txt lags behind.
            # Without this, incremental raw bars after the last qlib calendar date are dropped here.
            normalized_calendar = pd.DatetimeIndex(pd.to_datetime(calendar_list)).normalize()
            normalized_calendar = normalized_calendar.union(pd.DatetimeIndex(df.index).normalize())
            df = df.reindex(
                pd.DataFrame(index=normalized_calendar)
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
        if "amount" in df.columns and "volume" in df.columns:
            amount = pd.to_numeric(df["amount"], errors="coerce")
            volume = pd.to_numeric(df["volume"], errors="coerce")
            # baostock daily volume is lot-based (100 shares), while our normalized
            # volume has already been scaled by first_close. Converting ``amount``
            # back to the same normalized price space yields qlib-compatible VWAP.
            df["vwap"] = amount / volume.replace(0, np.nan) / 100.0
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
        self.old_qlib_data_dir = str(Path(old_qlib_data_dir).expanduser().resolve())
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

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        cal_path = Path(self.old_qlib_data_dir).joinpath("calendars", "day.txt")
        if cal_path.exists():
            lines = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lines:
                return pd.to_datetime(lines).normalize()
        return super()._get_calendar_list()

    @staticmethod
    def _stable_ratio(series_new: pd.Series, series_old: pd.Series, inverse: bool = False) -> float:
        new = pd.to_numeric(series_new, errors="coerce")
        old = pd.to_numeric(series_old, errors="coerce")
        if inverse:
            ratio = new / old.replace(0, np.nan)
        else:
            ratio = old / new.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            return np.nan
        return float(ratio.median())

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
        overlap_idx = df.index.intersection(old_df.index)
        future_df = df.loc[latest_date:].copy()
        if future_df.empty:
            return df.reset_index()

        # Prefer a stable median ratio from the full overlap window. This is
        # more robust than using a single overlap row, and better matches the
        # published qlib data scale when local source csv has been re-generated.
        overlap_df = pd.DataFrame(index=overlap_idx)
        if not overlap_df.empty:
            for col in self.column_list[:-1]:
                if col not in df.columns or col not in old_df.columns:
                    continue
                overlap_df[f"{col}_new"] = df.loc[overlap_idx, col]
                overlap_df[f"{col}_old"] = old_df.loc[overlap_idx, col]

        scale_map = {}
        for col in self.column_list[:-1]:
            if col not in future_df.columns:
                continue
            if not overlap_df.empty and f"{col}_new" in overlap_df.columns:
                ratio = self._stable_ratio(
                    overlap_df[f"{col}_new"],
                    overlap_df[f"{col}_old"],
                    inverse=(col == "volume"),
                )
            else:
                ratio = np.nan
            if not np.isfinite(ratio):
                if latest_date not in df.index:
                    continue
                new_latest_data = df.loc[latest_date]
                old_latest_data = old_df.loc[latest_date]
                if col == "volume":
                    base = pd.to_numeric(old_latest_data[col], errors="coerce")
                    ratio = pd.to_numeric(new_latest_data[col], errors="coerce") / base if pd.notna(base) and base != 0 else np.nan
                else:
                    base = pd.to_numeric(new_latest_data[col], errors="coerce")
                    ratio = pd.to_numeric(old_latest_data[col], errors="coerce") / base if pd.notna(base) and base != 0 else np.nan
            if not np.isfinite(ratio):
                continue
            scale_map[col] = float(ratio)
            if col == "volume":
                future_df[col] = future_df[col] / ratio
            else:
                future_df[col] = future_df[col] * ratio
        if scale_map:
            logger.debug(f"{symbol_name} overlap rescale factors: {scale_map}")
        # Only drop the overlap row when the first row is the old latest_date.
        if future_df.index[0].normalize() == latest_date.normalize():
            future_df = future_df.drop(future_df.index[0])
        return future_df.reset_index()


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

    def print_latest_cn_trade_date_baostock(self, start_back_days: int = 420, end_forward_days: int = 14) -> str:
        """CLI helper: print the last CN trading day (1d) from baostock ``query_trade_dates``."""
        d = query_latest_trading_day_cn_baostock(
            start_back_days=int(start_back_days), end_forward_days=int(end_forward_days)
        )
        if d is None:
            raise RuntimeError("Could not resolve latest CN trade date from baostock (login or empty calendar).")
        print(d, flush=True)
        return d

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
        start_from_bin_end: bool = False,
        bin_end_field: str = "close",
        skip_download: bool = False,
        skip_normalize: bool = False,
        repair_instruments_end_by_bin: bool = False,
        rebuild_normalize_from_source: bool = False,
        rewrite_scope_bins_from_normalize: bool = False,
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
            If True, refresh CSI index membership via cn_index **before download** so the current update
            uses real historical constituents instead of stale local files. Needs network.
        refresh_csi300_instruments_baostock:
            If True and scope is ``csi300``, generate ``instruments/csi300_baostock_snapshot.txt`` from
            baostock ``query_hs300_stocks`` and use that snapshot scope for this update. The historical
            ``instruments/csi300.txt`` file is kept unchanged. See
            ``update_csi300_instruments_from_baostock``.
        rebuild_normalize_from_source:
            If True, force re-run normalize from existing source csv files against the current qlib base
            even when the requested ``end_date`` is not later than the last calendar day. Use this for
            periodic realignment after replacing qlib release data.
        rewrite_scope_bins_from_normalize:
            If True, rebuild feature bin files for the current scope directly from normalized csv files
            after the incremental dump. This is slower than append-only update, but fixes cases where
            symbols re-enter the active prediction universe after earlier narrow-scope updates.
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
        if pd.Timestamp(end_date) <= pd.Timestamp(latest_date) and not rebuild_normalize_from_source:
            logger.warning(f"end_date={end_date} is not later than latest qlib date={latest_date}, nothing to update")
            return

        effective_scope = scope
        logger.info(
            "start incremental update: "
            f"scope={scope}, latest_qlib_date={latest_date}, end_date={end_date}, "
            f"source_dir={self.source_dir}, normalize_dir={self.normalize_dir}, qlib_dir={qlib_data_1d_dir}"
        )

        if update_index_instruments:
            self._refresh_index_instruments_before_update(qlib_data_1d_dir=qlib_data_1d_dir, scope=scope)

        if scope == "csi300" and not refresh_csi300_instruments_baostock and not update_index_instruments:
            logger.warning(
                "scope=csi300 is using the existing instruments/csi300.txt as-is. "
                "If that file is only a baostock snapshot rather than full historical constituents, "
                "prediction universes and scores may shift materially."
            )

        # Generate a separate baostock snapshot scope for recent-date predictions instead of
        # overwriting the historical csi300.txt file.
        if refresh_csi300_instruments_baostock and scope == "csi300":
            try:
                effective_scope = self.update_csi300_instruments_from_baostock(qlib_data_1d_dir=qlib_data_1d_dir)
                logger.warning(
                    f"using baostock snapshot scope={effective_scope} for this run; "
                    "historical csi300.txt was kept unchanged"
                )
            except Exception as e:
                fallback_scope = resolve_local_csi300_scope_fallback(qlib_data_1d_dir)
                if fallback_scope is None:
                    raise
                effective_scope = fallback_scope
                logger.warning(
                    "failed to refresh CSI300 baostock snapshot; "
                    f"falling back to existing local scope={effective_scope}: {e}"
                )

        if repair_instruments_end_by_bin:
            try:
                self._repair_instruments_end_by_bin(
                    qlib_data_1d_dir=qlib_data_1d_dir, scope=effective_scope, field=bin_end_field
                )
            except Exception as e:
                logger.warning(f"repair instruments end by bin failed: {e}")

        start_date = latest_date
        if start_from_bin_end:
            try:
                start_date = self._infer_bin_start_date(
                    qlib_data_1d_dir=qlib_data_1d_dir,
                    scope=effective_scope,
                    field=bin_end_field,
                )
                logger.info(f"override start_date from bin end: {start_date}")
            except Exception as e:
                logger.warning(f"fallback to latest_date={latest_date}, failed to infer bin end: {e}")
        if rebuild_normalize_from_source:
            start_date = self._infer_source_start_date(effective_scope, qlib_data_1d_dir)
            logger.info(f"rebuild normalize from source; start_date aligned to source coverage: {start_date}")

        if not skip_download:
            self.download_data(
                delay=delay,
                start=start_date,
                end=end_date,
                check_data_length=check_data_length,
                qlib_data_1d_dir=qlib_data_1d_dir,
                instrument_scope=effective_scope,
                include_index_symbols=include_index_symbols,
                source_skip_existing=source_skip_existing,
                normalize_dir=str(self.normalize_dir),
            )
        else:
            logger.info("skip download step by request")

        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        if not skip_normalize:
            self.normalize_data_1d_extend(
                qlib_data_1d_dir,
                qlib_instrument_market=effective_scope,
                normalize_skip_existing=(False if rebuild_normalize_from_source else normalize_skip_existing),
                request_end_exclusive=end_date,
            )
        else:
            logger.info("skip normalize step by request")

        dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        dump.dump()
        if rewrite_scope_bins_from_normalize:
            self._rewrite_scope_bins_from_normalize(
                qlib_data_1d_dir=qlib_data_1d_dir,
                scope=effective_scope,
            )

    @staticmethod
    def _index_names_for_scope(scope: str) -> List[str]:
        if scope == "csi300":
            return ["CSI300"]
        if scope == "csi100":
            return ["CSI100"]
        if scope == "csi500":
            return ["CSI500"]
        return ["CSI100", "CSI300", "CSI500"]

    def _refresh_index_instruments_before_update(self, qlib_data_1d_dir: str, scope: str) -> None:
        if scope == CSI300_BAOSTOCK_SNAPSHOT_SCOPE:
            return
        for index_name in self._index_names_for_scope(scope):
            try:
                get_instruments(qlib_data_1d_dir, index_name, market_index="cn_index")
                logger.info(f"refreshed historical index constituents before update: {index_name}")
            except Exception as e:
                logger.warning(f"skip index refresh {index_name}: {e}")

    def _infer_source_start_date(self, scope: str, qlib_data_1d_dir: str) -> str:
        instrument_path = Path(qlib_data_1d_dir).joinpath("instruments", f"{scope}.txt" if scope != "all" else "all.txt")
        if not instrument_path.exists():
            instrument_path = Path(qlib_data_1d_dir).joinpath("instruments", "all.txt")
        inst_df = pd.read_csv(instrument_path, sep="\t", header=None, usecols=[0], names=["symbol"])
        starts = []
        for symbol in inst_df["symbol"].astype(str):
            fname = self.source_dir.joinpath(f"{code_to_fname(symbol.lower())}.csv")
            if not fname.exists():
                continue
            try:
                src = pd.read_csv(fname, usecols=["date"], low_memory=False)
                if src.empty:
                    continue
                d = pd.to_datetime(src["date"], errors="coerce").min()
                if pd.notna(d):
                    starts.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
            except Exception:
                continue
        if starts:
            return min(starts)
        calendar_path = Path(qlib_data_1d_dir).joinpath("calendars/day.txt")
        first_date = pd.read_csv(calendar_path, header=None).iloc[0, 0]
        return pd.Timestamp(first_date).strftime("%Y-%m-%d")

    def rebuild_normalize_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        instrument_scope: str = None,
        include_index_symbols: bool = None,
        update_index_instruments: bool = False,
        refresh_csi300_instruments_baostock: bool = False,
        skip_download: bool = True,
    ):
        """
        Rebuild normalized csv files against the current qlib release base, then dump to bins.

        This is intended for periodic realignment after replacing ``qlib_bin`` with a new published release.
        It keeps baostock as the raw-data source and treats the existing qlib directory as the pricing scale base.
        """
        self.update_data_to_bin(
            qlib_data_1d_dir=qlib_data_1d_dir,
            end_date=end_date,
            instrument_scope=instrument_scope,
            include_index_symbols=include_index_symbols,
            source_skip_existing=True,
            normalize_skip_existing=False,
            update_index_instruments=update_index_instruments,
            refresh_csi300_instruments_baostock=refresh_csi300_instruments_baostock,
            skip_download=skip_download,
            rebuild_normalize_from_source=True,
            rewrite_scope_bins_from_normalize=True,
        )

    def _rewrite_scope_bins_from_normalize(self, qlib_data_1d_dir: str, scope: str) -> None:
        """
        Patch feature bin tails for every symbol in *scope* using the current normalized csv files.

        This preserves the existing long-history prefix from qlib release data and only overwrites
        the tail range covered by local normalized csv files. It avoids stale append-only tails when
        a symbol was skipped by a previous narrow update and later gets reintroduced into the active
        prediction universe.
        """
        import struct

        qlib_dir = Path(qlib_data_1d_dir).expanduser().resolve()
        scope_path = qlib_dir.joinpath("instruments", f"{scope}.txt" if scope != "all" else "all.txt")
        if not scope_path.exists():
            raise FileNotFoundError(f"scope instrument file not found: {scope_path}")

        dump = DumpDataUpdate(
            csv_path=self.normalize_dir,
            qlib_dir=str(qlib_dir),
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        calendar_list = dump._new_calendar_list
        inst_df = pd.read_csv(scope_path, sep="\t", header=None, usecols=[0], names=["symbol"])
        symbols = [str(s).strip().upper() for s in inst_df["symbol"] if str(s).strip()]

        rewritten = 0
        skipped = 0
        logger.info(f"rewrite scope bins from normalize csv: scope={scope}, symbols={len(symbols)}")
        for symbol in symbols:
            csv_path = self.normalize_dir.joinpath(f"{code_to_fname(symbol.lower())}.csv")
            if not csv_path.exists():
                skipped += 1
                continue
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
                if df.empty:
                    skipped += 1
                    continue
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
                df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
                if df.empty:
                    skipped += 1
                    continue
                if "vwap" not in df.columns and {"amount", "volume"}.issubset(df.columns):
                    amount = pd.to_numeric(df["amount"], errors="coerce")
                    volume = pd.to_numeric(df["volume"], errors="coerce")
                    df["vwap"] = amount / volume.replace(0, np.nan) / 100.0
                feature_dir = qlib_dir.joinpath("features", code_to_fname(symbol).lower())
                feature_dir.mkdir(parents=True, exist_ok=True)

                patch_start = pd.Timestamp(df["date"].min()).normalize()
                aligned = dump.data_merge_calendar(df.copy(), calendar_list)
                if aligned.empty:
                    skipped += 1
                    continue
                aligned = aligned.sort_index()
                patch_start_idx = dump.get_datetime_index(aligned, calendar_list)

                for field in dump.get_dump_fields(aligned.columns):
                    if field not in aligned.columns:
                        continue
                    bin_path = feature_dir.joinpath(f"{field.lower()}.{dump.freq}{dump.DUMP_FILE_SUFFIX}")
                    new_vals = np.array(aligned[field], dtype="<f")

                    if bin_path.exists():
                        raw = bin_path.read_bytes()
                        if len(raw) >= 8:
                            arr = struct.unpack("<" + "f" * (len(raw) // 4), raw)
                            old_start_idx = int(arr[0])
                            old_vals = np.array(arr[1:], dtype="<f")
                            keep_len = max(patch_start_idx - old_start_idx, 0)
                            prefix_vals = old_vals[:keep_len]
                            write_start_idx = old_start_idx if keep_len > 0 else patch_start_idx
                            merged_vals = (
                                np.concatenate([prefix_vals, new_vals]).astype("<f")
                                if keep_len > 0
                                else new_vals
                            )
                        else:
                            write_start_idx = patch_start_idx
                            merged_vals = new_vals
                    else:
                        write_start_idx = patch_start_idx
                        merged_vals = new_vals

                    np.hstack([write_start_idx, merged_vals]).astype("<f").tofile(str(bin_path.resolve()))
                rewritten += 1
            except Exception as e:
                logger.warning(f"rewrite bin failed for {symbol}: {e}")
        logger.info(f"rewrite scope bins done: rewritten={rewritten}, skipped={skipped}")

    @staticmethod
    def _infer_bin_start_date(qlib_data_1d_dir: str, scope: str, field: str = "close") -> str:
        """
        Infer a safe incremental start date based on existing bin tails.

        This prevents gaps when the bin files are behind the calendar/instruments end date.
        """
        import struct

        qlib_dir = Path(qlib_data_1d_dir)
        cal_path = qlib_dir.joinpath("calendars", "day.txt")
        cal = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not cal:
            raise ValueError("calendar is empty")

        inst_path = qlib_dir.joinpath("instruments", f"{scope}.txt" if scope != "all" else "all.txt")
        if not inst_path.exists():
            inst_path = qlib_dir.joinpath("instruments", "all.txt")
        inst_df = pd.read_csv(inst_path, sep="\t", header=None, names=["symbol", "start", "end"])
        if inst_df.empty:
            raise ValueError("instruments is empty")

        last_dates = []
        for sym in inst_df["symbol"].astype(str):
            sym = sym.strip().upper()
            if not sym:
                continue
            bin_path = qlib_dir.joinpath("features", sym.lower(), f"{field}.day.bin")
            if not bin_path.exists():
                continue
            data = bin_path.read_bytes()
            if len(data) < 8:
                continue
            date_index = int(struct.unpack("<f", data[:4])[0])
            n = len(data) // 4
            last_idx = date_index + (n - 2)
            if 0 <= last_idx < len(cal):
                last_dates.append(cal[last_idx])

        if not last_dates:
            raise ValueError("no valid bin tails found")
        # use the minimum tail to avoid gaps across symbols
        return min(last_dates)

    @staticmethod
    def _repair_instruments_end_by_bin(qlib_data_1d_dir: str, scope: str, field: str = "close") -> None:
        """
        Align instruments/all.txt end dates with actual bin tails to keep dump_update incremental.
        """
        import struct

        qlib_dir = Path(qlib_data_1d_dir)
        cal_path = qlib_dir.joinpath("calendars", "day.txt")
        cal = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not cal:
            raise ValueError("calendar is empty")

        scope_path = qlib_dir.joinpath("instruments", f"{scope}.txt" if scope != "all" else "all.txt")
        if not scope_path.exists():
            scope_path = qlib_dir.joinpath("instruments", "all.txt")
        scope_df = pd.read_csv(scope_path, sep="\t", header=None, names=["symbol", "start", "end"])
        scope_set = set(scope_df["symbol"].astype(str).str.strip())

        all_path = qlib_dir.joinpath("instruments", "all.txt")
        all_df = pd.read_csv(all_path, sep="\t", header=None, names=["symbol", "start", "end"])

        def _bin_last(sym: str) -> Optional[str]:
            bin_path = qlib_dir.joinpath("features", sym.lower(), f"{field}.day.bin")
            if not bin_path.exists():
                return None
            data = bin_path.read_bytes()
            if len(data) < 8:
                return None
            date_index = int(struct.unpack("<f", data[:4])[0])
            n = len(data) // 4
            last_idx = date_index + (n - 2)
            if 0 <= last_idx < len(cal):
                return cal[last_idx]
            return None

        changed = 0
        for i, row in all_df.iterrows():
            sym = str(row["symbol"]).strip().upper()
            if sym not in scope_set:
                continue
            last = _bin_last(sym)
            if last and str(row["end"]) != last:
                all_df.at[i, "end"] = last
                changed += 1

        if changed:
            backup = all_path.with_suffix(".txt.bak")
            all_path.replace(backup)
            all_df.to_csv(all_path, sep="\t", header=False, index=False)
            logger.info(f"repaired instruments/all.txt end dates from bin tails, changed={changed}")

    def update_csi300_instruments_from_baostock(
        self,
        qlib_data_1d_dir: str,
        as_of_date: str = None,
        lookback_calendar_days: int = 30,
        scope_name: str = CSI300_BAOSTOCK_SNAPSHOT_SCOPE,
    ):
        """
        Write a baostock-only CSI300 snapshot scope using ``query_hs300_stocks`` only.

        Notes
        -----
        - Does **not** use cn_index / Eastmoney / csindex (avoids RemoteDisconnected there).
        - Writes a **single membership segment** per symbol into ``instruments/<scope_name>.txt``:
          ``start`` = CSI300_INSTRUMENT_START, ``end`` = *as_of_date* (or last line of
          ``calendars/day.txt``). This is enough for recent ``D.list_instruments`` but **not**
          a full historical constituent history.

        Examples
        --------
        $ python collector.py update_csi300_instruments_from_baostock --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
        $ python collector.py update_csi300_instruments_from_baostock --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --as_of_date 2026-03-20
        """
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        inst_dir = Path(qlib_data_1d_dir).joinpath("instruments")
        inst_dir.mkdir(parents=True, exist_ok=True)
        target = inst_dir.joinpath(f"{scope_name}.txt")

        cal_path = Path(qlib_data_1d_dir).joinpath("calendars", "day.txt")
        if not cal_path.exists():
            raise FileNotFoundError(f"Missing calendar: {cal_path}")
        cal_lines = [x.strip() for x in cal_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not cal_lines:
            raise ValueError("calendars/day.txt is empty")

        if as_of_date is None:
            source_latest = infer_latest_trade_date_from_source_dir(
                self.source_dir,
                instrument_scope_file=inst_dir.joinpath("all.txt"),
            )
            as_of_date = source_latest or cal_lines[-1]

        try:
            login_result = bs_login_with_retry()
        except Exception as e:
            raise RuntimeError(f"baostock.login failed: {e}") from e
        if getattr(login_result, "error_code", None) != "0":
            raise RuntimeError(f"baostock.login failed: {getattr(login_result, 'error_msg', 'unknown error')}")
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
            try:
                bs.logout()
            except Exception:
                pass

        if not symbols:
            raise ValueError(
                f"No HS300 constituents from baostock for {as_of_date} (tried last {lookback_calendar_days} calendar days)."
            )

        if target.exists():
            bak = inst_dir.joinpath(f"{scope_name}.txt.bak")
            shutil.copy2(target, bak)
            logger.info(f"Backed up previous {scope_name}.txt to {bak}")

        end = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
        body = "".join(f"{sym}\t{CSI300_INSTRUMENT_START}\t{end}\n" for sym in symbols)
        target.write_text(body, encoding="utf-8")
        logger.info(
            f"Wrote {len(symbols)} CSI300 lines to {target} "
            f"(segment {CSI300_INSTRUMENT_START} ~ {end})"
        )

        meta = {
            "generated_at": datetime.datetime.now().isoformat(),
            "source": "baostock.query_hs300_stocks",
            "scope_name": scope_name,
            "as_of_date": end,
            "segment_start": CSI300_INSTRUMENT_START,
            "symbol_count": len(symbols),
            "historical_mode": "single_snapshot_expanded_to_one_segment",
            "warning": (
                "This file is a baostock snapshot expanded to one full segment per symbol. "
                "It is suitable for recent-date universe selection, but it is not a full "
                "historical CSI300 constituent history."
            ),
        }
        meta_path = inst_dir.joinpath(f"{scope_name}.baostock_meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"Wrote baostock snapshot metadata to {meta_path}")
        return scope_name


if __name__ == "__main__":
    fire.Fire(Run)
