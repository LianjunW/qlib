# Akshare Collector 调用路径文档

本文档详细说明 `update_data_to_bin` 方法的完整调用路径和执行流程。

## 调用路径概览

```
Run.update_data_to_bin()
  ├── Run.download_data()
  │     └── AkshareCollector(...).collector_data()
  │           ├── BaseCollector.collector_data()
  │           │     └── AkshareCollector._collector(instrument_list)
  │           │           └── BaseCollector._simple_collector(symbol)
  │           │                 ├── AkshareCollector.get_data()
  │           │                 │     └── AkshareCollector.get_data_from_akshare()
  │           │                 └── BaseCollector.save_instrument()
  │           └── (处理 mini_symbol_map)
  │
  ├── Run.normalize_data_1d_extend()
  │     └── Normalize.normalize()
  │           └── AkshareNormalizeCN1dExtend.normalize()
  │
  ├── DumpDataUpdate.dump()
  │
  └── get_instruments() (更新指数成分股)
```

## 详细调用链

### 1. 入口：`Run.update_data_to_bin()`

**位置**: `collector.py:955-1052`

**功能**: 一键完成数据更新流程

**主要步骤**:
1. 检查 qlib 数据是否存在，不存在则下载
2. 计算需要更新的日期范围（从最后交易日开始）
3. 调用 `download_data()` 下载新数据
4. 调用 `normalize_data_1d_extend()` 标准化数据
5. 调用 `DumpDataUpdate.dump()` 转换为 bin 格式
6. 更新指数成分股

**关键代码**:
```python
# 1. 检查并下载 qlib 数据
if not exists_qlib_data(qlib_data_1d_dir):
    GetData().qlib_data(...)

# 2. 计算日期范围
calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# 3. 下载数据
self.download_data(delay=delay, start=trading_date, end=end_date, ...)

# 4. 标准化数据
self.normalize_data_1d_extend(qlib_data_1d_dir)

# 5. 转换为 bin 格式
_dump = DumpDataUpdate(...)
_dump.dump()

# 6. 更新指数
get_instruments(...)
```

---

### 2. 数据下载：`Run.download_data()`

**位置**: `collector.py:852-892`

**功能**: 创建 `AkshareCollector` 实例并启动数据采集

**关键代码**:
```python
_class = getattr(self._cur_module, self.collector_class_name)  # 获取 AkshareCollector
_class(
    self.source_dir,
    max_workers=self.max_workers,
    max_collector_count=max_collector_count,
    delay=delay,
    start=start,
    end=end,
    interval=self.interval,
    ...
).collector_data()  # 启动采集
```

---

### 3. 采集器初始化：`AkshareCollector.__init__()`

**位置**: `collector.py:302-357`

**功能**: 初始化采集器，包括：
- 调用父类 `BaseCollector.__init__()`
- 初始化代理管理器（Clash 支持）
- 加载配置文件
- 获取股票列表

**关键步骤**:
```python
super(AkshareCollector, self).__init__(...)  # 初始化基类
self._init_proxy_manager(config_path)  # 初始化代理管理器
```

**父类初始化中的关键步骤** (`base.py:80`):
```python
# 获取股票列表并处理
self.instrument_list = sorted(set(self.get_instrument_list()))
```

**说明**:
- `self.get_instrument_list()`: 调用子类实现的抽象方法，获取股票列表
  - 对于 `AkshareCollector`，调用 `AkshareCollector.get_instrument_list()`
  - 该方法调用 `get_hs_stock_symbols()` 获取中国 A 股股票列表
- `set(...)`: 去重，确保股票代码唯一
- `sorted(...)`: 排序，保证处理顺序一致
- 最终结果存储在 `self.instrument_list` 中，供后续采集使用

**示例**:
```python
# AkshareCollector.get_instrument_list() 返回:
# ['000001.sz', '000002.sz', '600000.ss', '600001.ss', ...]

# 经过 sorted(set(...)) 处理后:
# ['000001.sz', '000002.sz', '600000.ss', '600001.ss', ...]  # 已去重并排序
```

**父类初始化中的关键步骤** (`base.py:80`):
```python
# 获取股票列表并处理
self.instrument_list = sorted(set(self.get_instrument_list()))
```

**说明**:
- `self.get_instrument_list()`: 调用子类实现的抽象方法，获取股票列表
  - 对于 `AkshareCollector`，调用 `AkshareCollector.get_instrument_list()`
  - 该方法调用 `get_hs_stock_symbols()` 获取中国 A 股股票列表
- `set(...)`: 去重，确保股票代码唯一
- `sorted(...)`: 排序，保证处理顺序一致
- 最终结果存储在 `self.instrument_list` 中，供后续采集使用

**示例**:
```python
# AkshareCollector.get_instrument_list() 返回:
# ['000001.sz', '000002.sz', '600000.ss', '600001.ss', ...]

# 经过 sorted(set(...)) 处理后:
# ['000001.sz', '000002.sz', '600000.ss', '600001.ss', ...]  # 已去重并排序
```

---

### 4. 数据采集：`BaseCollector.collector_data()`

**位置**: `base.py:200-216` (基类方法)

**功能**: 控制采集流程的主循环

**执行流程**:
```python
def collector_data(self):
    instrument_list = self.instrument_list  # 获取股票列表（已在 __init__ 中初始化）
    for i in range(self.max_collector_count):  # 最多重试 max_collector_count 次
        if not instrument_list:
            break
        instrument_list = self._collector(instrument_list)  # 采集数据，返回失败的股票
    # 处理数据量不足的股票
    for _symbol, _df_list in self.mini_symbol_map.items():
        _df = pd.concat(_df_list, sort=False)
        self.save_instrument(_symbol, _df)
```

**说明**: 
- `self.instrument_list`: 在 `__init__` 中通过 `sorted(set(self.get_instrument_list()))` 初始化
  - 已去重和排序，确保处理顺序一致
  - 如果设置了 `limit_nums`，会截取前 N 个股票（用于测试）
- `max_collector_count` 默认为 2，表示最多采集 2 轮
- 第一轮采集失败的股票会在第二轮重试
- `mini_symbol_map` 存储数据量不足的股票（需要多次采集合并）

---

### 5. 分批采集：`AkshareCollector._collector()`

**位置**: `collector.py:578-609` (重写基类方法)

**功能**: 分批采集数据，实现防封禁策略

**执行流程**:
```python
def _collector(self, instrument_list):
    error_symbol = []
    total_batches = (len(instrument_list) + self.batch_size - 1) // self.batch_size
    
    for batch_idx in range(total_batches):
        batch = instrument_list[start_idx:end_idx]  # 获取当前批次
        
        # 处理当前批次
        for symbol in batch:
            result = self._simple_collector(symbol)  # 采集单个股票
            if result != self.NORMAL_FLAG:
                error_symbol.append(symbol)
        
        # 批次间休息（防封禁）
        if batch_idx < total_batches - 1:
            time.sleep(self.batch_rest_time + random.uniform(-10, 10))
    
    return error_symbol  # 返回失败的股票列表
```

**特点**:
- **分批处理**: 将股票列表分成小批次（默认 50 只/批）
- **批次休息**: 每批完成后强制休息（默认 60 秒）
- **错误收集**: 收集失败的股票，用于重试

---

### 6. 单个股票采集：`BaseCollector._simple_collector()`

**位置**: `base.py:135-150` (基类方法)

**功能**: 采集单个股票的数据

**执行流程**:
```python
def _simple_collector(self, symbol: str):
    self.sleep()  # 延迟（防封禁）
    df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)  # 获取数据
    
    _result = self.NORMAL_FLAG
    if self.check_data_length > 0:
        _result = self.cache_small_data(symbol, df)  # 检查数据量
    
    if _result == self.NORMAL_FLAG:
        self.save_instrument(symbol, df)  # 保存数据
    
    return _result
```

**说明**:
- `check_data_length`: 如果设置了，会检查数据量是否足够
- 数据量不足的股票会暂存在 `mini_symbol_map` 中，等待合并

---

### 7. 获取数据：`AkshareCollector.get_data()`

**位置**: `collector.py:516-576`

**功能**: 获取股票数据，支持增量更新

**执行流程**:
```python
def get_data(self, symbol, interval, start_datetime, end_datetime):
    # 1. 增量更新：检查本地已有数据
    last_date = self._get_last_date(symbol)
    if last_date and last_date >= start_datetime:
        start_datetime = last_date + pd.Timedelta(days=1)  # 从最后日期+1开始
    
    # 2. 检查日期范围
    days_diff = (end_datetime - start_datetime).days
    max_retries = 1 if days_diff <= 2 else self.retry  # 小日期范围只重试1次
    
    # 3. 获取数据（带重试）
    @deco_retry(retry=max_retries, retry_sleep=random.uniform(1, 3))
    def _get_simple():
        resp = self.get_data_from_akshare(symbol, start_str, end_str)
        if resp is None or resp.empty:
            raise ValueError("empty result")
        return resp
    
    return _get_simple()
```

**特点**:
- **增量更新**: 自动检测本地已有数据，只下载缺失部分
- **智能重试**: 小日期范围（≤2天）只重试1次，避免无效重试
- **错误处理**: 空数据不记录为严重错误

---

### 8. 从 akshare 获取数据：`AkshareCollector.get_data_from_akshare()`

**位置**: `collector.py:415-514`

**功能**: 调用 akshare API 获取数据，支持代理

**执行流程**:
```python
@exponential_backoff_retry(max_retries=5, base_delay=2.0)
def get_data_from_akshare(self, symbol, start, end, adjust="qfq"):
    # 1. 动态随机延迟
    time.sleep(random.uniform(0.5, 2.0))
    
    # 2. 获取代理设置
    if self.proxy_manager:
        proxy_dict = self.proxy_manager.get_proxy()
        # 设置环境变量供 akshare 使用
        os.environ['HTTP_PROXY'] = current_proxy
        os.environ['HTTPS_PROXY'] = current_proxy
    
    # 3. 调用 akshare API
    df = ak.stock_zh_a_hist(
        symbol=symbol_code,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust=adjust
    )
    
    # 4. 数据转换
    df = df.rename(columns={...})  # 重命名列
    df["symbol"] = self.normalize_symbol(symbol)  # 添加 symbol 列
    df["date"] = pd.to_datetime(df["date"])  # 转换日期格式
    
    return df
```

**特点**:
- **防封禁**: 动态随机延迟（0.5-2.0秒）
- **代理支持**: 自动使用 Clash 或自定义代理
- **指数退避**: 失败后按指数增长延迟重试
- **错误处理**: 403 错误触发长休息（30-60秒）

---

### 9. 保存数据：`BaseCollector.save_instrument()`

**位置**: `base.py:152-173` (基类方法)

**功能**: 保存股票数据到 CSV 文件

**执行流程**:
```python
def save_instrument(self, symbol, df: pd.DataFrame):
    symbol = self.normalize_symbol(symbol)  # 标准化股票代码
    symbol = code_to_fname(symbol)  # 转换为文件名格式
    instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
    
    if instrument_path.exists():
        _old_df = pd.read_csv(instrument_path)
        df = pd.concat([_old_df, df], sort=False)  # 合并已有数据
    
    df.to_csv(instrument_path, index=False)  # 保存到文件
```

**说明**:
- 如果文件已存在，会合并新旧数据
- 支持增量更新（通过合并实现）

---

### 10. 数据标准化：`Run.normalize_data_1d_extend()`

**位置**: `collector.py:910-953`

**功能**: 标准化原始数据，用于增量更新

**执行流程**:
```python
def normalize_data_1d_extend(self, old_qlib_data_dir, ...):
    _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")  # 获取 AkshareNormalizeCN1dExtend
    yc = Normalize(
        source_dir=self.source_dir,
        target_dir=self.normalize_dir,
        normalize_class=_class,
        ...
    )
    yc.normalize()  # 执行标准化
```

**标准化类**: `AkshareNormalizeCN1dExtend`
- 继承自 `AkshareNormalizeCN1d`
- 用于增量更新，会与旧数据对齐

---

### 11. 转换为 Bin 格式：`DumpDataUpdate.dump()`

**位置**: `scripts/dump_bin.py`

**功能**: 将标准化后的 CSV 数据转换为 Qlib 的 bin 格式

**说明**:
- 读取 `normalize_dir` 中的 CSV 文件
- 转换为二进制格式并保存到 `qlib_data_1d_dir`
- 更新交易日历和股票列表

---

### 12. 更新指数成分股：`get_instruments()`

**位置**: `data_collector/cn_index/collector.py` 或 `data_collector/us_index/collector.py`

**功能**: 更新指数成分股列表（如 CSI300、CSI100）

**说明**:
- 根据 region 选择对应的 index collector
- 更新指数成分股信息

---

## 数据流向

```
1. 原始数据下载
   akshare API → AkshareCollector.get_data_from_akshare()
   → BaseCollector.save_instrument()
   → source_dir/*.csv

2. 数据标准化
   source_dir/*.csv
   → Normalize.normalize()
   → AkshareNormalizeCN1dExtend.normalize()
   → normalize_dir/*.csv

3. 转换为 Bin 格式
   normalize_dir/*.csv
   → DumpDataUpdate.dump()
   → qlib_data_1d_dir/features/*.bin

4. 更新指数
   qlib_data_1d_dir
   → get_instruments()
   → qlib_data_1d_dir/instruments/*.txt
```

## 关键参数说明

### `update_data_to_bin` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `qlib_data_1d_dir` | str | 必需 | Qlib 数据目录 |
| `end_date` | str | None | 结束日期，默认自动计算 |
| `delay` | float | 1 | 请求延迟（秒） |
| `batch_size` | int | 50 | 每批处理的股票数量 |
| `batch_rest_time` | int | 60 | 批次间休息时间（秒） |

### 防封禁参数

| 参数 | 位置 | 说明 |
|------|------|------|
| `delay` | `AkshareCollector.__init__` | 基础延迟 |
| `batch_size` | `AkshareCollector.__init__` | 批次大小 |
| `batch_rest_time` | `AkshareCollector.__init__` | 批次休息时间 |
| 随机延迟 | `get_data_from_akshare()` | 0.5-2.0 秒 |
| 指数退避 | `exponential_backoff_retry` | 失败后延迟递增 |

## 执行时间线示例

假设有 100 只股票，`batch_size=50`，`batch_rest_time=60`：

```
T+0s:   开始采集
T+0s:   处理批次 1 (50 只股票)
T+300s: 批次 1 完成，休息 60 秒
T+360s: 处理批次 2 (50 只股票)
T+660s: 批次 2 完成
T+660s: 开始标准化
T+700s: 标准化完成
T+700s: 转换为 bin 格式
T+750s: Bin 转换完成
T+750s: 更新指数
T+760s: 全部完成
```

## 错误处理流程

1. **网络错误**: 
   - 触发指数退避重试（最多 5 次）
   - 403 错误触发长休息（30-60秒）

2. **空数据**:
   - 小日期范围（≤2天）只重试 1 次
   - 记录为 DEBUG 级别，不阻塞流程

3. **数据量不足**:
   - 暂存在 `mini_symbol_map`
   - 多轮采集后合并

4. **批次失败**:
   - 失败的股票加入 `error_symbol` 列表
   - 下一轮重试

## 注意事项

1. **并发控制**: 
   - 下载阶段 `max_workers=1`（推荐）
   - 标准化阶段可以设置较大值（如 16）

2. **增量更新**:
   - 自动检测本地已有数据
   - 只下载缺失部分

3. **防封禁**:
   - 批次间强制休息
   - 动态随机延迟
   - 代理轮换

4. **数据完整性**:
   - `check_data_length` 可检查数据量
   - 数据量不足的股票会多轮采集

