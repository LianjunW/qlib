# Akshare Data Collector

基于 Akshare 的中国 A 股数据采集器，包含完善的防封禁策略。

> 📖 **详细调用路径文档**: 查看 [CALL_PATH.md](./CALL_PATH.md) 了解 `update_data_to_bin` 的完整执行流程

## 功能特性

- ✅ 支持日线数据采集（1d）
- ✅ 增量更新：自动检测本地已有数据，只下载缺失部分
- ✅ 防封禁策略：
  - 动态随机延迟（0.5-2.0秒）
  - 代理池支持
  - 指数退避重试
  - 分批抓取（默认每批50只股票）
  - 批次间强制休息（默认60秒）
- ✅ 数据标准化：符合 Qlib 格式要求

## 安装依赖

```bash
pip install akshare pandas numpy loguru fire fake-useragent pyyaml
```

## 使用方法

### 1. 下载数据

```bash
# 基本用法
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --region CN \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --interval 1d \
    --max_workers 1 \
    --delay 0.5

# 带批次控制（推荐）
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --region CN \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --interval 1d \
    --max_workers 1 \
    --delay 0.5 \
    --batch_size 50 \
    --batch_rest_time 60

# 使用配置文件（代理等）
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --region CN \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --interval 1d \
    --config_path ./akshare_config.yaml
```

### 2. 标准化数据

```bash
python collector.py normalize_data \
    --source_dir ~/.qlib/stock_data/source \
    --normalize_dir ~/.qlib/stock_data/normalize \
    --region CN \
    --interval 1d \
    --max_workers 16
```

### 3. 转换为 Qlib 格式

#### 方式一：使用 update_data_to_bin（推荐，对齐 yahoo collector 用法）

```bash
# 一键完成：下载 -> 标准化 -> 转换为 bin 格式 -> 更新指数
python collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
```

#### 方式二：手动步骤

使用 Qlib 的 `dump_bin.py` 脚本：

```bash
python scripts/dump_bin.py dump_update \
    --csv_path ~/.qlib/stock_data/normalize \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --freq day \
    --date_field_name date \
    --symbol_field_name symbol \
    --exclude_fields symbol,date
```

## 配置说明

### 基本配置

编辑 `akshare_config.yaml` 文件可以配置：

- **proxies**: 自定义代理列表（可选）
  ```yaml
  proxies:
    - "http://127.0.0.1:7890"
    - "http://127.0.0.1:7891"
  ```

### Clash 代理集成

采集器支持自动集成 Clash/Mihomo 代理，无需手动配置：

1. **自动检测**: 采集器会自动检测以下路径的 Clash 配置：
   - `~/.config/mihomo/config.yaml`
   - `~/.config/clash/config.yaml`
   - `/root/.config/mihomo/config.yaml`
   - `/root/.config/clash/config.yaml`

2. **使用方式**:
   ```bash
   # 自动使用 Clash（默认启用）
   python collector.py download_data \
       --source_dir ~/.qlib/stock_data/source \
       --region CN \
       --start 2020-01-01 \
       --end 2020-12-31 \
       --use_clash true
   
   # 指定 Clash 配置文件路径
   python collector.py download_data \
       --source_dir ~/.qlib/stock_data/source \
       --region CN \
       --start 2020-01-01 \
       --end 2020-12-31 \
       --clash_config_path /root/.config/mihomo/config.yaml
   
   # 禁用 Clash，使用自定义代理
   python collector.py download_data \
       --source_dir ~/.qlib/stock_data/source \
       --region CN \
       --start 2020-01-01 \
       --end 2020-12-31 \
       --use_clash false
   ```

3. **代理轮换策略**:
   - `round_robin`: 轮询（默认）
   - `random`: 随机选择
   - `weighted`: 根据成功率加权选择
   
   ```bash
   python collector.py download_data \
       --source_dir ~/.qlib/stock_data/source \
       --region CN \
       --start 2020-01-01 \
       --end 2020-12-31 \
       --proxy_rotation_strategy weighted
   ```

## 防封禁策略说明

1. **动态随机延迟**: 每次请求前随机延迟 0.5-2.0 秒
2. **Clash 代理集成**: 
   - 自动检测并使用 Clash/Mihomo 本地代理
   - 支持从 Clash 配置读取代理节点信息
   - 智能代理轮换（轮询/随机/加权）
   - 自动标记失败代理并切换
3. **自定义代理池**: 支持配置多个代理，自动切换失败代理
4. **指数退避重试**: 失败后按指数增长延迟重试
5. **403 错误处理**: 遇到 403 错误时自动切换代理并长休息（30-60秒）
6. **分批抓取**: 将股票列表分成小批次，每批完成后强制休息
7. **增量更新**: 自动检测本地已有数据，只下载缺失部分
8. **低并发**: 默认 `max_workers=1`，避免高并发被封
9. **代理健康统计**: 记录代理成功率，优先使用高成功率代理

## 注意事项

- ⚠️ **强烈建议将 `max_workers` 设置为 1**，避免高并发导致封禁
- ⚠️ 如果遇到 403 错误，会自动触发长休息模式（30-60秒）
- ⚠️ 大批量下载时建议设置合理的 `batch_size` 和 `batch_rest_time`
- ⚠️ 首次下载全量数据可能需要较长时间，建议分批进行

## 参数说明

### download_data 参数

- `source_dir`: 原始数据保存目录
- `region`: 地区（CN/US/BR等），默认 CN
- `start`: 开始日期（YYYY-MM-DD）
- `end`: 结束日期（YYYY-MM-DD）
- `interval`: 数据频率，目前仅支持 1d
- `max_workers`: 并发数，**强烈建议设为 1**
- `delay`: 基础延迟时间（秒）
- `batch_size`: 每批处理的股票数量，默认 50
- `batch_rest_time`: 批次间休息时间（秒），默认 60
- `config_path`: akshare 配置文件路径
- `clash_config_path`: Clash 配置文件路径（可选，默认自动检测）
- `use_clash`: 是否使用 Clash 代理，默认 True
- `proxy_rotation_strategy`: 代理轮换策略（round_robin/random/weighted），默认 round_robin

### normalize_data 参数

- `source_dir`: 原始数据目录
- `normalize_dir`: 标准化数据保存目录
- `region`: 地区
- `interval`: 数据频率
- `max_workers`: 并发数，标准化时可以设置较大值（如 16）

### update_data_to_bin 参数（对齐 yahoo collector）

- `qlib_data_1d_dir`: qlib 数据目录（必需）
- `end_date`: 结束日期，默认自动计算
- `check_data_length`: 数据长度检查
- `delay`: 延迟时间，默认 1 秒
- `exists_skip`: 是否跳过已存在数据，默认 False
- `batch_size`: 批次大小，默认 50
- `batch_rest_time`: 批次休息时间（秒），默认 60

**示例**:
```bash
# 基本用法（对齐 yahoo collector）
python collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data

# 带参数
python collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
    --delay 0.5 \
    --batch_size 30 \
    --batch_rest_time 90
```

