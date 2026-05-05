# Qlib Workflow 使用文档

两个核心脚本：`workflow_by_code_v2.py`（主入口）和 `workflow_backtrader_bridge.py`（backtrader 回测引擎）。

## 快速开始

```bash
cd examples/

# 1. 训练模型（自动运行 Qlib 原生回测）
python workflow_by_code_v2.py train gbdt

# 2. 用 backtrader 引擎回测（输出 NAV 曲线）
python workflow_by_code_v2.py bridge --plot-output nav.html
```

训练只需做一次；之后调参、换回测引擎、换时间段均不需要重新训练。

当前本地 Qlib 日历最新交易日为 `2026-04-10`。如果你已经有训练好的 recorder，优先用 `predict` 复用 `params.pkl` 扩展到最新日期，而不是重新训练。

已验证可复用的源模型 recorder:
- `workflow/a82904595add4caeaadea5e2e10903c7`：已有训练权重 `params.pkl`
- `workflow_latest/ece683722ed74ac19fce330615094098`：基于上面的模型扩展预测到 `2026-04-10` 后生成
- `workflow_latest_refresh/5ceac4f6c45e4cde89e6fdcd45950755`：在 `2026-04-12` 发布的数据包上重新生成预测并回测后的最新 recorder

```bash
cd examples/

# 查看已有训练产物
python workflow_by_code_v2.py list --experiment workflow

# 基于已训练模型扩展到最新日期，不重新训练
python workflow_by_code_v2.py predict \
  --experiment workflow \
  --recorder-id a82904595add4caeaadea5e2e10903c7 \
  --start 2024-01-01 --end 2026-04-10 \
  --new-experiment workflow_latest

# 如需 backtrader NAV 曲线，再跑 bridge
python workflow_by_code_v2.py bridge \
  --experiment workflow_latest \
  --recorder-id ece683722ed74ac19fce330615094098 \
  --plot-output backtest_log/workflow_latest_nav_2026-04-10.html
```

已生成 NAV 图：`examples/backtest_log/workflow_latest_nav_2026-04-10.html`

## 调试记录

### 基于最新 Qlib 数据包的 backtest vs bridge 对比

测试对象统一为：
- 数据包：`https://github.com/chenditc/investment_data/releases/download/2026-04-12/qlib_bin.tar.gz`
- 实验：`workflow_latest_refresh`
- Recorder：`5ceac4f6c45e4cde89e6fdcd45950755`
- 时间区间：`2024-01-01 ~ 2026-04-10`
- 策略参数：`topk=20`、`n_drop=2`、`hold_thresh=1`
- 预测股票池：`348` 只（相较旧数据下的 `300` 只明显扩大）

执行命令：

```bash
python workflow_by_code_v2.py predict \
  --experiment workflow \
  --recorder-id a82904595add4caeaadea5e2e10903c7 \
  --start 2024-01-01 --end 2026-04-10 \
  --new-experiment workflow_latest_refresh

python workflow_by_code_v2.py backtest \
  --experiment workflow_latest_refresh \
  --recorder-id 5ceac4f6c45e4cde89e6fdcd45950755 \
  --start 2024-01-01 --end 2026-04-10

python workflow_by_code_v2.py bridge \
  --experiment workflow_latest_refresh \
  --recorder-id 5ceac4f6c45e4cde89e6fdcd45950755 \
  --start 2024-01-01 --end 2026-04-10 \
  --plot-output backtest_log/workflow_latest_refresh_bridge_2026-04-10.html
```

实测结果：

| 指标 | Qlib `backtest` | Backtrader `bridge` |
|------|------------------|---------------------|
| 组合累计收益 | `122.76%` | `116.45%` |
| 组合年化收益 | `44.63%` | `42.63%` |
| 最大回撤 | `23.83%` | `21.58%` |
| 基准年化收益 | `14.88%` | `15.58%` |
| 超额年化收益 | `25.26%`（含成本） | `27.06%` |

说明：
- `backtest` 终端默认打印的是超额收益指标，不是组合绝对收益；上表中的 Qlib 组合累计收益、组合年化收益、最大回撤，是从 `report_normal_1day.pkl` 的账户曲线反推出来的。
- `bridge` 结果来自 Backtrader 的真实持仓/现金曲线，输出图保存在 `examples/backtest_log/workflow_latest_refresh_bridge_2026-04-10.html`。
- `bridge` 的累计收益和年化收益仍略低于 `backtest`，主要因为它加入了 A 股 `100` 股整手、现金约束和 `95%` 仓位限制，更接近实盘执行。
- 两边基准年化只剩小幅差异，说明在最新数据包下，两种回测器的结果方向和量级已经更接近。
- 这轮结果与旧调试记录差异很大，核心原因不是代码改动，而是新数据包将预测股票池从 `300` 只扩展到 `348` 只，同时 benchmark 序列也发生了变化。

---

## workflow_by_code_v2.py

统一入口，7 个子命令。

### train — 训练模型

```bash
python workflow_by_code_v2.py train [gbdt|mlp|mlp_deep] [--experiment NAME]
```

- 训练模型，生成 `pred.pkl`（预测）、`label.pkl`、`params.pkl`（模型权重）
- 预测范围自动扩展为回测区间内的成分股并集（避免退出指数的股票丢失信号）
- 训练完成后自动运行一次 Qlib 原生回测

### backtest — Qlib 原生回测

```bash
python workflow_by_code_v2.py backtest \
  [--recorder-id ID] [--experiment NAME] \
  [--topk 20] [--n-drop 2] [--hold-thresh 1] \
  [--start 2024-01-01] [--end 2026-04-10]
```

- 使用 `TopkDropoutStrategy`，基于 `PortAnaRecord` 运行
- `--recorder-id` 不指定时自动选择最新的含 `pred.pkl` 的 recorder
- 输出：benchmark 收益、超额收益（含/不含成本）、最大回撤等

### bridge — Backtrader 回测

```bash
python workflow_by_code_v2.py bridge \
  [--recorder-id ID] [--experiment NAME] \
  [--topk 20] [--n-drop 2] [--hold-thresh 1] \
  [--start 2024-01-01] [--end 2026-04-10] \
  [--plot-output result.html] [--verbose]
```

- 调用 `workflow_backtrader_bridge.py`，使用 backtrader 引擎运行回测
- 支持 A 股手数（100 股整手）、佣金模拟
- `--plot-output` 输出策略 vs 基准的 NAV 曲线图（`.html` 或 `.png`）

### predict — 为新时间段生成预测

```bash
python workflow_by_code_v2.py predict \
  --start 2024-01-01 --end 2026-04-10 \
  [--recorder-id ID] [--experiment NAME] [--new-experiment NAME]
```

- 从已有 recorder 加载模型，为新时间段生成 `pred.pkl`
- 适合把已有模型扩展到数据集最新交易日，生成后自动运行回测

### layer — 分层收益分析

```bash
python workflow_by_code_v2.py layer \
  [--recorder-id ID] [--experiment NAME] \
  [--n-groups 5] [--output-dir ./layer_analysis]
```

- 基于 `pred.pkl` + `label.pkl`，按分数分为 N 组
- 输出：分组累计收益曲线、IC、自相关等 HTML 图表

### analyze — GBDT 因子重要性

```bash
python workflow_by_code_v2.py analyze \
  [--recorder-id ID] [--experiment NAME] \
  [--top-n 10] [--output chart.html]
```

- 仅支持 GBDT 模型
- 输出 Top-N 因子名称、重要度、Alpha158 语义描述

### list — 列出 Recorder

```bash
python workflow_by_code_v2.py list [--experiment NAME]
```

- 显示实验中所有 recorder 的 ID、创建时间、`pred.pkl` 状态

---

## workflow_backtrader_bridge.py

也可独立使用，参数更丰富：

```bash
python workflow_backtrader_bridge.py \
  [--experiment-name workflow] [--recorder-id ID] \
  [--topk 20] [--n-drop 2] [--hold-thresh 1] \
  [--start-time 2024-01-01] [--end-time 2026-04-10] \
  [--account 1000000] [--max-instruments 100] \
  [--sell-missing-signal] [--missing-signal-sell-after 3] \
  [--plot] [--plot-output result.html] [--verbose]
```

独有参数：

| 参数 | 说明 |
|------|------|
| `--account` | 初始资金（默认 1,000,000） |
| `--max-instruments` | 限制加载的最大股票数量 |
| `--sell-missing-signal` | 强制卖出连续缺失信号的持仓 |
| `--missing-signal-sell-after N` | 缺失信号 N 天后强制卖出（默认 3） |
| `--plot` | 显示 NAV 图（matplotlib） |

---

## Python API

```python
from workflow_by_code_v2 import (
    train_model,
    run_backtest_only,
    generate_predictions_for_new_period,
    run_layer_analysis,
    analyze_feature_importance,
)

# 训练
rid = train_model(model_type="gbdt", experiment_name="my_exp")

# 回测（修改策略参数，不重新训练）
analysis = run_backtest_only(recorder_id=rid, topk=30, n_drop=3)

# 为新时间段生成预测 + 回测
new_rid = generate_predictions_for_new_period(
    recorder_id=rid,
    test_start_time="2024-01-01",
    test_end_time="2026-04-10",
)
run_backtest_only(recorder_id=new_rid, experiment_name="my_exp_new_period")

# 分层分析
run_layer_analysis(recorder_id=rid, n_groups=5)

# 因子重要性（仅 GBDT）
fi = analyze_feature_importance(recorder_id=rid, top_n=10)
```

```python
from workflow_backtrader_bridge import run_backtrader_bridge

summary = run_backtrader_bridge(
    recorder_id=rid,
    topk=20,
    plot_output="nav.html",
)
print(f"年化收益: {summary['annual_return']:.2%}")
```

---

## 默认配置

| 配置项 | 默认值 |
|--------|--------|
| 训练集 | 2010–2014, 2016–2019, 2020.06–2021 |
| 验证集 | 2022-01-01 ~ 2023-12-31 |
| 回测起始 | 2024-01-01 |
| 回测结束 | 数据集最新交易日（自动读取） |
| 股票池 | CSI300 成分股并集 |
| 特征集 | Alpha158（158 维） |
| topk / n_drop / hold_thresh | 20 / 2 / 1 |
| 初始资金 | 1,000,000 |
| 交易成本 | 买卖各万一，最低 1 元 |

修改 `STRATEGY_CONFIG` / `BACKTEST_CONFIG` 可全局调整默认值，不需要重新训练。

---

## 典型工作流

```
train ──→ pred.pkl ──┬──→ backtest  （Qlib 原生，快速风险分析）
                     ├──→ bridge    （Backtrader，NAV 曲线 + 逐笔模拟）
                     ├──→ layer     （分层收益图）
                     └──→ analyze   （因子重要性）

predict ──→ new pred.pkl ──→ backtest / bridge（新时间段）
```

---

## Research TODO：提升 IC / TopK / Top1 能力

当前 `workflow_by_code_v2.py` 的 GBDT + Alpha158 更适合作为横截面排序信号，而不是涨跌二分类器。后续优化应优先提升 RankIC、TopK 头部排序稳定性，以及单票交易场景下的 Top1 命中质量。

### 当前开发进度

- 2026-05-05：新增 `examples/workflow_signal_diagnostics.py`，用于读取已有 qlib recorder，统一输出回测口径、IC/RankIC、TopK、Top1、score 分位数、年份拆解和 Top1 score-gap 过滤诊断。该脚本不重训模型，也不重跑回测。
- 已验证命令：

```bash
.venv/bin/python examples/workflow_signal_diagnostics.py \
  --recorder-id 45ee81fb4f68405eb91ffbac7f18573f \
  --experiment workflow \
  --start 2024-01-01 \
  --end 2026-04-10
```

- 当前 recorder `workflow/45ee81fb4f68405eb91ffbac7f18573f` 在 `2024-01-02 ~ 2026-04-10` 的诊断结果：
  - 预测覆盖：`189064` 条，`348` 只股票
  - Daily IC mean / IR：`0.017116 / 0.1117`
  - Daily RankIC mean / IR：`0.007023 / 0.0447`
  - RankIC 为正天数比例：`50.55%`
  - AUC(`label > 0`)：`0.5099`
  - 组合累计收益 / 年化收益：`116.75% / 42.72%`
  - 含成本超额年化 / 最大回撤：`24.03% / -16.56%`
  - Top20 平均收益 / Top-Bottom spread：`0.1883% / 0.1751%`
  - Top20 跑赢全池 / 跑赢 Bottom20 天数比例：`54.38% / 54.74%`
  - Top20 简单信号路径年化 / 最大回撤：`53.53% / -19.99%`
  - Top1 平均收益 / 中位收益：`0.3305% / -0.1390%`
  - Top1 正收益天数比例：`48.18%`
  - Top1 日换手：`81.02%`
  - Top1 简单信号路径年化 / 最大回撤：`96.30% / -39.67%`
  - Top1 `score_gap_2` p70 过滤覆盖率 / 平均收益 / 正收益天数比例：`30.11% / 0.6555% / 52.12%`
  - Top1 `score_gap_2` p90 过滤后表现转差：覆盖率 `10.04%`，平均收益 `-0.1732%`
  - 年份拆解 RankIC：2024 `0.006189`，2025 `0.005988`，2026 `0.014220`

说明：`workflow_signal_diagnostics.py` 中的 TopK/Top1 `path_ann` 和 `path_mdd` 是基于 `label` 的简单等权信号路径估算，用于比较信号形态；它不是 `PortAnaRecord` 或 Backtrader 的真实交易回测，不包含换手成本、涨跌停、停牌、整手和现金约束。

### 1. 建立回归与验证框架

- [x] 固定 recorder、回测区间和交易成本，建立第一版可重复 benchmark 诊断入口。
- [x] 区分组合绝对收益、基准收益、超额收益，避免把 `report_normal_1day.pkl` 反推的组合收益和 `port_analysis_1day.pkl` 的超额收益混用。
- [ ] 增加 walk-forward 验证：滚动训练、验证、测试，避免只看单一区间。
- [x] 增加基础 IC、RankIC、TopK spread、Top1 指标的一键诊断。
- [x] 按年份拆解 IC、RankIC、TopK spread。
- [ ] 按牛/熊/震荡市场、行业、市值、波动率分组拆解 IC、RankIC、TopK spread。
- [ ] 做交易成本和滑点敏感性测试：万一、万三、万五，以及不同最低手续费。
- [ ] 做 TopK / `n_drop` / `hold_thresh` 网格，观察收益、回撤、换手和稳定性。

### 2. 针对 IC / RankIC 不高的实验

- [ ] 将 label 从绝对未来收益改为相对收益：个股未来收益减 CSI300、行业均值或市值/行业中性收益。
- [ ] 测试不同预测 horizon：1D、3D、5D、10D forward return，优先观察 RankIC 和 TopK spread。
- [ ] 将 label 改为每日截面 rank 或 z-score，弱化市场整体涨跌噪声。
- [ ] 对 label 做 winsorize / 去极值，降低极端收益样本对训练的干扰。
- [ ] 尝试排序目标：LightGBM `lambdarank`、pairwise ranking，或回归每日 rank-normalized label。
- [ ] 过滤低质量样本：停牌、成交额过低、涨跌停不可交易、新股上市不足 N 天、复权或 volume 异常。
- [ ] 检查 Alpha158 因子共线和噪声，删除长期低贡献或不稳定因子。

### 3. TopK 排序能力优化

- [x] 评估 Top5、Top10、Top20、Top30、Top50 的平均未来收益、跑赢全池比例、跑赢 BottomK 比例和换手。
- [x] 使用分位数组合检查 score 单调性：最高 20% 是否稳定优于最低 20%。
- [ ] 引入 score spread 过滤：当 TopK 与中位数或 BottomK 的分差不足时减少调仓或降仓。
- [ ] 对 TopK 做行业、市值、Beta、波动率、流动性约束，验证收益是否仍然存在。
- [ ] 比较日频调仓、3 日持有、5 日持有，降低换手并观察 TopK spread 是否更稳定。

### 4. 每天最多交易一支股票的 Top1 方向

- [ ] 不直接假设 Top1 可交易；单票策略需要单独验证头部排序精度。
- [x] 增加 Top1 指标：Top1 平均收益、胜率、跑赢全池、跑赢 Top20 均值、换手率。
- [x] 检查 Top1 是否落在真实未来收益前 5% / 前 10% 的命中率。
- [x] 测试 score gap 过滤：只有第一名明显领先第二名或 Top20 均值时才交易，否则空仓。
- [ ] 增加流动性、涨跌停、停牌、价格异常过滤，避免最高分股票不可执行。
- [x] 比较 Top1、Top3、Top5、小权重 TopK 的简单信号路径收益/回撤，确认集中持仓是否值得承担风险。
- [ ] 使用真实交易回测比较 Top1、Top3、Top5、小权重 TopK，纳入换手成本、涨跌停、停牌、整手和现金约束。

### 5. 因子与模型扩展

- [ ] 在验证框架稳定后再扩展因子，避免盲目堆因子导致过拟合。
- [ ] 优先尝试行业相对强弱、残差动量、流动性冲击、换手率、波动率压缩/放大等与现有 Alpha158 互补的因子。
- [ ] 如果有可靠数据源，再引入财务、估值、资金流、融资融券、北向资金等中低频因子。
- [ ] 尝试 ensemble：不同 horizon、不同训练窗口、不同因子子集、不同模型的 rank average。
- [ ] 对比 GBDT、CatBoost、XGBoost、线性模型和排序模型，重点看 out-of-sample RankIC 和 TopK/Top1 稳定性，而不是只看训练期收益。
