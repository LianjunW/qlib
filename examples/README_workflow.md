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
