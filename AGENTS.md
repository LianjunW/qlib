# Agent 协作约定（本仓库）

本文件供自动化 Agent 与协作者默认阅读：约定工作方式，并集中列出本仓库与量化工作流相关的主要命令。回测与训练细节见 `examples/README_workflow.md`；「日线更新 + 日报 JSON」完整流程与参数见 `examples/README_daily_update_latest.md`。

## 工作方式

1. **Todo / 进度**
   - 多步骤、易遗漏或可并行拆分的任务：先列出 todo，执行过程中同步更新状态。
   - 完成一项即标记为 **completed**，不要长期保留虚假的 **in_progress**。
   - 范围变化（用户改需求）时：取消不再适用的项或改写描述，避免清单与事实脱节。

2. **完成状态与记录**
   - 实验结论、可复现命令与数字快照：写入 `examples/README_workflow.md`（如「研究 TODO」、`调试记录`）或 `examples/exp_record.md`，便于下次接续。
   - 大规模结构化日报可选：`examples/daily_report/` 下按日期 JSON（若任务需要）。

3. **代码与命令**
   - 优先在仓库内实际执行命令验证，不要只给出「待用户运行」的说明。
   - 修改范围贴合任务；不顺带大段无关重构或擅自添加未要求的文档（本 `AGENTS.md` 及用户点名要的文档除外）。

## 环境与路径

- 工作目录：仓库根目录或 `examples/`（以下命令若在 `examples/` 下执行，路径按脚本所在目录理解）。
- Python：若使用虚拟环境，可用 `.venv/bin/python`（或你本机等价路径）前缀。
- Qlib 数据：默认数据目录见各脚本 `--qlib-data-dir`（诊断脚本默认 `~/.qlib/qlib_data/cn_data`）。

## 主要命令速查

`workflow_by_code_v2.py` 等脚本习惯在 `examples/` 下执行（`cd examples`）。**Daily 日报一键脚本**在仓库根目录执行、`examples/...` 路径更显式，见下节「Daily 日报数据」。

### `workflow_by_code_v2.py`（统一入口）

| 目的 | 命令示例 |
|------|-----------|
| 训练 + 自动原生回测 | `python workflow_by_code_v2.py train gbdt` |
| 列出 recorder | `python workflow_by_code_v2.py list --experiment workflow` |
| 扩展预测到新区间（不重新训练） | `python workflow_by_code_v2.py predict --experiment workflow --recorder-id <ID> --start 2024-01-01 --end 2026-04-10 --new-experiment workflow_latest` |
| Qlib 原生回测 | `python workflow_by_code_v2.py backtest --experiment <EXP> --recorder-id <ID> [--start ... --end ...]` |
| Backtrader 回测 / NAV 图 | `python workflow_by_code_v2.py bridge --experiment <EXP> --recorder-id <ID> --plot-output backtest_log/out.html` |
| 分层收益 / IC 图 | `python workflow_by_code_v2.py layer --recorder-id <ID> [--experiment NAME]` |
| GBDT 因子重要性 | `python workflow_by_code_v2.py analyze --recorder-id <ID>` |

快速上手（训练 + bridge）：见 `examples/README_workflow.md`「快速开始」。

### `workflow_signal_diagnostics.py`（仅诊断，不重训、不重跑回测）

从已有 recorder 读 `pred`/`label`/报告，输出 IC、RankIC、TopK、Top1 等文本或 JSON：

```bash
python workflow_signal_diagnostics.py \
  --recorder-id <RECORDER_ID> \
  --experiment workflow \
  --start 2024-01-01 \
  --end 2026-04-10
```

加 `--json` 可输出 JSON。

### Daily 日报数据（`examples/daily_report/YYYY-MM-DD.json`）

产出为当日打分汇总 JSON（Top-K、spread、信号摘要等），非 `workflow_signal_diagnostics` 那种 recorder 诊断。

**一键：增量日线数据 → union 重写 → 打分写 JSON**（推荐，在仓库根目录执行）：

```bash
.venv/bin/python examples/baostock_daily_update_full_pipeline.py
```

等价薄封装（同参数透传）：

```bash
python examples/update_latest_data_and_score.py
```

常用选项（完整表格见 `examples/README_daily_update_latest.md`）：`--skip-collector`（数据已新，只做 union/打分）、`--skip-score`（只更新数据不写 JSON）、`--dry-run-score`（打印不落盘）、`--output-dir examples/daily_report`、`--recorder-id`、`--score-date`、`--refresh-readme-snapshot`（打分后刷新 README 中的快照节）。

**仅打分**（qlib 数据已是最新、不做 baostock 增量时）：

```bash
python examples/daily_update_and_score.py
python examples/daily_update_and_score.py --recorder-id <ID> --score-date 2026-04-10 --output-dir examples/daily_report
```

**用已有日报刷新文档里的「Latest score snapshot」表格**（选取 `daily_report` 下合适日期的 JSON 重写 `README_daily_update_latest.md` 标记区）：

```bash
python examples/refresh_readme_daily_latest_snapshot.py
```

### `workflow_backtrader_bridge.py`

可直接调用，参数比 `bridge` 子命令更细（资金、整手、缺失信号卖出等），详见 `examples/README_workflow.md`。

## 延伸阅读

- `examples/README_workflow.md` — 子命令说明、API 示例、调试对比表。
- `examples/README_daily_update_latest.md` — baostock 增量、`daily_report` JSON、与官方 `qlib_bin` 基线配合。
- `examples/README_union_constituent_prediction.md` — 成分预测相关说明（若使用该路径）。
