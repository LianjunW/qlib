

## normal mode
```
 backtest --experiment same_label --single-model --recorder-id 9bc75beec25c442c9830551cc401c094
 ```
 结果:
 ```
 ktest loop: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 383/383 [00:01<00:00, 233.49it/s]
[3403129:MainThread](2026-02-26 17:39:05,648) INFO - qlib.workflow - [record_temp.py:515] - Portfolio analysis record 'port_analysis_1day.pkl' has been saved as the artifact of the Experiment 783647345654475464
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000510
std                0.012169
annualized_return  0.121344
information_ratio  0.646341
max_drawdown      -0.157869
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.000561
std                0.008823
annualized_return  0.133564
information_ratio  0.981262
max_drawdown      -0.101324
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.000464
std                0.008820
annualized_return  0.110453
information_ratio  0.811701
max_drawdown      -0.103797
[3403129:MainThread](2026-02-26 17:39:05,655) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 783647345654475464
'The following are analysis results of indicators(1day).'
        value
ffr  0.997567
pa   0.000000
pos  0.000000
```

## v3 流程测试：
```
python3 /root/projects/qlib/examples/workflow_by_code_v3.py  backtest --experiment same_label --single-model --recorder-id 9bc75beec25c442c9830551cc401c094

```
结果：
```
ifact of the Experiment 877120977940944901
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000510
std                0.012169
annualized_return  0.121344
information_ratio  0.646341
max_drawdown      -0.157869
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.000492
std                0.008022
annualized_return  0.117147
information_ratio  0.946638
max_drawdown      -0.091950
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.000297
std                0.008027
annualized_return  0.070606
information_ratio  0.570171
max_drawdown      -0.106783
[3404614:MainThread](2026-02-26 17:42:30,327) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 877120977940944901
'The following are analysis results of indicators(1day).'
     value
ffr    1.0
pa     0.0
pos    0.0

================================================================================
回测完成！
```


##  双模型测试：
```
python3 /root/projects/qlib/examples/workflow_by_code_v3.py  backtest --experiment same_label
```
结果：
```
/root/projects/qlib/qlib/utils/index_data.py:492: RuntimeWarning: Mean of empty slice
  return np.nanmean(self.data)
backtest loop: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 383/383 [00:01<00:00, 271.28it/s]
[3405134:MainThread](2026-02-26 17:43:29,487) INFO - qlib.workflow - [record_temp.py:515] - Portfolio analysis record 'port_analysis_1day.pkl' has been saved as the artifact of the Experiment 236826590547050849
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000510
std                0.012169
annualized_return  0.121344
information_ratio  0.646341
max_drawdown      -0.157869
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean              -0.000652
std                0.012513
annualized_return -0.155074
information_ratio -0.803345
max_drawdown      -0.322561
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean              -0.000667
std                0.012524
annualized_return -0.158646
information_ratio -0.821076
max_drawdown      -0.323997
[3405134:MainThread](2026-02-26 17:43:29,495) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 236826590547050849
'The following are analysis results of indicators(1day).'
        value
ffr  0.115922
pa   0.000000
pos  0.000000

```


##  双模型模式，实际同一模型
```
 python3 /root/projects/qlib/examples/workflow_by_code_v3.py  backtest --experiment same_label_buy --single-model
 ```
 结果：
 ```
 backtest loop: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 383/383 [00:01<00:00, 231.59it/s]
[3406773:MainThread](2026-02-26 17:50:37,764) INFO - qlib.workflow - [record_temp.py:515] - Portfolio analysis record 'port_analysis_1day.pkl' has been saved as the artifact of the Experiment 236826590547050849
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000510
std                0.012169
annualized_return  0.121344
information_ratio  0.646341
max_drawdown      -0.157869
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.000492
std                0.008022
annualized_return  0.117147
information_ratio  0.946638
max_drawdown      -0.091950
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.000297
std                0.008027
annualized_return  0.070606
information_ratio  0.570171
max_drawdown      -0.106783
[3406773:MainThread](2026-02-26 17:50:37,772) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 236826590547050849
'The following are analysis results of indicators(1day).'
     value
ffr    1.0
pa     0.0
pos    0.0

================================================================================
回测完成！
================================================================================
 ```

 ## 不同模型(diff_label)
 train
 ```
 python3 /root/projects/qlib/examples/workflow_by_code_v3.py train --model-type mlp --experiment diff_label
 ```
 test：
 ```
 python3 /root/projects/qlib/examples/workflow_by_code_v3.py backtest --experiment diff_label
 ```
result:
```
3411011:MainThread](2026-02-26 18:08:39,737) INFO - qlib.workflow - [record_temp.py:515] - Portfolio analysis record 'port_analysis_1day.pkl' has been saved as the artifact of the Experiment 629338130236322863
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000510
std                0.012169
annualized_return  0.121344
information_ratio  0.646341
max_drawdown      -0.157869
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean              -0.000512
std                0.012009
annualized_return -0.121901
information_ratio -0.657973
max_drawdown      -0.305883
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean              -0.000530
std                0.012010
annualized_return -0.126031
information_ratio -0.680242
max_drawdown      -0.305883
[3411011:MainThread](2026-02-26 18:08:39,744) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 629338130236322863
'The following are analysis results of indicators(1day).'
        value
ffr  0.127226
pa   0.000000
pos  0.000000

================================================================================
回测完成！
=========================================================
``` 

## union constituent + lower cost backtrader bridge
时间：
`2026-03-14 22:01`

调整：
`workflow_backtrader_bridge.py` / `workflow_by_code_v2.py` 中回测成本使用更低配置：
`open_cost=0.0001`, `close_cost=0.0001`, `min_cost=1`

命令：
```bash
.venv/bin/python examples/workflow_backtrader_bridge.py \
  --recorder-id e8a9a60ffcfd4788976ab061547aba68 \
  --plot-output backtest_log/union_pred_backtrader_trade100.html
```

结果：
```text
Recorder ID: e8a9a60ffcfd4788976ab061547aba68
Experiment: workflow
Date range: 2024-01-01 -> 2025-08-01
Strategy: topk=20, n_drop=2, hold_thresh=1
Sell missing signal: False
Loaded 336 instrument feeds into backtrader

Backtrader run completed
Initial cash: 1000000.00
Final value:  1427391.65
Total return: 42.74%
Annual return: 26.38%
Max drawdown: 20.22%
Benchmark (SH000300) annual return: 12.62%
Excess annual return: 13.76%
Trading days: 383
Saved plot to backtest_log/union_pred_backtrader_trade100.html
```

结论：
在 union constituent 预测覆盖修复基础上，降低交易成本后，bridge 回测收益由此前 README 中记录的 `34.12% / 21.31%` 提升到 `42.74% / 26.38%`，同时最大回撤约为 `20.22%`。

## union constituent 扩展到最新数据集时间范围
时间：
`2026-03-14 22:14 ~ 22:25`

调整：
默认回测结束时间不再固定写死，而是读取
`~/.qlib/qlib_data/cn_data/calendars/day.txt` 最后一个交易日。
本次数据集最新日期为 `2026-03-13`，因此测试区间扩展为
`2024-01-01 ~ 2026-03-13`。

训练命令：
```bash
.venv/bin/python examples/workflow_by_code_v2.py train
```

训练结果：
```text
使用模型: LightGBM (GBDT)
使用 csi300 在 2024-01-01 ~ 2026-03-13 的成分股并集，共 348 只股票生成预测。

数据集配置信息
训练集: 2010-01-01 ~ 2021-12-31
验证集: 2022-01-01 ~ 2023-12-31
测试集: 2024-01-01 ~ 2026-03-13

训练数据集大小: 725511 条记录

{'IC': np.float64(0.016433805509203665),
 'ICIR': np.float64(0.10742484634356497),
 'Rank IC': np.float64(0.0061630005820730895),
 'Rank ICIR': np.float64(0.03945567552506563)}

训练完成！Recorder ID: a82904595add4caeaadea5e2e10903c7
```

bridge 回测命令：
```bash
python3 examples/workflow_backtrader_bridge.py \
  --recorder-id a82904595add4caeaadea5e2e10903c7 \
  --plot-output backtest_log/union_pred_backtrader_trade100_test_to20260313.html
```

bridge 回测结果：
```text
Recorder ID: a82904595add4caeaadea5e2e10903c7
Experiment: workflow
Date range: 2024-01-01 -> 2026-03-13
Strategy: topk=20, n_drop=2, hold_thresh=1
Sell missing signal: False
Loaded 348 instrument feeds into backtrader

Backtrader run completed
Initial cash: 1000000.00
Final value:  2107457.36
Total return: 110.75%
Annual return: 42.64%
Max drawdown: 21.58%
Benchmark (SH000300) annual return: 16.57%
Excess annual return: 26.07%
Trading days: 529
Saved plot to backtest_log/union_pred_backtrader_trade100_test_to20260313.html
```

结论：
在保持 `topk=20`、`n_drop=2`、`hold_thresh=1` 和低交易成本配置不变的前提下，
将 union constituent 预测与 bridge 回测时间范围扩展到 `2026-03-13` 后，
新 recorder `a82904595add4caeaadea5e2e10903c7` 的 bridge 回测结果达到
`总收益 110.75%`、`年化 42.64%`、`超额年化 26.07%`，覆盖股票数增至 `348` 只。
