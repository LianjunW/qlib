# Union-Constituent Prediction Repro

## Background

The original prediction pipeline used point-in-time `CSI300_MARKET` constituents
to generate `pred.pkl`. When a stock left the index during the backtest window,
Qlib stopped producing scores for it even though price data was still available.

This caused two side effects in `examples/workflow_backtrader_bridge.py`:

1. holdings could lose `signal_score` mid-period;
2. the `sell_missing_signal` fallback changed backtest results materially.

The fix in this change set is to generate predictions on the union of CSI300
constituents across the target backtest window, so removed constituents keep
receiving scores as long as they remain in the expanded universe and have
enough features.

## Code Changes

- `examples/workflow_by_code_v2.py`
  - add `ensure_qlib_initialized()`
  - add `get_union_instruments_for_period()`
  - add `resolve_prediction_instruments()`
  - make `train_model()` default to the union of CSI300 constituents in the
    configured backtest period
  - make `generate_predictions_for_new_period()` default to the same union logic
- `examples/workflow_backtrader_bridge.py`
  - keep the missing-signal fallback logic
  - document that it is now only a compatibility fallback, not the primary
    source of alpha once prediction coverage is continuous
- `examples/analyze_missing_signal_alpha.py`
  - attribute excess returns from keeping missing-signal holdings
- `examples/analyze_missing_signal_cases.py`
  - inspect the top stock-level missing-signal case studies
- `examples/analyze_missing_signal_reason.py`
  - split missing-signal contribution into `out_of_universe_missing` vs
    `in_universe_missing`

## Reproduction

Use the project virtual environment:

```bash
.venv/bin/python examples/workflow_by_code_v2.py train
```

Observed training recorder:

```text
e8a9a60ffcfd4788976ab061547aba68
```

The training log showed that prediction generation expanded `csi300` to the
period union:

```text
使用 csi300 在 2024-01-01 ~ 2025-08-01 的成分股并集，共 336 只股票生成预测。
```

Run the Backtrader bridge with the new recorder:

```bash
.venv/bin/python examples/workflow_backtrader_bridge.py \
  --recorder-id e8a9a60ffcfd4788976ab061547aba68 \
  --plot-output backtest_log/union_pred_backtrader.html
```

Run the missing-signal fallback comparison:

```bash
.venv/bin/python examples/workflow_backtrader_bridge.py \
  --recorder-id e8a9a60ffcfd4788976ab061547aba68 \
  --sell-missing-signal \
  --missing-signal-sell-after 20
```

Run the analysis scripts if you want to inspect the old issue or verify that
the fallback no longer matters materially:

```bash
.venv/bin/python examples/analyze_missing_signal_alpha.py
.venv/bin/python examples/analyze_missing_signal_cases.py
.venv/bin/python examples/analyze_missing_signal_reason.py
```

## Test Results

### Baseline Before Union-Pred Fix

Old recorder with point-in-time constituent predictions:

- `keep missing`: total return `25.33%`, annual return `16.02%`
- `sell missing after 20 days`: total return `20.56%`, annual return `13.09%`

This showed that missing-signal handling was materially affecting performance.

### After Union-Pred Fix

Recorder `e8a9a60ffcfd4788976ab061547aba68`:

- default bridge run:
  - total return `34.12%`
  - annual return `21.31%`
  - max drawdown `21.12%`
  - benchmark annual return `12.62%`
  - excess annual return `8.68%`
- `--sell-missing-signal --missing-signal-sell-after 20`:
  - total return `34.12%`
  - annual return `21.31%`
  - max drawdown `21.12%`
  - benchmark annual return `12.62%`
  - excess annual return `8.68%`

Result: after expanding prediction coverage to the union of constituents,
missing-signal fallback handling no longer changes the outcome.

## Analysis Conclusion

Before the fix, almost all missing-signal excess return came from
`out_of_universe_missing`, meaning holdings continued to work mainly because the
stocks left `csi300` but still had tradable price data.

After the fix, the root cause is removed upstream:

- stocks that leave the index no longer lose scores immediately;
- bridge-level missing-signal handling becomes a safety net only;
- backtest performance improves and becomes insensitive to the missing-signal
  fallback switch.
