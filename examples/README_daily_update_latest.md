# Daily Update And Score

This document records the verified workflow for keeping CN daily qlib data current with baostock while preserving compatibility with the official `qlib_bin` scale.

## Goal

Use:

- official `qlib_bin` as the long-history baseline
- baostock as the recent daily updater
- `examples/baostock_daily_update_full_pipeline.py` as the unified entrypoint for latest update + score
- `examples/daily_update_and_score.py` only for the final scoring stage when data is already fresh

This avoids two common problems:

- only updating the latest CSI300 snapshot instead of the whole prediction-union scope
- corrupting recent feature-bin tails when incrementally appending symbols that were skipped in earlier updates

## Latest score snapshot

**Baostock 日线日历（最新交易日）**：在仓库根目录执行：

```bash
python scripts/data_collector/baostock/collector.py print_latest_cn_trade_date_baostock
```

**刷新下面表格**：根据 baostock 返回的最后一个交易日，在 `examples/daily_report/` 中选取日期不晚于该日的最新 `YYYY-MM-DD.json`，并重写本节的标记区。

```bash
python examples/refresh_readme_daily_latest_snapshot.py
```

`examples/baostock_daily_update_full_pipeline.py` 在未指定 `--score-date` 时，会用 `min(day.txt 末行, baostock 最新交易日)` 作为打分日；增量更新会把 `--end_date` 设为「baostock 最新交易日 + 1 天」（左开右开区间上界）。可加 `--refresh-readme-snapshot` 在打分结束后自动执行上述刷新脚本。

<!-- DAILY_SCORE_SNAPSHOT_BEGIN -->

_(Generated from `examples/daily_report/2026-04-21.json`; refresh with `examples/refresh_readme_daily_latest_snapshot.py`.)_

Saved report: `examples/daily_report/2026-04-21.json` (generated `2026-04-24T10:04:59.316503`).

| Item | Value |
|------|-------|
| Score date | 2026-04-21 |
| Instruments scored | 300 |
| Market signal | NEUTRAL — Low score spread, limited alpha opportunities |
| Mean / median score | -0.0158 / -0.0164 |
| Top-K mean / bottom-K mean | 0.0172 / -0.0443 |
| Spread (top − bottom) | 0.0615 |

Top 5 by model score: SH600547 山东黄金 (0.0423), SZ300759 康龙化成 (0.0326), SH688506 百利天恒 (0.0278), SZ300433 蓝思科技 (0.0258), SZ000807 云铝股份 (0.0250).

<!-- DAILY_SCORE_SNAPSHOT_END -->


## Recommended layout

- official baseline data:
  `/root/projects/qlib/official_qlib_2026-04-17/cn_data`
- active working qlib data:
  `~/.qlib/qlib_data/cn_data`
- baostock raw csv:
  `scripts/data_collector/baostock/source`
- baostock normalized csv:
  `scripts/data_collector/baostock/normalize`

## One-shot script (data update → score)

If your goal is simply "run one command to update the latest日线数据并打分", use this entrypoint first:

```bash
/root/projects/qlib/.venv/bin/python examples/baostock_daily_update_full_pipeline.py
```

Equivalent thin wrapper (same flags):

```bash
python examples/update_latest_data_and_score.py
```

`examples/baostock_daily_update_full_pipeline.py` runs the same flow as **§2–§5** below in one command: baostock incremental `update_data_to_bin` (CSI300 scope) → build a **dynamic** prediction-union file `instruments/csi300_score_union_latest.txt` (last `day.txt` minus 120 days, matching `daily_update_and_score.py`) → `rewrite_scope_bins_from_normalize` for that union → call `daily_update_and_score.run_pipeline` and write `examples/daily_report/<date>.json`.

Common options:

| Flag | Meaning |
|------|---------|
| `--skip-collector` | Skip §2 download/normalize/dump; run union + tail rewrite + score only. |
| `--skip-tail-rewrite` | Skip bin tail rewrite from normalize (only if you are sure §4 is unnecessary). |
| `--skip-score` | Stop after data steps (no JSON report). |
| `--dry-run-score` | Run scoring and print the report; do not save JSON. |
| `--qlib-data-dir PATH` | Override default `~/.qlib/qlib_data/cn_data`. |
| `--refresh-stock-name-map` | Refresh `examples/local_stock_names.json` via baostock before scoring. |
| `--data-through-date YYYY-MM-DD` | Skip Baostock `query_trade_dates`; set incremental upper bound so bars through that calendar day are requested (`end_date` open = next day). When scoring without `--score-date`, caps the default score day at this date (no extra Baostock calendar call). |

Options passed through to scoring (same as `daily_update_and_score.py`): `--recorder-id`, `--experiment-name`, `--topk`, `--n-drop`, `--output-dir`, `--score-date`, `--stock-name-map`.

Example: refresh names and write report to a custom directory:

```bash
/root/projects/qlib/.venv/bin/python examples/baostock_daily_update_full_pipeline.py \
  --refresh-stock-name-map \
  --output-dir /root/projects/qlib/examples/daily_report
```

The manual steps in §2–§5 remain useful when you need fixed-date reproducibility (e.g. `csi300_score_union_20251218_20260417`) or to compare against an official baseline.

## 1. Prepare official baseline once

Download and unpack the official release into a persistent project directory:

```bash
mkdir -p /root/projects/qlib/official_qlib_2026-04-17/cn_data
curl -L https://github.com/chenditc/investment_data/releases/download/2026-04-17/qlib_bin.tar.gz \
  -o /root/projects/qlib/official_qlib_2026-04-17/qlib_bin.tar.gz
tar -zxvf /root/projects/qlib/official_qlib_2026-04-17/qlib_bin.tar.gz \
  -C /root/projects/qlib/official_qlib_2026-04-17/cn_data \
  --strip-components=1
```

To reset the active qlib directory back to the official baseline:

```bash
rsync -a --delete /root/projects/qlib/official_qlib_2026-04-17/cn_data/ ~/.qlib/qlib_data/cn_data/
```

## 2. Update latest baostock raw and normalized data

This step keeps baostock source and normalize data current and lets `day.txt` advance to the latest trade date.

```bash
/root/projects/qlib/.venv/bin/python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --source_dir scripts/data_collector/baostock/source \
  --normalize_dir scripts/data_collector/baostock/normalize \
  --region CN \
  --interval 1d \
  --instrument_scope csi300 \
  --source_skip_existing True \
  --normalize_skip_existing True \
  --refresh_csi300_instruments_baostock True \
  --start_from_bin_end True
```

Quick checks:

```bash
tail -n 5 ~/.qlib/qlib_data/cn_data/calendars/day.txt
tail -n 5 scripts/data_collector/baostock/source/sh600519.csv
tail -n 5 scripts/data_collector/baostock/normalize/sh600519.csv
```

## 3. Build prediction-union scope

`daily_update_and_score.py` predicts on the union of CSI300 constituents across the prediction window, not just the latest snapshot. So the recent baostock tail patch must cover that whole union.

Create the scope file from the official baseline:

```bash
/root/projects/qlib/.venv/bin/python - <<'PY'
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from pathlib import Path

qlib.init(
    provider_uri='/root/projects/qlib/official_qlib_2026-04-17/cn_data',
    region=REG_CN,
    expression_cache=None,
    dataset_cache=None,
)
union = sorted(
    D.list_instruments(
        D.instruments('csi300'),
        start_time='2025-12-18',
        end_time='2026-04-17',
        as_list=True,
        freq='day',
    )
)
out = Path('~/.qlib/qlib_data/cn_data/instruments/csi300_score_union_20251218_20260417.txt').expanduser()
out.write_text(''.join(f'{s}\t2005-04-08\t2026-04-17\n' for s in union), encoding='utf-8')
print(out, len(union))
PY
```

## 4. Apply safe tail patch from baostock normalize data

This is the key verified step.

It does **not** replace full-history bins. It preserves the official qlib prefix and only overwrites the tail range covered by current normalized csv files for the scope symbols.

```bash
/root/projects/qlib/.venv/bin/python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --source_dir scripts/data_collector/baostock/source \
  --normalize_dir scripts/data_collector/baostock/normalize \
  --region CN \
  --interval 1d \
  --instrument_scope csi300_score_union_20251218_20260417 \
  --skip_download True \
  --skip_normalize True \
  --rewrite_scope_bins_from_normalize True \
  --end_date 2026-04-19
```

What this fixes:

- symbols in the prediction union that are absent from the latest CSI300 snapshot
- stale append-only bin tails
- mismatched recent `$close`, `$factor`, and `$vwap` values after incremental updates

## 5. Run score

```bash
/root/projects/qlib/.venv/bin/python examples/daily_update_and_score.py --score-date 2026-04-21
```

Or let the script use the latest date in `day.txt`:

```bash
/root/projects/qlib/.venv/bin/python examples/daily_update_and_score.py
```

## 6. Compare with official result

Official:

```bash
/root/projects/qlib/.venv/bin/python examples/daily_update_and_score.py \
  --qlib-data-dir /root/projects/qlib/official_qlib_2026-04-17/cn_data \
  --score-date 2026-04-17 \
  --output-dir /root/projects/qlib/examples/daily_report_official
```

Patched baostock:

```bash
/root/projects/qlib/.venv/bin/python examples/daily_update_and_score.py \
  --score-date 2026-04-17 \
  --output-dir /root/projects/qlib/examples/daily_report_baostock_unionfix_tailpatch
```

Verified result from this workflow:

- total scored instruments: `311 vs 311`
- top-20 ranking: identical
- only tiny floating-point score differences remain

## Notes

- `rewrite_scope_bins_from_normalize` is intended for recent-tail repair on top of an official qlib baseline.
- Do not use recent `normalize/*.csv` to fully replace long-history bins unless those csv files themselves contain full history.
- The active qlib directory can always be restored from the official baseline with the `rsync` command above.
 history.
- The active qlib directory can always be restored from the official baseline with the `rsync` command above.
