# Daily Update And Score

This document records the verified workflow for keeping CN daily qlib data current with baostock while preserving compatibility with the official `qlib_bin` scale.

## Goal

Use:

- official `qlib_bin` as the long-history baseline
- baostock as the recent daily updater
- `examples/daily_update_and_score.py` for latest score generation

This avoids two common problems:

- only updating the latest CSI300 snapshot instead of the whole prediction-union scope
- corrupting recent feature-bin tails when incrementally appending symbols that were skipped in earlier updates

## Recommended layout

- official baseline data:
  `/root/projects/qlib/official_qlib_2026-04-17/cn_data`
- active working qlib data:
  `~/.qlib/qlib_data/cn_data`
- baostock raw csv:
  `scripts/data_collector/baostock/source`
- baostock normalized csv:
  `scripts/data_collector/baostock/normalize`

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
/root/projects/qlib/.venv/bin/python examples/daily_update_and_score.py --score-date 2026-04-17
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
