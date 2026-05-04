# Collector Data

## Collector *Baostock* daily data to qlib

This directory adds a `baostock`-based daily updater for CN market data and keeps the command style close to `scripts/data_collector/yahoo`.

## Requirements

```bash
pip install -r scripts/data_collector/baostock/requirements.txt
```

## Recommended Use

Treat the published `qlib_bin` as the baseline dataset, and use `baostock` only for raw daily tail updates.

- Historical price scale baseline: official `qlib_bin`
- Historical CSI300/CSI100/CSI500 membership: `scripts/data_collector/cn_index/collector.py`
- Recent-date fallback universe only: baostock snapshot scope `csi300_baostock_snapshot`
- Periodic scale realignment after replacing `qlib_bin`: `rebuild_normalize_to_bin`

This avoids mixing three separate concerns into one collector:

- raw quote fetching
- historical index constituent reconstruction
- historical scale anchoring

## Full Daily Build

1. Download raw daily csv data:

```bash
python scripts/data_collector/baostock/collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_baostock_1d --interval 1d --region CN --start 2020-01-01 --end 2026-03-23
```

2. Normalize the raw csv data:

```bash
python scripts/data_collector/baostock/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_baostock_1d --normalize_dir ~/.qlib/stock_data/normalize/cn_baostock_1d --interval 1d --region CN
```

3. Dump normalized data into qlib bin format:

```bash
python scripts/dump_bin.py dump_all --csv_path ~/.qlib/stock_data/normalize/cn_baostock_1d --qlib_dir ~/.qlib/qlib_data/cn_data_baostock --freq day --exclude_fields date,symbol
```

## Incremental Update

The new command below downloads from the latest day already present in `qlib_data_1d_dir`, normalizes with overlap against the existing qlib data, then appends new bars into the bin files:

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --region CN --interval 1d
```

### Only CSI300 (recommended for CSI300 experiments)

Uses `instruments/csi300.txt` under your qlib data dir as the symbol list. For historical backtests, refresh that file from `cn_index` before the current run so the symbol universe reflects real constituent history.

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --region CN --interval 1d \
  --instrument_scope csi300 \
  --update_index_instruments True
```

You can also set scope once on the `Run` constructor (same effect for subcommands that use `self.instrument_scope`):

```bash
python scripts/data_collector/baostock/collector.py --instrument_scope csi300 update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d \
  --update_index_instruments True
```

Optional flags:

- `--source_skip_existing False` — force re-download even if local **source** or **normalize** csv (same `sh600519.csv` basename) already has daily rows through `end_date - 1` (collector `end_date` is open-interval)
- `--normalize_skip_existing False` — always re-run normalize from source (default skips per symbol when the normalize output is already fresh, so a skipped download cannot overwrite good normalize files with stale source)
- `--update_index_instruments True` — refresh `CSI300` / `CSI100` / `CSI500` history via `cn_index` **before download**, so the current run uses real historical constituents
- `--refresh_csi300_instruments_baostock True` — generate a separate baostock snapshot scope `instruments/csi300_baostock_snapshot.txt` and use that scope for the current run; `instruments/csi300.txt` stays unchanged
- `--rebuild_normalize_from_source True` — force re-run normalize from existing source csv files against the current qlib base even if there is no new trading day yet

### Periodic Realignment Against A New Published `qlib_bin`

After replacing `~/.qlib/qlib_data/cn_data` with a new published release, keep the release data as the baseline and realign your normalized baostock csv files against it:

```bash
python scripts/data_collector/baostock/collector.py rebuild_normalize_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --region CN --interval 1d \
  --instrument_scope csi300 \
  --update_index_instruments True
```

This does not rely on old baostock history still being recoverable from the vendor. It rebuilds normalized csv files from your local `source_dir` and re-dumps them using the current qlib release as the scale base.

### Refresh CSI300 constituents without `cn_index`

If `cn_index` / `parse_instruments` fails (e.g. Eastmoney closes the connection), generate a separate snapshot scope from baostock only:

```bash
python scripts/data_collector/baostock/collector.py update_csi300_instruments_from_baostock \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
```

Optional `--as_of_date YYYY-MM-DD` (default: last date in `calendars/day.txt`). The snapshot is written to `instruments/csi300_baostock_snapshot.txt` as one segment per symbol (`2005-04-08` ~ *as_of_date*), which is enough for recent `list_instruments` but **not** full historical index rebalancing.

One-shot update + constituent refresh:

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d \
  --instrument_scope csi300 --refresh_csi300_instruments_baostock True
```

Specify an update end date with open interval semantics:

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d --end_date 2026-03-23
```

## Notes

- Currently only `CN` and `1d` are supported.
- The collector fetches raw prices with `adjustflag=3` and also fetches adjusted close with `adjustflag=2` to derive `factor`.
- Default `end_date` is `today + 1 day`, so the latest closed trading day can be included.
- `instrument_scope` must match a file `instruments/<scope>.txt` inside `qlib_data_1d_dir` (e.g. `csi300` → `instruments/csi300.txt`).
- Recommended split:
  - `csi300.txt`: true historical constituents from `cn_index`
  - `csi300_baostock_snapshot.txt`: recent baostock snapshot only
