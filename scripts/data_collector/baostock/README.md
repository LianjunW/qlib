# Collector Data

## Collector *Baostock* daily data to qlib

This directory adds a `baostock`-based daily updater for CN market data and keeps the command style close to `scripts/data_collector/yahoo`.

## Requirements

```bash
pip install -r scripts/data_collector/baostock/requirements.txt
```

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
python scripts/data_collector/baostock/collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d
```

### Only CSI300 (recommended for CSI300 experiments)

Uses `instruments/csi300.txt` under your qlib data dir as the symbol list (no full-market pull). By default skips re-download when the source csv already has data through `end_date - 1 day`, and does **not** refresh index constituents (avoids flaky cn_index / Eastmoney calls).

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
  --region CN --interval 1d \
  --instrument_scope csi300
```

You can also set scope once on the `Run` constructor (same effect for subcommands that use `self.instrument_scope`):

```bash
python scripts/data_collector/baostock/collector.py --instrument_scope csi300 update_data_to_bin \
  --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d
```

Optional flags:

- `--source_skip_existing False` — force re-download even if local source csv looks complete
- `--update_index_instruments True` — refresh `CSI300` membership via `cn_index` (needs network; may fail behind firewalls)

Specify an update end date with open interval semantics:

```bash
python scripts/data_collector/baostock/collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --region CN --interval 1d --end_date 2026-03-23
```

## Notes

- Currently only `CN` and `1d` are supported.
- The collector fetches raw prices with `adjustflag=3` and also fetches adjusted close with `adjustflag=2` to derive `factor`.
- Default `end_date` is `today + 1 day`, so the latest closed trading day can be included.
- `instrument_scope` must match a file `instruments/<scope>.txt` inside `qlib_data_1d_dir` (e.g. `csi300` → `instruments/csi300.txt`).
