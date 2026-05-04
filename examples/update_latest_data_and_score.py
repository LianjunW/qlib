#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alias entrypoint for the unified latest daily update + score workflow."""

import sys
from pathlib import Path

_ex = Path(__file__).resolve().parent
if str(_ex) not in sys.path:
    sys.path.insert(0, str(_ex))

from baostock_daily_update_full_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
