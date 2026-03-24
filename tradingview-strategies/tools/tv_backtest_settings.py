#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


INITIAL_CAPITAL_USD = 10_000.0
FIXED_ORDER_SIZE_USD = 1_000.0
FIXED_ORDER_FRACTION = FIXED_ORDER_SIZE_USD / INITIAL_CAPITAL_USD
POSITION_SIZING_LABEL = "fixed_10pct_of_10k"


def apply_fixed_order_size(signals: np.ndarray) -> np.ndarray:
    """Convert any directional signal stream into fixed +/-10% target exposure.

    The repo backtester interprets signal magnitude as target position fraction.
    For TradingView parity across converted strategies, every non-zero signal is
    normalized to a fixed 10% target position on a 10k account.
    """
    arr = np.asarray(signals, dtype=np.float64)
    scaled = np.zeros_like(arr, dtype=np.float64)
    scaled[arr > 0] = FIXED_ORDER_FRACTION
    scaled[arr < 0] = -FIXED_ORDER_FRACTION
    return scaled
