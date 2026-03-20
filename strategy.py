#!/usr/bin/env python3
"""
strategy.py - Current Strategy Under Test
==========================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Signal convention:
    +1.0 = full long position
    -1.0 = full short position
     0.0 = flat (no position)
    Fractional values for partial positions (e.g., 0.5 = half long)

IMPORTANT: Signals must NOT use future data. Use only data up to and
including the current bar. The backtest engine handles fill delay
(signal at bar t -> fill at bar t+1).
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "sma_crossover_baseline"
timeframe = "1h"
leverage = 1.0

# Strategy parameters
FAST_PERIOD = 20
SLOW_PERIOD = 50


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Simple Moving Average Crossover - Baseline Strategy.

    Long when fast SMA > slow SMA, short when fast SMA < slow SMA.

    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]

    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    close = prices["close"].values
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)

    # Compute SMAs
    fast_sma = pd.Series(close).rolling(window=FAST_PERIOD, min_periods=FAST_PERIOD).mean().values
    slow_sma = pd.Series(close).rolling(window=SLOW_PERIOD, min_periods=SLOW_PERIOD).mean().values

    # Generate signals (only after we have enough data for both SMAs)
    for i in range(SLOW_PERIOD, n):
        if np.isnan(fast_sma[i]) or np.isnan(slow_sma[i]):
            signals[i] = 0.0
        elif fast_sma[i] > slow_sma[i]:
            signals[i] = 1.0   # Long
        elif fast_sma[i] < slow_sma[i]:
            signals[i] = -1.0  # Short
        else:
            signals[i] = 0.0   # Flat

    return signals
