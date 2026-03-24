#!/usr/bin/env python3
"""
TradingView conversion: MACD Trend Enhancement Strategy
Compatibility: partial
Source Pine: 2o1g6Qo5.pine

Adaptation:
- Pine's stop logic is translated into next-bar signal changes.
- Intrabar stop fills are approximated with bar-level trigger detection.
"""

import numpy as np
import pandas as pd

name = "tv_macd_trend_enhancement"
timeframe = "1h"
leverage = 1.0

FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_SMOOTHING = 9
STOP_LOSS_PCT = 0.02
BE_TRIGGER_PCT = 0.015
TRAIL_OFFSET_PCT = 0.01


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)


def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    open_ = prices["open"].to_numpy(dtype=np.float64)
    high = prices["high"].to_numpy(dtype=np.float64)
    low = prices["low"].to_numpy(dtype=np.float64)
    close = prices["close"].to_numpy(dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    fast_ema = _ema(close, FAST_LENGTH)
    slow_ema = _ema(close, SLOW_LENGTH)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, SIGNAL_SMOOTHING)

    gold_cross = _crossover(macd_line, signal_line)
    death_cross = _crossunder(macd_line, signal_line)
    trailing_high = (
        pd.Series(high)
        .rolling(window=5, min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
    )

    signals = np.zeros(n, dtype=np.float64)

    in_position = False
    pending_entry = False
    activate_at = -1
    entry_bar = -1
    entry_price = np.nan
    reached_be = False

    for i in range(n):
        if pending_entry and i == activate_at:
            in_position = True
            pending_entry = False
            entry_bar = i
            entry_price = open_[i]
            reached_be = False

        target = 1.0 if (in_position or pending_entry) else 0.0

        if in_position:
            if high[i] >= entry_price * (1.0 + BE_TRIGGER_PCT):
                reached_be = True

            initial_sl = entry_price * (1.0 - STOP_LOSS_PCT)
            current_sl = entry_price if reached_be else initial_sl
            trailing_sl = trailing_high[i] * (1.0 - TRAIL_OFFSET_PCT)
            final_sl = max(current_sl, trailing_sl)

            can_exit = i > entry_bar
            stop_hit = can_exit and low[i] <= final_sl
            trend_reversal = can_exit and death_cross[i]

            if stop_hit or trend_reversal:
                target = 0.0
                in_position = False
                entry_bar = -1
                entry_price = np.nan
                reached_be = False

        if (not in_position) and (not pending_entry) and gold_cross[i]:
            target = 1.0
            pending_entry = True
            activate_at = i + 1

        signals[i] = target

    return signals
