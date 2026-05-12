#!/usr/bin/env python3
# 6h_Fibonacci_Retracement_1dTrend_VolumeFilter
# Hypothesis: Price retracing to 61.8% Fibonacci level of prior 1d swing (high/low) with trend alignment and volume confirmation captures high-probability reversals.
# Uses 1d swing points to define retracement levels, enters at 61.8% level in direction of 1d trend, filtered by volume spike.
# Works in bull/bear by following 1d trend. Target: 15-25 trades/year.

name = "6h_Fibonacci_Retracement_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate 1d swing high/low for Fibonacci retracement
    # Swing high: highest high over last 20 1d bars
    # Swing low: lowest low over last 20 1d bars
    swing_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    swing_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    swing_range = swing_high - swing_low

    # Fibonacci 61.8% retracement level from swing low in uptrend, from swing high in downtrend
    fib_618_long = swing_low + 0.618 * swing_range  # 61.8% retracement from swing low
    fib_618_short = swing_high - 0.618 * swing_range  # 61.8% retracement from swing high

    # Align Fibonacci levels to 6h timeframe
    fib_618_long_aligned = align_htf_to_ltf(prices, df_1d, fib_618_long)
    fib_618_short_aligned = align_htf_to_ltf(prices, df_1d, fib_618_short)

    # Volume momentum: current > 1.8x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_momentum = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 and swing lookback warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(fib_618_long_aligned[i]) or 
            np.isnan(fib_618_short_aligned[i]) or np.isnan(volume_momentum[i]) or
            np.isnan(swing_range[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near 61.8% fib retracement (support) + 1d EMA50 uptrend + volume momentum
            if (close[i] <= fib_618_long_aligned[i] * 1.005 and  # Within 0.5% above fib level
                close[i] >= fib_618_long_aligned[i] * 0.995 and  # Within 0.5% below fib level
                close[i] > ema_50_1d_aligned[i] and 
                volume_momentum[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near 61.8% fib retracement (resistance) + 1d EMA50 downtrend + volume momentum
            elif (close[i] <= fib_618_short_aligned[i] * 1.005 and  # Within 0.5% above fib level
                  close[i] >= fib_618_short_aligned[i] * 0.995 and  # Within 0.5% below fib level
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_momentum[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above 1d EMA50 (trend exhaustion) or reaches swing high
            if close[i] > ema_50_1d_aligned[i] * 1.02 or close[i] >= swing_high[i] * 0.998:  # Near swing high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below 1d EMA50 (trend exhaustion) or reaches swing low
            if close[i] < ema_50_1d_aligned[i] * 0.98 or close[i] <= swing_low[i] * 1.002:  # Near swing low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals