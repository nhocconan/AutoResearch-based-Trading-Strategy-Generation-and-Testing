#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_RSI20_80_Filter
# Hypothesis: Enter long when price breaks above 20-period Donchian high with RSI < 20 (oversold), enter short when price breaks below 20-period Donchian low with RSI > 80 (overbought).
# Uses 1d EMA50 trend filter to align with higher timeframe momentum. Designed to capture reversals in both bull and bear markets with low frequency.
# RSI extremes filter ensures entries occur during exhaustion, reducing false breakouts. Donchian breakout provides clear entry/exit levels.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Donchian_Breakout_20_RSI20_80_Filter"
timeframe = "12h"
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

    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + RSI < 20 (oversold) + price > EMA50 (uptrend filter)
            if close[i] > high_roll[i] and rsi[i] < 20 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + RSI > 80 (overbought) + price < EMA50 (downtrend filter)
            elif close[i] < low_roll[i] and rsi[i] > 80 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals