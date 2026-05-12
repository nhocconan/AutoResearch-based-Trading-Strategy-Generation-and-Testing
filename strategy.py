#!/usr/bin/env python3
"""
12h_ThreeBar_HighLow_Breakout_1dTrend
Hypothesis: In trending markets, price tends to make consecutive higher highs (uptrend) or lower lows (downtrend). 
We identify 3-bar momentum: a new 3-bar high (higher than prior 2-bar high) in uptrend, or 
a new 3-bar low (lower than prior 2-bar low) in downtrend. 
Trend filter: 1-day EMA34. Only take longs in 1-day uptrend, shorts in downtrend.
Volume confirmation: current volume > 1.5x 20-bar average to avoid weak breakouts.
Position size: 0.25 (25% of capital) to manage drawdown. 
Exit when trend reverses or momentum fails (price fails to make new 3-bar high/low).
Designed for low trade frequency (~15-25/year) to minimize fee drag, works in both bull and bear markets via short side.
"""

name = "12h_ThreeBar_HighLow_Breakout_1dTrend"
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
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate 3-bar high and low for momentum
    # 3-bar high = max(high[i], high[i-1], high[i-2])
    high_3bar = np.maximum(np.maximum(high, np.roll(high, 1)), np.roll(high, 2))
    low_3bar = np.minimum(np.minimum(low, np.roll(low, 1)), np.roll(low, 2))
    # Prior 2-bar high/low for comparison
    high_2bar = np.maximum(high, np.roll(high, 1))
    low_2bar = np.minimum(low, np.roll(low, 1))

    # New 3-bar high > prior 2-bar high? (higher high)
    new_3bar_high = high_3bar > high_2bar
    # New 3-bar low < prior 2-bar low? (lower low)
    new_3bar_low = low_3bar < low_2bar

    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(3, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: new 3-bar high (higher high) + 1-day uptrend + volume spike
            if new_3bar_high[i] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: new 3-bar low (lower low) + 1-day downtrend + volume spike
            elif new_3bar_low[i] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend turns down or fails to make new 3-bar high
            if close[i] < ema34_1d_aligned[i] or not new_3bar_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend turns up or fails to make new 3-bar low
            if close[i] > ema34_1d_aligned[i] or not new_3bar_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals