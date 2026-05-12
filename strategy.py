#!/usr/bin/env python3
"""
6h_LiquiditySweep_1dOrderBlock
Hypothesis: On 6h, take long when price sweeps below prior 6h low and closes back above it with bullish 1d order block (close > open) and volume spike; take short when price sweeps above prior 6h high and closes back below it with bearish 1d order block (close < open) and volume spike. Uses 1d trend filter (price > 200 EMA for long, < 200 EMA for short). Designed to work in both bull and bear markets by capturing institutional order flow around liquidity sweeps.
"""

name = "6h_LiquiditySweep_1dOrderBlock"
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
    open_price = prices['open'].values

    # Get 1d data for trend and order block
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values

    # 6-period high/low for liquidity sweep detection (prior bar)
    high_max_6 = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_min_6 = pd.Series(low).rolling(window=6, min_periods=6).min().values

    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # 1d order block: bullish if close > open, bearish if close < open
    bullish_ob = (close_1d > open_1d).astype(float)
    bearish_ob = (close_1d < open_1d).astype(float)
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob)
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob)

    # Volume confirmation: volume > 1.5x 24-period average (4 hours worth of 6m bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_6[i]) or np.isnan(low_min_6[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(bullish_ob_aligned[i]) or 
            np.isnan(bearish_ob_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: liquidity sweep below prior low + close back above + bullish 1d OB + volume spike + above 1d EMA200
            if (low[i] < low_min_6[i-1] and  # swept below prior low
                close[i] > low_min_6[i-1] and  # closed back above
                bullish_ob_aligned[i] > 0.5 and  # bullish 1d order block
                volume[i] > vol_avg_24[i] * 1.5 and  # volume spike
                close[i] > ema200_1d_aligned[i]):  # above 1d EMA200
                signals[i] = 0.25
                position = 1
            # SHORT: liquidity sweep above prior high + close back below + bearish 1d OB + volume spike + below 1d EMA200
            elif (high[i] > high_max_6[i-1] and  # swept above prior high
                  close[i] < high_max_6[i-1] and  # closed back below
                  bearish_ob_aligned[i] > 0.5 and  # bearish 1d order block
                  volume[i] > vol_avg_24[i] * 1.5 and  # volume spike
                  close[i] < ema200_1d_aligned[i]):  # below 1d EMA200
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below prior low or bearish 1d OB appears
            if low[i] < low_min_6[i] or bearish_ob_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above prior high or bullish 1d OB appears
            if high[i] > high_max_6[i] or bullish_ob_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals