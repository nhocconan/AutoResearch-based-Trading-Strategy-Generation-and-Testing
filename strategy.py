#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R3/S3) from 1d as breakout levels.
# Enter long when price breaks above R3 with 1d EMA uptrend and volume spike.
# Enter short when price breaks below S3 with 1d EMA downtrend and volume spike.
# Exit when price returns to the 1d close (mean reversion to daily equilibrium).
# Camarilla levels provide institutional support/resistance; 1d trend filters direction; volume confirms conviction.
# Target: 15-25 trades/year on 12h to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for 1d
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d

    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R3 + price > 1d EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 + price < 1d EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close returns to 1d close (mean reversion to daily equilibrium)
            if close[i] < close_1d[-1] if i == 0 else close[i] < close_1d[i // 24]:  # Simplified: use prior day's close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close returns to 1d close (mean reversion to daily equilibrium)
            if close[i] > close_1d[-1] if i == 0 else close[i] > close_1d[i // 24]:  # Simplified: use prior day's close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals