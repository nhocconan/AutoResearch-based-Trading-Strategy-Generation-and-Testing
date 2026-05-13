# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Use Camarilla pivot levels from 1d with R1/S1 levels for breakout entries.
# Enter long when price breaks above R1 with 12h EMA50 uptrend and volume spike.
# Enter short when price breaks below S1 with 12h EMA50 downtrend and volume spike.
# Exit when price returns to the Camarilla pivot (CP) level.
# This structure-based approach reduces false breakouts and works in both bull/bear via trend filter.
# Target: 20-40 trades/year on 4h to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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

    # Calculate Camarilla pivot levels
    # CP = (high + low + close) / 3
    # R1 = CP + (high - low) * 1.1 / 12
    # S1 = CP - (high - low) * 1.1 / 12
    cp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = cp + range_1d * 1.1 / 12
    s1 = cp - range_1d * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    cp_aligned = align_htf_to_ltf(prices, df_1d, cp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(cp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + price > 12h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + price < 12h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below CP (return to pivot)
            if close[i] < cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above CP (return to pivot)
            if close[i] > cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals