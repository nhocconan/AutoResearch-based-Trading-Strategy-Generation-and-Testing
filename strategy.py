#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h trend direction (via EMA50) and 1d volume spike to filter 1h Camarilla R1/S1 breakouts.
# Enter long when price breaks above R1 with 4h EMA50 uptrend and 1d volume spike.
# Enter short when price breaks below S1 with 4h EMA50 downtrend and 1d volume spike.
# Exit when price returns to the Camarilla Pivot point (central level).
# This structure-based approach reduces false breakouts and works in both bull/bear via trend and volume filters.
# Target: 15-30 trades/year on 1h to minimize fee drag.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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

    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Camarilla pivot levels for 1h
    # Using previous bar's high, low, close
    ph = np.roll(high, 1)
    pl = np.roll(low, 1)
    pc = np.roll(close, 1)
    ph[0] = high[0]
    pl[0] = low[0]
    pc[0] = close[0]

    # Camarilla levels
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # Pivot = (H+L+C)/3
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    pivot = (ph + pl + pc) / 3

    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)  # Using 4h index for alignment (same length as 1h after alignment)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + 4h EMA50 uptrend + 1d volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.8):
                signals[i] = 0.20
                position = 1
            # SHORT: Close breaks below S1 + 4h EMA50 downtrend + 1d volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.8):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below pivot (return to value area)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close crosses back above pivot (return to value area)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals