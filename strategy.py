#!/usr/bin/env python3
"""
4H_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R1 level with volume >2x average and 12h EMA50 trending up; sell when price breaks below Camarilla S1 level with volume >2x average and 12h EMA50 trending down. Uses Camarilla pivot levels from daily data for structure, volume confirmation, and trend filter to avoid false breakouts, targeting low trade frequency (<30/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

name = "4H_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using previous day's data to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have NaN due to roll, handled by isnan check later
    camarilla_range = prev_high_1d - prev_low_1d
    r1_level = prev_close_1d + camarilla_range * 1.1 / 12
    s1_level = prev_close_1d - camarilla_range * 1.1 / 12

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_level[i]) or np.isnan(s1_level[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 12h uptrend + volume spike
            if (close[i] > r1_level[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 12h downtrend + volume spike
            elif (close[i] < s1_level[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR trend turns down
            if close[i] < s1_level[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR trend turns up
            if close[i] > r1_level[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals