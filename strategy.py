# 12h_Camarilla_Pivot_R1_S1_1dTrend_VolumeConfirm
# Hypothesis: 12h price action at Camarilla R1/S1 levels with daily trend and volume confirmation
# captures institutional-level reversals while minimizing false signals. Daily trend aligns with
# higher-timeframe momentum, volume confirms breakout strength, Camarilla provides precise levels.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "12h_Camarilla_Pivot_R1_S1_1dTrend_VolumeConfirm"
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

    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate weekly high/low for Camarilla pivot points (use previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's high/low/close for current week's Camarilla levels
    high_1w_prev = df_1w['high'].shift(1).values  # Previous week high
    low_1w_prev = df_1w['low'].shift(1).values    # Previous week low
    close_1w_prev = df_1w['close'].shift(1).values # Previous week close
    
    # Align weekly data to 12h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w_prev)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w_prev)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_prev)

    # Calculate Camarilla pivot levels for R1 and S1
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_1w_aligned - low_1w_aligned
    r1_level = close_1w_aligned + 1.1 * camarilla_range / 12.0
    s1_level = close_1w_aligned - 1.1 * camarilla_range / 12.0

    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1_level[i]) or np.isnan(s1_level[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches S1 level with volume spike and daily uptrend
            if low[i] <= s1_level[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 level with volume spike and daily downtrend
            elif high[i] >= r1_level[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 level or daily trend turns down
            if high[i] >= r1_level[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 level or daily trend turns up
            if low[i] <= s1_level[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals