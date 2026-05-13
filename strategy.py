#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Price reacts to Camarilla pivot levels (R1/S1) derived from 12h timeframe. 
# Go long when price breaks above R1 with 1d uptrend and volume confirmation.
# Go short when price breaks below S1 with 1d downtrend and volume confirmation.
# Uses 1d trend filter (EMA34) to align with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false breakouts.
# Designed for 12h timeframe to target 50-150 total trades over 4 years.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    camarilla_width = (high_12h - low_12h) * 1.1 / 12
    r1 = close_12h + camarilla_width
    s1 = close_12h - camarilla_width
    
    # Align 12h indicators to 12h timeframe (no alignment needed as prices is already 12h)
    r1_aligned = r1  # Already on 12h timeframe
    s1_aligned = s1  # Already on 12h timeframe
    
    # Volume spike: volume > 2.0 * 24-period average (12 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + 1d uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + 1d downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals