#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla pivot levels (R1/S1) derived from 1d high-low-close,
# breaks above R1 for long, breaks below S1 for short, filtered by 1d EMA34 trend and volume spikes.
# 12h timeframe reduces trade frequency vs 4h to avoid fee drag, targeting 50-150 total trades over 4 years.
# Works in bull markets via breakout above R1 with uptrend and volume spike.
# Works in bear markets via breakdown below S1 with downtrend and volume spike.
# Volume confirmation ensures breakouts are genuine, not false breakouts.
# Trend filter ensures alignment with higher timeframe momentum.
# Designed for 12h to balance trade frequency and signal quality.

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike on 12h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 + above 1d EMA34 trend + volume spike
            if close[i] > r1_12h[i] and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + below 1d EMA34 trend + volume spike
            elif close[i] < s1_12h[i] and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < R1 or price below 1d EMA34
            if close[i] < r1_12h[i] or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > S1 or price above 1d EMA34
            if close[i] > s1_12h[i] or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals