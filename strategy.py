#!/usr/bin/env python3
# 1d_1wCamarilla_R1S1_Breakout_1wEMA34_Trend_Volume
# Uses weekly Camarilla pivot levels (R1/S1) as breakout levels with weekly trend filter (EMA34)
# and weekly volume confirmation. Designed for 1d timeframe to capture major weekly pivot breaks
# with trend alignment. Works in both bull and bear markets by following the weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing.

name = "1d_1wCamarilla_R1S1_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels (R1/S1 - tighter range for fewer trades)
    r1 = pp + range_1w * 1.1 / 12
    s1 = pp - range_1w * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly volume filter (20-period MA) with threshold
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (2.0 * vol_ma_20)
    volume_spike_1d = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(volume_spike_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: break above R1 with uptrend and volume
            if close[i] > r1_1d[i] and close[i] > ema_34_1d[i] and volume_spike_1d[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S1 with downtrend and volume
            elif close[i] < s1_1d[i] and close[i] < ema_34_1d[i] and volume_spike_1d[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns to EMA34 or breaks below S1
            if bars_since_entry >= 2 and (close[i] < ema_34_1d[i] or close[i] < s1_1d[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to EMA34 or breaks above R1
            if bars_since_entry >= 2 and (close[i] > ema_34_1d[i] or close[i] > r1_1d[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals