#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla pivot breakouts with 1-week EMA34 trend filter and volume spike work in both bull and bear markets. Trend filter reduces whipsaw, volume confirms participation, and tight entry limits trades to avoid fee drag. Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # R1 = PP + (range * 1.1/12)
    # S1 = PP - (range * 1.1/12)
    # R2 = PP + (range * 1.1/6)
    # S2 = PP - (range * 1.1/6)
    # R3 = PP + (range * 1.1/4)
    # S3 = PP - (range * 1.1/4)
    # R4 = PP + (range * 1.1/2)
    # S4 = PP - (range * 1.1/2)
    
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    s2 = pp - (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    s3 = pp - (range_1d * 1.1 / 4)
    r4 = pp + (range_1d * 1.1 / 2)
    s4 = pp - (range_1d * 1.1 / 2)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_long = close[i] > r1[i-1]  # Break above R1
        breakout_short = close[i] < s1[i-1]  # Break below S1
        
        # Trend filter from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout (reverse position)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0