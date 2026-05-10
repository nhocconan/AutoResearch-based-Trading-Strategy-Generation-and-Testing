# 1h_4H_Camarilla_R1_S1_Volume_Trend
# Hypothesis: Use 4h Camarilla pivot points (R1/S1) as dynamic support/resistance levels.
# Enter long when price breaks above R1 with volume confirmation in a 1d uptrend (price > EMA34).
# Enter short when price breaks below S1 with volume confirmation in a 1d downtrend (price < EMA34).
# Use 1h timeframe for precise entry timing, 4h for Camarilla levels, 1d for trend filter.
# Designed for low trade frequency (target: 60-150 trades over 4 years) to minimize fee drag.
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling breakdowns in downtrends.

#!/usr/bin/env python3
name = "1h_4H_Camarilla_R1_S1_Volume_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    r1_4h = close_4h + camarilla_range
    s1_4h = close_4h - camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume
            if close[i] > r1_4h_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume
            elif close[i] < s1_4h_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend fails
            if close[i] < s1_4h_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or trend fails
            if close[i] > r1_4h_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals