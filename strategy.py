#!/usr/bin/env python3
# 4h_1d_Pivot_R1S1_Breakout_Volume_Trend
# Hypothesis: Trade breakouts from daily Camarilla R1/S1 levels on 4h timeframe with volume confirmation and 4h EMA trend filter.
# Works in bull and bear markets by using price action breakouts (direction agnostic) filtered by higher timeframe trend.

name = "4h_1d_Pivot_R1S1_Breakout_Volume_Trend"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1
    s1_1d = close_1d - (range_1d * 1.1 / 6)
    r1_1d = close_1d + (range_1d * 1.1 / 6)
    
    # Align daily levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # 4h EMA34 for trend filter
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_34[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above R1 with volume and trend confirmation
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_34[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with volume and trend confirmation
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1
            if close[i] < s1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1
            if close[i] > r1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals